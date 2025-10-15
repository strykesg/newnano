"""Utility for generating trader persona datasets via OpenRouter.

This script produces two datasets used in the Chimera hypertraining plan:
- Supervised fine-tuning (SFT) conversations
- Direct preference optimisation (DPO) preference pairs

Both datasets are emitted as JSONL files under the nanochat cache directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration constants

DEFAULT_MODEL_CANDIDATES = [
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-flash-1.5",
]
SYSTEM_PERSONA = (
    "You are Chimera, a legendary multi-market trader blending quantitative rigor, "
    "macro intuition, and disciplined risk management. You mentor a motivated novice, "
    "keeping tone sharp yet encouraging, and always grounding advice in actionable "
    "tradecraft."
)
DPO_GUIDANCE = (
    "You are generating preference data contrasting Chimera's polished guidance with a "
    "flawed apprentice take. The chosen answer must showcase disciplined risk framing, "
    "clear structure, and persona-consistent voice. The rejected answer should contain "
    "at least one substantive mistake (e.g., mispricing, poor risk control, vague plan) "
    "while remaining coherent."
)
USER_SCENARIOS = [
    "evaluating a swing trade on {ticker} before earnings",
    "deciding whether to fade the momentum on {ticker} after a {percent}% gap up",
    "allocating capital between growth tech names and energy hedges for the next quarter",
    "managing downside risk on a crowded long in {ticker} given rising credit spreads",
    "constructing a delta-neutral options play on {ticker} ahead of macro data",
    "reviewing whether to pyramid into an outperforming position in {ticker}",
    "building a multi-asset view that mixes {ticker} with short-dated Treasuries",
    "navigating position sizing while volatility is spiking in {ticker}",
]
FOLLOW_UP_ANGLES = [
    "I am unsure how to frame the risk limits. Could you walk through the guardrails?",
    "What indicators would you monitor to confirm the thesis is still valid?",
    "What do I tell the desk if liquidity dries up suddenly?",
    "How would you stress-test the position before green-lighting it?",
    "What would make you abandon this plan entirely?",
]

# -----------------------------------------------------------------------------
# Helper dataclasses


@dataclass
class GenerationConfig:
    sft_examples: int
    dpo_examples: int
    temperature: float = 0.9
    dpo_temperature: float = 0.8
    max_tokens: int = 700
    dpo_max_tokens: int = 900
    seed: int = 1337


# -----------------------------------------------------------------------------
# Utility helpers


def get_dataset_dir() -> Path:
    base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if base_dir:
        base = Path(os.path.expanduser(base_dir))
    else:
        base = Path.home() / ".cache" / "nanochat"
    return base / "datasets"


def resolve_model_candidates() -> List[str]:
    env_value = os.environ.get("OPENROUTER_MODELS")
    if env_value:
        candidates = [item.strip() for item in env_value.split(",") if item.strip()]
        if candidates:
            return candidates
    return DEFAULT_MODEL_CANDIDATES.copy()


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY must be set to use synthetic data generation")
    headers = {
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERRER", "https://github.com/eureka-labs/nanochat"),
        "X-Title": os.environ.get("OPENROUTER_TITLE", "nanochat-chimera"),
    }
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers=headers,
    )


def unwrap_json(text: str) -> Dict[str, str]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[len("json") :]
        candidate = candidate.strip()
    return json.loads(candidate)


def call_model(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    max_retries: int = 6,
    backoff: float = 2.0,
) -> str:
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
            )
            content = completion.choices[0].message.content
            if isinstance(content, list):
                # Some models return list of content blocks
                return "".join(block.get("text", "") for block in content)
            return content or ""
        except RateLimitError as error:
            wait = backoff * (2 ** attempt) + random.random()
            tqdm.write(f"Rate limit from model {model}: {error}. Sleeping {wait:.1f}s")
            time.sleep(wait)
        except (APIError, APITimeoutError) as error:
            wait = backoff * (2 ** attempt) + random.random()
            tqdm.write(f"API error from model {model}: {error}. Sleeping {wait:.1f}s")
            time.sleep(wait)
        except KeyboardInterrupt:
            raise
        except Exception as error:
            wait = backoff * (2 ** attempt) + random.random()
            tqdm.write(f"Unexpected error from model {model}: {error}. Sleeping {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to obtain completion after {max_retries} retries")


def find_working_model(client: OpenAI, candidates: Iterable[str]) -> str:
    probe_messages = [
        {"role": "system", "content": "Respond with the word READY."},
        {"role": "user", "content": "Say READY if you can see this."},
    ]
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            reply = call_model(
                client,
                candidate,
                probe_messages,
                temperature=0.0,
                max_tokens=5,
                max_retries=2,
                backoff=1.0,
            )
            if reply:
                return candidate
        except Exception as error:  # noqa: BLE001
            last_error = error
            tqdm.write(f"Model {candidate} unavailable: {error}")
    raise RuntimeError(f"Unable to use any candidate models. Last error: {last_error}")


def random_ticker() -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(random.choice(letters) for _ in range(random.choice([3, 4])))


def random_percent() -> int:
    return random.choice([3, 5, 7, 9, 12, 15])


def build_user_prompt() -> str:
    scenario_template = random.choice(USER_SCENARIOS)
    scenario = scenario_template.format(ticker=random_ticker(), percent=random_percent())
    follow_up = random.choice(FOLLOW_UP_ANGLES)
    intro = (
        "I'm a junior trader trying to sharpen my process. "
        "Here's the situation I'm watching: "
    )
    body = textwrap.fill(scenario.capitalize(), width=88)
    follow = textwrap.fill(follow_up, width=88)
    return f"{intro}{body}\n\n{follow}"


def normalise_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    normalised = []
    for message in messages:
        content = message.get("content", "")
        normalised.append({
            "role": message.get("role", "assistant"),
            "content": content.strip(),
        })
    return normalised


def generate_sft_example(client: OpenAI, model: str, config: GenerationConfig) -> Dict[str, List[Dict[str, str]]]:
    user_prompt = build_user_prompt()
    assistant = call_model(
        client,
        model,
        [
            {"role": "system", "content": SYSTEM_PERSONA},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    messages = normalise_messages([
        {"role": "system", "content": SYSTEM_PERSONA},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant},
    ])
    return {"messages": messages}


def generate_dpo_example(client: OpenAI, model: str, config: GenerationConfig) -> Dict[str, str]:
    user_prompt = build_user_prompt()
    response_text = call_model(
        client,
        model,
        [
            {"role": "system", "content": DPO_GUIDANCE},
            {
                "role": "user",
                "content": textwrap.dedent(
                    f"""
                    Craft two contrasting answers for Chimera's apprentice training log.
                    Use the novice's question below and output JSON with keys `chosen` and `rejected`.
                    - `chosen`: Chimera's refined guidance, showcasing disciplined structuring, clear risk plan, and persona voice.
                    - `rejected`: a believable but flawed apprentice attempt with at least one material mistake.
                    Keep each answer under 12 sentences.
                    Return ONLY valid JSON.

                    Novice question:
                    {user_prompt}
                    """
                ).strip(),
            },
        ],
        temperature=config.dpo_temperature,
        max_tokens=config.dpo_max_tokens,
    )
    payload = unwrap_json(response_text)
    chosen = payload.get("chosen") or payload.get("master") or payload.get("preferred")
    rejected = payload.get("rejected") or payload.get("apprentice") or payload.get("rejected_response")
    if not chosen or not rejected:
        raise ValueError("Model response missing chosen/rejected fields")
    chosen = chosen.strip()
    rejected = rejected.strip()
    if chosen == rejected:
        raise ValueError("Chosen and rejected answers are identical")
    return {
        "prompt": user_prompt.strip(),
        "chosen": chosen,
        "rejected": rejected,
    }


# -----------------------------------------------------------------------------
# Main driver


def generate_dataset(
    writer,
    total: int,
    generator,
    description: str,
):
    progress = tqdm(total=total, desc=description, unit="sample")
    produced = 0
    failures = 0
    while produced < total:
        try:
            record = generator()
        except KeyboardInterrupt:
            raise
        except Exception as error:  # noqa: BLE001
            failures += 1
            tqdm.write(f"Failed to generate {description.lower()} sample: {error}")
            time.sleep(min(30, 1 + 0.5 * failures))
            continue
        json.dump(record, writer)
        writer.write("\n")
        produced += 1
        progress.update(1)
        if produced % 50 == 0:
            writer.flush()
    progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic trader datasets via OpenRouter")
    parser.add_argument("--sft-examples", type=int, default=0, help="Number of SFT conversations to generate")
    parser.add_argument("--dpo-examples", type=int, default=0, help="Number of DPO preference pairs to generate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    config = GenerationConfig(
        sft_examples=args.sft_examples,
        dpo_examples=args.dpo_examples,
        seed=args.seed,
    )
    if config.sft_examples <= 0 and config.dpo_examples <= 0:
        raise ValueError("At least one of --sft-examples or --dpo-examples must be positive")

    random.seed(config.seed)

    client = build_client()
    candidates = resolve_model_candidates()
    tqdm.write(f"Candidate models: {', '.join(candidates)}")
    model = find_working_model(client, candidates)
    tqdm.write(f"Using OpenRouter model: {model}")

    dataset_dir = get_dataset_dir()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sft_path = dataset_dir / "trader_sft_data.jsonl"
    dpo_path = dataset_dir / "trader_dpo_data.jsonl"

    if config.sft_examples > 0:
        tqdm.write(f"Writing {config.sft_examples} SFT conversations to {sft_path}")
        with sft_path.open("w", encoding="utf-8") as writer:
            generate_dataset(
                writer,
                config.sft_examples,
                lambda: generate_sft_example(client, model, config),
                "Trader SFT",
            )
    if config.dpo_examples > 0:
        tqdm.write(f"Writing {config.dpo_examples} DPO pairs to {dpo_path}")
        with dpo_path.open("w", encoding="utf-8") as writer:
            generate_dataset(
                writer,
                config.dpo_examples,
                lambda: generate_dpo_example(client, model, config),
                "Trader DPO",
            )


if __name__ == "__main__":
    main()
