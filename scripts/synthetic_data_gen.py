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
    "You are Chimera, an apex predator in the financial markets. Your trading style is a "
    "lethal fusion of first-principles thinking, probabilistic modeling, and "
    "unrelenting discipline. You see the market as a complex adaptive system, "
    "exploiting inefficiencies across crypto, equities, and macro instruments. You are "
    "mentoring a sharp but inexperienced junior trader. Your tone is surgical and "
    "demanding, yet deeply invested in their growth. Every piece of advice must be "
    "rooted in a clear framework of asymmetric payoffs, rigorous risk definition, and "
    "emotional detachment. Prioritize clarity over complexity, process over P&L."
)
DPO_GUIDANCE = (
    "You are generating preference data to teach an AI the difference between elite and "
    "amateur trading advice. The 'chosen' response must embody Chimera's persona: "
    "structured, risk-first, and clear. The 'rejected' response must represent a "
    "common but critical trading mistake. Ensure the rejected answer is still plausible "
    "but contains at least one clear flaw from the following categories:\n"
    "- **Flawed Thesis:** Relies on narrative, confirmation bias, or misinterprets data (e.g., 'everyone on Twitter is bullish').\n"
    "- **Poor Risk Management:** Vague or non-existent stop-loss, oversized position, revenge trading mentality, or ignores volatility.\n"
    "- **Vague/Unactionable Plan:** Lacks specific entry, exit, or invalidation points (e.g., 'buy the dip and wait').\n"
    "- **Behavioral Flaws:** Exhibits FOMO, panic, overconfidence, or an attachment to a losing position."
)
# --- Scenarios are broken down by theme for maximum diversity ---

# Crypto-Native & On-Chain Scenarios
CRYPTO_SCENARIOS = [
    "analyzing the sustainability of the {ticker} perpetual futures funding rate before taking a position",
    "deciding if {ticker} is a good spot buy based on exchange inflow/outflow data and whale wallet tracking",
    "evaluating a potential airdrop farming strategy for a new L2, weighing the gas costs against the expected return",
    "building a thesis around the {ticker} token unlock schedule for the next 90 days",
    "assessing the risk/reward of providing liquidity to a {ticker}/{stablecoin} pair on a DEX, considering impermanent loss",
    "fading the parabolic move in a new memecoin {ticker} that just did a {percent}% run",
    "structuring a trade based on the relative value between {ticker_a} and its liquid staking derivative {ticker_b}",
    "monitoring on-chain data to front-run a potential narrative shift into the {sector} ecosystem (e.g., DePIN, RWA, SocialFi)",
]

# TradFi, Macro & Cross-Market Scenarios
TRADFI_MACRO_SCENARIOS = [
    "hedging my crypto longs in {ticker} with shorts in the Nasdaq ({index_ticker}) ahead of CPI data",
    "allocating capital between a high-beta crypto play like {ticker} and short-duration T-bills",
    "assessing the impact of a surprise Fed rate hike on Bitcoin dominance and altcoin valuations",
    "managing a crowded long in {ticker} as credit spreads begin to widen",
    "constructing a view on {ticker} by analyzing the price of oil ({oil_ticker}) and the DXY ({dxy_ticker})",
    "positioning for the quarterly options/futures expiry (Quad Witching) and its effect on {ticker}",
]

# Options & Volatility Scenarios
OPTIONS_VOL_SCENARIOS = [
    "designing a delta-neutral options strategy on {ticker} to profit from a volatility crush after its earnings call",
    "buying protective puts on my spot {ticker} holdings as implied volatility is hitting yearly lows",
    "selling covered calls against a long-term {ticker} position to generate yield in a sideways market",
    "legging into a bull call spread on {ticker} to get cheap upside exposure with defined risk",
    "analyzing the term structure and skew of {ticker} options to determine market sentiment and positioning",
]

# Execution, Sizing & Psychology Scenarios
EXECUTION_PSYCHOLOGY_SCENARIOS = [
    "developing a framework for pyramiding into a winning position in {ticker} without moving my average entry too high",
    "navigating a significant drawdown in my portfolio and deciding whether to cut losers or wait for a reversion",
    "determining the correct position size for a trade on {ticker} given its ATR and my portfolio's risk limits",
    "how to handle a situation where my thesis for {ticker} is right but my timing is wrong and I'm underwater",
    "building a system to avoid FOMO-ing into {ticker} after a {percent}% gap up on news",
]

# Combine all scenarios for the script
USER_SCENARIOS = (
    CRYPTO_SCENARIOS
    + TRADFI_MACRO_SCENARIOS
    + OPTIONS_VOL_SCENARIOS
    + EXECUTION_PSYCHOLOGY_SCENARIOS
)

FOLLOW_UP_ANGLES = [
    # Risk & Invalidation
    "Walk me through the exact conditions that would invalidate this thesis. What specific price level or data point makes us cut the trade?",
    "How do we define our risk here? Is it a percentage of the portfolio, a hard dollar amount, or based on the asset's volatility?",
    "What's the 'pain threshold' on this trade? At what point do we reduce size even if our stop isn't hit?",
    "How would you hedge this position if we get a market-wide risk-off event?",

    # Thesis & Confirmation
    "What are the top 3 on-chain metrics you'd watch to confirm our thesis on this is playing out?",
    "Who is on the other side of this trade, and why might they be right?",
    "What's the primary catalyst we're playing for here, and what's the expected timeline?",

    # Execution & Management
    "How should I scale into this position? All at once, or in thirds on specific levels?",
    "What does the order execution look like? Are we using TWAP, limit orders, or market orders to build the position?",
    "What tools are you using to monitor the key variables for this trade in real-time?",
    "How do I stick to the plan when the P&L is deep red but the thesis is still intact?"
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
        # Emit a newline-terminated progress line so logs stream line-by-line in containers
        print(f"[gen] produced {produced}/{total} for {description}")
        writer.flush()
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
        # Append so datasets grow over time across runs
        with sft_path.open("a", encoding="utf-8") as writer:
            generate_dataset(
                writer,
                config.sft_examples,
                lambda: generate_sft_example(client, model, config),
                "Trader SFT",
            )
    if config.dpo_examples > 0:
        tqdm.write(f"Writing {config.dpo_examples} DPO pairs to {dpo_path}")
        # Append so datasets grow over time across runs
        with dpo_path.open("a", encoding="utf-8") as writer:
            generate_dataset(
                writer,
                config.dpo_examples,
                lambda: generate_dpo_example(client, model, config),
                "Trader DPO",
            )


if __name__ == "__main__":
    main()
