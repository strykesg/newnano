"""Direct Preference Optimisation (DPO) finetuning for nanochat."""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from nanochat.common import DummyWandb, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.report import get_report

# -----------------------------------------------------------------------------
# Configurable defaults (override via configurator CLI)

run = "dummy"
dtype = "bfloat16"
device_batch_size = 4
num_epochs = 1
max_iterations = -1
target_examples_per_step = 32
beta = 0.1
max_length = 2048
shuffle_seed = 1234
dpo_dataset_path = None
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 1.0

config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Chimera, a legendary multi-market trader blending quantitative rigor, "
    "macro intuition, and disciplined risk management. You mentor a motivated novice, "
    "keeping tone sharp yet encouraging, and always grounding advice in actionable tradecraft."
)

PAD_TOKEN = "<|assistant_end|>"


@dataclass
class PreferenceExample:
    chosen_ids: List[int]
    chosen_mask: List[int]
    rejected_ids: List[int]
    rejected_mask: List[int]


def _default_dataset_path() -> Path:
    base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if base_dir:
        base = Path(os.path.expanduser(base_dir))
    else:
        base = Path.home() / ".cache" / "nanochat"
    return base / "datasets" / "trader_dpo_data.jsonl"


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _build_conversation(prompt: str, completion: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    }


def _tokenize_pair(tokenizer, prompt: str, chosen: str, rejected: str, *, max_tokens: int) -> PreferenceExample:
    chosen_conv = _build_conversation(prompt, chosen)
    rejected_conv = _build_conversation(prompt, rejected)
    chosen_ids, chosen_mask = tokenizer.render_conversation(chosen_conv, max_tokens=max_tokens)
    rejected_ids, rejected_mask = tokenizer.render_conversation(rejected_conv, max_tokens=max_tokens)
    if sum(chosen_mask) == 0 or sum(rejected_mask) == 0:
        raise ValueError("Conversation truncation removed assistant supervision tokens")
    return PreferenceExample(chosen_ids, chosen_mask, rejected_ids, rejected_mask)


def load_preferences(tokenizer, *, max_tokens: int) -> List[PreferenceExample]:
    dataset_path = Path(dpo_dataset_path) if dpo_dataset_path else _default_dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Trader DPO dataset not found at {dataset_path}. "
            "Run scripts.synthetic_data_gen to create it."
        )
    examples: List[PreferenceExample] = []
    for row in _load_jsonl(dataset_path):
        prompt = row.get("prompt", "").strip()
        chosen = row.get("chosen", "").strip()
        rejected = row.get("rejected", "").strip()
        if not prompt or not chosen or not rejected:
            continue
        examples.append(_tokenize_pair(tokenizer, prompt, chosen, rejected, max_tokens=max_tokens))
    if not examples:
        raise ValueError(f"No valid DPO examples loaded from {dataset_path}")
    return examples


def _prepare_batch(sequences: List[List[int]], masks: List[List[int]], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(sequences)
    lengths = [len(seq) - 1 for seq in sequences]
    max_len = max(lengths)
    inputs = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len), -1, dtype=torch.long)
    for row, (seq, mask) in enumerate(zip(sequences, masks)):
        # shift by one token for LM targets
        inp = torch.tensor(seq[:-1], dtype=torch.long)
        tgt = torch.tensor(seq[1:], dtype=torch.long)
        supervision = torch.tensor(mask[1:], dtype=torch.bool)
        tgt = tgt.masked_fill(~supervision, -1)
        inputs[row, : inp.size(0)] = inp
        targets[row, : tgt.size(0)] = tgt
    return inputs, targets


def _collate_batch(batch: List[PreferenceExample], tokenizer) -> dict:
    pad_token_id = tokenizer.encode_special(PAD_TOKEN)
    chosen_inputs, chosen_targets = _prepare_batch(
        [sample.chosen_ids for sample in batch],
        [sample.chosen_mask for sample in batch],
        pad_token_id,
    )
    rejected_inputs, rejected_targets = _prepare_batch(
        [sample.rejected_ids for sample in batch],
        [sample.rejected_mask for sample in batch],
        pad_token_id,
    )
    return {
        "chosen_inputs": chosen_inputs,
        "chosen_targets": chosen_targets,
        "rejected_inputs": rejected_inputs,
        "rejected_targets": rejected_targets,
    }


def preference_loader(examples: List[PreferenceExample], tokenizer, batch_size: int, *, world_size: int, rank: int):
    order = list(range(len(examples)))
    rng = random.Random(shuffle_seed)
    while True:
        rng.shuffle(order)
        local_indices = order[rank::world_size]
        batch: List[PreferenceExample] = []
        for idx in local_indices:
            batch.append(examples[idx])
            if len(batch) == batch_size:
                yield _collate_batch(batch, tokenizer)
                batch = []
        if batch:
            yield _collate_batch(batch, tokenizer)
            batch = []


def conversation_logprobs(model, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size = inputs.size(0)
    losses = model(inputs, targets=targets, loss_reduction='none')
    losses = losses.view(batch_size, -1)
    mask = (targets != -1).to(losses.dtype)
    log_probs = -(losses * mask).sum(dim=1)
    return log_probs


def main() -> None:
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    autocast_dtype = dtype_map[dtype]
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
    model, tokenizer, _meta = load_model("sft", device, phase="train")
    reference_model = copy.deepcopy(model)
    reference_model.to(device)
    for param in reference_model.parameters():
        param.requires_grad_(False)
    reference_model.eval()

    max_tokens = min(max_length, model.config.sequence_len)
    examples = load_preferences(tokenizer, max_tokens=max_tokens)
    num_examples = len(examples)
    print0(f"Loaded {num_examples} DPO preference pairs")

    optimizers = model.setup_optimizers(
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
    )

    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["lr"] * init_lr_frac
            group["initial_lr"] = group["lr"]

    examples_per_step = device_batch_size * ddp_world_size
    assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples_per_step"
    grad_accum_steps = target_examples_per_step // examples_per_step

    steps_per_epoch = math.ceil(num_examples / target_examples_per_step)
    num_iterations = steps_per_epoch * num_epochs
    if max_iterations > 0:
        num_iterations = min(num_iterations, max_iterations)
    if num_iterations == 0:
        num_iterations = 1

    train_loader = preference_loader(
        examples,
        tokenizer,
        device_batch_size,
        world_size=ddp_world_size,
        rank=ddp_rank,
    )

    use_dummy_wandb = run == "dummy" or ddp_rank != 0
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-dpo", name=run, config=user_config, save_code=True)

    running_loss = 0.0
    running_pref = 0.0
    running_margin = 0.0
    total_tokens = 0

    for step in range(num_iterations):
        model.train()
        batch_losses = []
        batch_pref = []
        batch_margin = []
        batch_tokens = 0
        for micro_step in range(grad_accum_steps):
            batch = next(train_loader)
            chosen_inputs = batch["chosen_inputs"].to(device)
            chosen_targets = batch["chosen_targets"].to(device)
            rejected_inputs = batch["rejected_inputs"].to(device)
            rejected_targets = batch["rejected_targets"].to(device)
            active_tokens = int((chosen_targets != -1).sum().item() + (rejected_targets != -1).sum().item())
            with autocast_ctx:
                chosen_logprobs = conversation_logprobs(model, chosen_inputs, chosen_targets)
                rejected_logprobs = conversation_logprobs(model, rejected_inputs, rejected_targets)
            with torch.no_grad(), autocast_ctx:
                ref_chosen = conversation_logprobs(reference_model, chosen_inputs, chosen_targets)
                ref_rejected = conversation_logprobs(reference_model, rejected_inputs, rejected_targets)
            policy_diff = chosen_logprobs - rejected_logprobs
            ref_diff = ref_chosen - ref_rejected
            advantages = beta * (policy_diff - ref_diff)
            loss = -F.logsigmoid(advantages)
            preference_accuracy = (policy_diff > 0).float().mean()
            margin = policy_diff.mean()
            mean_loss = loss.mean()
            (mean_loss / grad_accum_steps).backward()
            batch_losses.append(mean_loss.detach())
            batch_pref.append(preference_accuracy.detach())
            batch_margin.append(margin.detach())
            batch_tokens += active_tokens
        if ddp:
            for tensor in batch_losses + batch_pref + batch_margin:
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            batch_tokens_tensor = torch.tensor([batch_tokens], dtype=torch.long, device=device)
            dist.all_reduce(batch_tokens_tensor, op=dist.ReduceOp.SUM)
            batch_tokens = batch_tokens_tensor.item()
        mean_loss_item = torch.stack(batch_losses).mean().item()
        pref_item = torch.stack(batch_pref).mean().item()
        margin_item = torch.stack(batch_margin).mean().item()

        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        running_loss = 0.9 * running_loss + 0.1 * mean_loss_item if step > 0 else mean_loss_item
        running_pref = 0.9 * running_pref + 0.1 * pref_item if step > 0 else pref_item
        running_margin = 0.9 * running_margin + 0.1 * margin_item if step > 0 else margin_item
        total_tokens += batch_tokens

        print0(
            f"Step {step:05d}/{num_iterations:05d} | loss {mean_loss_item:.6f} | pref_acc {pref_item:.4f} | margin {margin_item:.4f} | tokens {batch_tokens:,}"
        )
        wandb_run.log({
            "step": step,
            "loss": mean_loss_item,
            "preference_accuracy": pref_item,
            "margin": margin_item,
            "tokens": batch_tokens,
        })

    if ddp_rank == 0:
        base_dir = get_base_dir()
        depth = model.config.n_layer
        checkpoint_dir = os.path.join(base_dir, "chatdpo_checkpoints", f"d{depth}")
        metadata = {
            "step": num_iterations,
            "loss": running_loss,
            "preference_accuracy": running_pref,
            "margin": running_margin,
            "tokens": total_tokens,
            "beta": beta,
            "model_config": model.config.__dict__,
        }
        save_checkpoint(checkpoint_dir, num_iterations, model.state_dict(), None, metadata)
        print0(f"✅ Saved DPO checkpoint to {checkpoint_dir}")
        get_report().log(
            section="Chat DPO",
            data=[
                user_config,
                {
                    "Iterations": num_iterations,
                    "Loss": running_loss,
                    "Preference accuracy": running_pref,
                    "Margin": running_margin,
                    "Tokens": total_tokens,
                },
            ],
        )

    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
