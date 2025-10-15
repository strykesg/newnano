"""Trader persona supervised fine-tuning dataset loader."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List

from tasks.common import Task


def _default_dataset_path() -> Path:
    base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if base_dir:
        base = Path(os.path.expanduser(base_dir))
    else:
        base = Path.home() / ".cache" / "nanochat"
    return base / "datasets" / "trader_sft_data.jsonl"


class TraderSFT(Task):
    """Loads Chimera SFT conversations and exposes deterministic splits."""

    def __init__(
        self,
        *,
        split: str = "train",
        data_path: str | os.PathLike[str] | None = None,
        val_fraction: float = 0.1,
        seed: int = 1234,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert split in {"train", "val"}, "split must be 'train' or 'val'"
        if not (0.0 < val_fraction < 1.0):
            raise ValueError("val_fraction must be within (0, 1)")
        path = Path(data_path) if data_path else _default_dataset_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Trader SFT dataset not found at {path}. "
                "Run scripts.synthetic_data_gen to create it."
            )
        self._examples: List[dict] = []
        with path.open("r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                messages = payload.get("messages")
                if not isinstance(messages, list):
                    raise ValueError("Each JSON line must contain a messages list")
                self._examples.append({"messages": messages})
        if not self._examples:
            raise ValueError(f"Trader SFT dataset at {path} is empty")
        indices = list(range(len(self._examples)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        split_index = max(1, int(round(len(indices) * (1.0 - val_fraction))))
        train_indices = indices[:split_index]
        val_indices = indices[split_index:]
        if not val_indices:
            val_indices = train_indices[-1:]
            train_indices = train_indices[:-1]
        selected = train_indices if split == "train" else val_indices
        self._data = [self._examples[i] for i in selected]

    def num_examples(self) -> int:  # type: ignore[override]
        return len(self._data)

    def get_example(self, index: int):  # type: ignore[override]
        return self._data[index]
