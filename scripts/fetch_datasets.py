from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator
from urllib.request import urlopen, Request


def get_dataset_dir() -> Path:
    base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if base_dir:
        base = Path(os.path.expanduser(base_dir))
    else:
        base = Path.home() / ".cache" / "nanochat"
    target = base / "datasets"
    target.mkdir(parents=True, exist_ok=True)
    return target


def iter_jsonl(url: str) -> Iterator[dict]:
    req = Request(url, headers={"User-Agent": "nanochat-fetch/1.0"})
    with urlopen(req, timeout=60) as resp:  # nosec - trusted endpoint you control
        for raw in resp:
            try:
                line = raw.decode("utf-8").strip()
            except Exception:
                continue
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield obj


def is_valid_sft(record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"system", "user", "assistant"}:
            return False
        if not isinstance(content, str) or not content.strip():
            return False
    return True


def is_valid_dpo(record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    prompt = record.get("prompt")
    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if not all(isinstance(x, str) and x.strip() for x in [prompt, chosen, rejected]):
        return False
    if chosen.strip() == rejected.strip():
        return False
    return True


def write_jsonl(path: Path, items: Iterator[dict]) -> int:
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch validated datasets from datagen server")
    parser.add_argument("--sft-url", required=False, default="https://datagen.stera.ventures/datasets/trader_sft_data.jsonl")
    parser.add_argument("--dpo-url", required=False, default="https://datagen.stera.ventures/datasets/trader_dpo_data.jsonl")
    parser.add_argument("--out-dir", required=False, default=None, help="Override output directory (defaults to NANOCHAT cache)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else get_dataset_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_path = out_dir / "trader_sft_data.jsonl"
    dpo_path = out_dir / "trader_dpo_data.jsonl"

    # Stream, validate, and write SFT
    sft_stream = (obj for obj in iter_jsonl(args.sft_url) if is_valid_sft(obj))
    sft_count = write_jsonl(sft_path, sft_stream)

    # Stream, validate, and write DPO
    dpo_stream = (obj for obj in iter_jsonl(args.dpo_url) if is_valid_dpo(obj))
    dpo_count = write_jsonl(dpo_path, dpo_stream)

    print(json.dumps({
        "ok": True,
        "out_dir": str(out_dir),
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
        "sft_count": sft_count,
        "dpo_count": dpo_count,
    }))


if __name__ == "__main__":
    main()


