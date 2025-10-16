from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# We will call into the existing generator script via subprocess to avoid importing heavy deps at module import time
import subprocess


app = FastAPI(title="Nanochat Data Generation Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STATE = {
    "running": False,
    "last_exit_code": None,  # type: Optional[int]
    "last_run_started_at": None,
    "last_run_finished_at": None,
}


def get_dataset_dir() -> Path:
    base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if base_dir:
        base = Path(os.path.expanduser(base_dir))
    else:
        base = Path.home() / ".cache" / "nanochat"
    datasets = base / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    return datasets


def desired_sizes() -> tuple[int, int]:
    # Environment-driven targets; defaults keep the loop active but modest
    sft_target = int(os.environ.get("SFT_TARGET", "0"))
    dpo_target = int(os.environ.get("DPO_TARGET", "0"))
    return sft_target, dpo_target


def current_line_counts() -> tuple[int, int]:
    data_dir = get_dataset_dir()
    sft_path = data_dir / "trader_sft_data.jsonl"
    dpo_path = data_dir / "trader_dpo_data.jsonl"
    def count_lines(p: Path) -> int:
        if not p.exists():
            return 0
        with p.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    return count_lines(sft_path), count_lines(dpo_path)


def run_generator_once(sft_to_add: int, dpo_to_add: int) -> int:
    """Run the generator script once to add the requested number of samples.

    Returns the process return code.
    """
    if sft_to_add <= 0 and dpo_to_add <= 0:
        return 0
    cmd = [
        "python",
        "/app/scripts/synthetic_data_gen.py",
        "--sft-examples",
        str(sft_to_add),
        "--dpo-examples",
        str(dpo_to_add),
    ]
    env = os.environ.copy()
    # Ensure dataset dir exists per NANOCHAT_BASE_DIR
    get_dataset_dir()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        # Surface error to container logs
        print(proc.stdout)
        print(proc.stderr)
    return proc.returncode


def generator_loop():
    STATE["running"] = True
    try:
        while True:
            target_sft, target_dpo = desired_sizes()
            cur_sft, cur_dpo = current_line_counts()

            add_sft = max(0, target_sft - cur_sft)
            add_dpo = max(0, target_dpo - cur_dpo)

            if add_sft > 0 or add_dpo > 0:
                STATE["last_run_started_at"] = time.time()
                rc = run_generator_once(add_sft, add_dpo)
                STATE["last_exit_code"] = rc
                STATE["last_run_finished_at"] = time.time()
            # Sleep a bit before checking again
            time.sleep(int(os.environ.get("DATAGEN_POLL_SECS", "60")))
    finally:
        STATE["running"] = False


# Mount static hosting of the datasets directory at /datasets
datasets_dir = get_dataset_dir()
app.mount("/datasets", StaticFiles(directory=str(datasets_dir)), name="datasets")


@app.get("/status")
def status():
    target_sft, target_dpo = desired_sizes()
    cur_sft, cur_dpo = current_line_counts()
    data_dir = str(get_dataset_dir())
    return {
        "running": STATE["running"],
        "targets": {"sft": target_sft, "dpo": target_dpo},
        "current": {"sft": cur_sft, "dpo": cur_dpo},
        "dataset_dir": data_dir,
        "sft_url": "/datasets/trader_sft_data.jsonl",
        "dpo_url": "/datasets/trader_dpo_data.jsonl",
        "last_exit_code": STATE["last_exit_code"],
        "last_run_started_at": STATE["last_run_started_at"],
        "last_run_finished_at": STATE["last_run_finished_at"],
    }


def _start_background_thread():
    t = threading.Thread(target=generator_loop, daemon=True)
    t.start()


@app.on_event("startup")
def on_startup():
    _start_background_thread()


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


