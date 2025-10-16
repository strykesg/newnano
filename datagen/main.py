from __future__ import annotations

import os
import threading
import time
from pathlib import Path
import shutil
import getpass
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# We will call into the existing generator script via subprocess to avoid importing heavy deps at module import time
import subprocess
from contextlib import asynccontextmanager


def get_poll_secs() -> int:
    try:
        return max(1, int(os.environ.get("DATAGEN_POLL_SECS", "60")))
    except Exception:
        return 60


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: kick off background thread
    _start_background_thread()
    yield
    # Shutdown: nothing required; daemon thread will exit with process


app = FastAPI(title="Nanochat Data Generation Service", lifespan=lifespan)
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
    "last_stdout": "",
    "last_stderr": "",
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


def desired_increments() -> tuple[int, int]:
    # Support two names for convenience
    inc_sft = int(os.environ.get("SFT_INCREMENT", os.environ.get("SFT_TARGET_INCREMENT", "0")))
    inc_dpo = int(os.environ.get("DPO_INCREMENT", os.environ.get("DPO_TARGET_INCREMENT", "0")))
    return max(0, inc_sft), max(0, inc_dpo)


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
    # Allow a max runtime per cycle to avoid long blocking runs
    max_seconds = int(os.environ.get("MAX_RUN_SECONDS", "600"))
    try:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=max_seconds)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            rc = 124  # timeout code
            print(f"[datagen] generator timed out after {max_seconds}s; will continue next cycle")
    except Exception as err:  # noqa: BLE001
        stdout, stderr = "", f"spawn error: {err}"
        rc = 1
    if rc != 0:
        # Surface error to container logs
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
    # Store limited logs for diagnostics
    STATE["last_stdout"] = (stdout or "")[-8000:]
    STATE["last_stderr"] = (stderr or "")[-8000:]
    return rc


def generator_loop():
    STATE["running"] = True
    try:
        while True:
            target_sft, target_dpo = desired_sizes()
            cur_sft, cur_dpo = current_line_counts()

            add_sft = max(0, target_sft - cur_sft)
            add_dpo = max(0, target_dpo - cur_dpo)

            inc_sft, inc_dpo = desired_increments()
            # If there is no remaining target work for a split, fall back to increments
            if add_sft == 0 and inc_sft > 0:
                add_sft = inc_sft
            if add_dpo == 0 and inc_dpo > 0:
                add_dpo = inc_dpo

            if add_sft > 0 or add_dpo > 0:
                STATE["last_run_started_at"] = time.time()
                print(f"[datagen] starting run: add_sft={add_sft} add_dpo={add_dpo} current=({cur_sft},{cur_dpo})")
                rc = run_generator_once(add_sft, add_dpo)
                STATE["last_exit_code"] = rc
                STATE["last_run_finished_at"] = time.time()
                print(f"[datagen] finished run: rc={rc}")
            else:
                print(
                    f"[datagen] no work: targets=({target_sft},{target_dpo}) current=({cur_sft},{cur_dpo}) "
                    f"incs=({inc_sft},{inc_dpo})"
                )
            # Sleep a bit before checking again
            time.sleep(get_poll_secs())
    finally:
        STATE["running"] = False


# Mount static hosting of the datasets directory at /datasets
datasets_dir = get_dataset_dir()
app.mount("/datasets", StaticFiles(directory=str(datasets_dir)), name="datasets")


@app.get("/")
def index():
    return {
        "message": "nanochat datagen",
        "status": "/status",
        "datasets": "/datasets/",
        "diagnostics": "/diagnostics",
    }


@app.get("/status")
def status():
    target_sft, target_dpo = desired_sizes()
    cur_sft, cur_dpo = current_line_counts()
    data_dir = str(get_dataset_dir())
    writable, write_error = test_write_access(get_dataset_dir())
    return {
        "running": STATE["running"],
        "targets": {"sft": target_sft, "dpo": target_dpo},
        "current": {"sft": cur_sft, "dpo": cur_dpo},
        "dataset_dir": data_dir,
        "dataset_dir_writable": writable,
        "dataset_dir_error": write_error,
        "sft_url": "/datasets/trader_sft_data.jsonl",
        "dpo_url": "/datasets/trader_dpo_data.jsonl",
        "last_exit_code": STATE["last_exit_code"],
        "last_run_started_at": STATE["last_run_started_at"],
        "last_run_finished_at": STATE["last_run_finished_at"],
        "poll_secs": get_poll_secs(),
    }


def mask(s: Optional[str]) -> str:
    if not s:
        return ""
    if len(s) <= 6:
        return "***"
    return s[:3] + "***" + s[-3:]


def test_write_access(path: Path) -> tuple[bool, Optional[str]]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe.tmp"
        with probe.open("w", encoding="utf-8") as f:
            f.write("ok")
        probe.unlink(missing_ok=True)
        return True, None
    except Exception as err:  # noqa: BLE001
        return False, str(err)


@app.get("/diagnostics")
def diagnostics():
    dataset = get_dataset_dir()
    writable, write_error = test_write_access(dataset)
    total, used, free = shutil.disk_usage(dataset)
    inc_sft, inc_dpo = desired_increments()
    env_info = {
        "OPENROUTER_API_KEY_present": bool(os.environ.get("OPENROUTER_API_KEY")),
        "OPENROUTER_API_KEY_masked": mask(os.environ.get("OPENROUTER_API_KEY")),
        "OPENROUTER_MODELS": os.environ.get("OPENROUTER_MODELS"),
        "OPENROUTER_REFERRER": os.environ.get("OPENROUTER_REFERRER"),
        "OPENROUTER_TITLE": os.environ.get("OPENROUTER_TITLE"),
        "SFT_TARGET": os.environ.get("SFT_TARGET", "0"),
        "DPO_TARGET": os.environ.get("DPO_TARGET", "0"),
        "SFT_INCREMENT": str(inc_sft),
        "DPO_INCREMENT": str(inc_dpo),
        "DATAGEN_POLL_SECS": str(get_poll_secs()),
    }
    return {
        "user": getpass.getuser(),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "dataset_dir": str(dataset),
        "dataset_dir_writable": writable,
        "dataset_dir_error": write_error,
        "disk": {"total": total, "used": used, "free": free},
        "env": env_info,
        "last_exit_code": STATE["last_exit_code"],
        "last_stdout_tail": STATE["last_stdout"],
        "last_stderr_tail": STATE["last_stderr"],
    }


def _start_background_thread():
    t = threading.Thread(target=generator_loop, daemon=True)
    t.start()


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


