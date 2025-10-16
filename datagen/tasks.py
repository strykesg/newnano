from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

import requests

from .celery_app import celery_app


def _post_ingest(record: dict) -> None:
    url = os.environ.get("INGEST_URL") or f"http://127.0.0.1:{os.environ.get('PORT','8981')}/ingest"
    try:
        resp = requests.post(url, json=record, timeout=30)
        resp.raise_for_status()
    except Exception as err:  # noqa: BLE001
        raise RuntimeError(f"failed to ingest record: {err}")


def _run_generator_once(sft: int, dpo: int) -> None:
    if sft <= 0 and dpo <= 0:
        return
    cmd = [
        "python",
        "/app/scripts/synthetic_data_gen.py",
        "--sft-examples",
        str(int(sft)),
        "--dpo-examples",
        str(int(dpo)),
        "--emit-to-stdout",
        "--quiet",
    ]
    env = os.environ.copy()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        _post_ingest(obj)
    proc.wait()


@celery_app.task(name="datagen.generate_batch", max_retries=3, default_retry_delay=10)
def generate_batch(sft: int = 0, dpo: int = 0) -> dict:
    _run_generator_once(sft, dpo)
    return {"sft": int(sft), "dpo": int(dpo)}


