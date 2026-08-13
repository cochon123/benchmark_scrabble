#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import tarfile
import time
from collections.abc import Mapping
from pathlib import Path


ROOT = Path("/content/scrabble_ai")
SOURCE_ARCHIVE = Path("/content/v42_source.tar.gz")
DATA_ARCHIVE = Path("/content/v42_data.tar.gz")
OUTPUT = Path("/content/v42_colab_audit.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


def main() -> None:
    started = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    for path in (SOURCE_ARCHIVE, DATA_ARCHIVE):
        with tarfile.open(path, "r:gz") as archive:
            archive.extractall(ROOT, filter="data")
    sys.path.insert(0, str(ROOT))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    verification = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_v42_data.py")],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    verified = json.loads(verification.stdout)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51"], check=True)
        from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", use_fast=True)
    full_lengths, completion_lengths = [], []
    types: dict[str, int] = {}
    for filename in ("stage1.jsonl", "stage2.jsonl", "stage3.jsonl"):
        for line in (ROOT / "data" / "general_v42" / filename).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            types[row["record_type"]] = types.get(row["record_type"], 0) + 1
            encoded = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
            if isinstance(encoded, Mapping):
                encoded = encoded["input_ids"]
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            full_lengths.append(len(encoded))
            completion_lengths.append(len(tokenizer(row["messages"][-1]["content"], add_special_tokens=False)["input_ids"]))
    payload = {
        "passed": bool(verified["passed"]),
        "environment": "Google Colab CPU",
        "archives": {"source_sha256": sha256(SOURCE_ARCHIVE), "data_sha256": sha256(DATA_ARCHIVE)},
        "verification": verified,
        "records": len(full_lengths),
        "record_types": dict(sorted(types.items())),
        "token_lengths": {
            "full_p50": percentile(full_lengths, 0.50),
            "full_p95": percentile(full_lengths, 0.95),
            "full_p99": percentile(full_lengths, 0.99),
            "full_max": max(full_lengths),
            "completion_mean": statistics.mean(completion_lengths),
            "completion_p99": percentile(completion_lengths, 0.99),
            "completion_max": max(completion_lengths),
        },
        "elapsed_seconds": time.time() - started,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["passed"]:
        raise SystemExit("V4.2 Colab audit failed")


if __name__ == "__main__":
    main()
