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
SOURCE_ARCHIVE = Path("/content/v4_source.tar.gz")
DATA_ARCHIVE = Path("/content/v4_data.tar.gz")
OUTPUT = Path("/content/v4_colab_audit.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


def main() -> None:
    started = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")
    with tarfile.open(DATA_ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")
    sys.path.insert(0, str(ROOT))

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    verification = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_v4_pilot_data.py"),
            "--data-dir",
            str(ROOT / "data" / "general_v4_pilot"),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    verification_payload = json.loads(verification.stdout)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51"],
            check=True,
        )
        from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", use_fast=True)
    prompt_completion_lengths: list[int] = []
    completion_lengths: list[int] = []
    types: dict[str, int] = {}
    for filename in ("train.jsonl", "validation.jsonl"):
        path = ROOT / "data" / "general_v4_pilot" / filename
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            record_type = str(row["record_type"])
            types[record_type] = types.get(record_type, 0) + 1
            messages = row["messages"]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            if isinstance(encoded, Mapping):
                encoded = encoded["input_ids"]
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            prompt_completion_lengths.append(len(encoded))
            completion_lengths.append(
                len(tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"])
            )

    result = {
        "passed": bool(verification_payload["passed"]),
        "environment": "Google Colab CPU",
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "archives": {
            "source_sha256": sha256(SOURCE_ARCHIVE),
            "data_sha256": sha256(DATA_ARCHIVE),
        },
        "verification": verification_payload,
        "records": len(prompt_completion_lengths),
        "record_types": dict(sorted(types.items())),
        "token_lengths": {
            "full_p50": percentile(prompt_completion_lengths, 0.50),
            "full_p95": percentile(prompt_completion_lengths, 0.95),
            "full_p99": percentile(prompt_completion_lengths, 0.99),
            "full_max": max(prompt_completion_lengths),
            "completion_mean": statistics.mean(completion_lengths),
            "completion_p99": percentile(completion_lengths, 0.99),
            "completion_max": max(completion_lengths),
            "fits_1280": max(prompt_completion_lengths) <= 1280,
            "completion_fits_128": max(completion_lengths) <= 128,
        },
        "elapsed_seconds": time.time() - started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"] or not result["token_lengths"]["completion_fits_128"]:
        raise SystemExit("V4 Colab audit failed")


if __name__ == "__main__":
    main()
