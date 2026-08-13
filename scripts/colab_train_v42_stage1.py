#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


SOURCE_ARCHIVE = Path("/content/v42_source.tar.gz")
DATA_ARCHIVE = Path("/content/v42_data.tar.gz")
ADAPTER_ARCHIVE = Path("/content/v41_stage2_adapter.tar.gz")
TRAIN_SCRIPT = Path("/content/train_qlora.py")
ROOT = Path("/content/scrabble_ai")
INIT_ADAPTER = Path("/content/v41_stage2")
OUTPUT = Path("/content/v42_stage1")
STATUS = Path("/content/v42_stage1_status.json")
FINAL_ARCHIVE = Path("/content/v42_stage1_adapter.tar.gz")
EXPECTED = {
    SOURCE_ARCHIVE: "619aff5ad7fea32e4b58a1e3387ec08b371aa8c0c2ca1f66d946a66cefb8ad81",
    DATA_ARCHIVE: "b60f069d7d55d43832e73f90d8a98a23acb398c86825cfb4eed15658affc3a8d",
    ADAPTER_ARCHIVE: "d996495477c51ca3d60ece9580ad6adb25554d9913fd8ed6e371354454dd8782",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.time()
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {path}: {actual} != {expected}")
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")
    with tarfile.open(DATA_ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")
    INIT_ADAPTER.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ADAPTER_ARCHIVE, "r:gz") as archive:
        archive.extractall(INIT_ADAPTER, filter="data")
    (ROOT / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRAIN_SCRIPT, ROOT / "scripts/train_qlora.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "accelerate>=1.2",
            "bitsandbytes>=0.45",
            "datasets>=3.2",
            "jinja2>=3.1",
            "peft>=0.14",
            "safetensors>=0.4",
            "transformers>=4.51",
        ],
        check=True,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    command = [
        sys.executable,
        str(ROOT / "scripts/train_qlora.py"),
        "--model",
        "Qwen/Qwen3-4B-Instruct-2507",
        "--init-adapter",
        str(INIT_ADAPTER),
        "--train-file",
        str(ROOT / "data/general_v42/stage1.jsonl"),
        "--validation-file",
        str(ROOT / "data/general_v42/validation.jsonl"),
        "--output-dir",
        str(OUTPUT),
        "--max-length",
        "1216",
        "--max-steps",
        "500",
        "--learning-rate",
        "3e-5",
        "--batch-size",
        "1",
        "--gradient-accumulation",
        "8",
        "--checkpoint-every",
        "100",
        "--logging-steps",
        "10",
        "--save-total-limit",
        "5",
        "--skip-final-eval",
        "--seed",
        "7421",
    ]
    print(json.dumps({"command": command, "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()}, indent=2), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    keep = [path for path in OUTPUT.iterdir() if path.is_file()]
    with tarfile.open(FINAL_ARCHIVE, "w:gz") as archive:
        for path in sorted(keep):
            archive.add(path, arcname=path.name)
    metrics = json.loads((OUTPUT / "training_metrics.json").read_text(encoding="utf-8"))
    status = {
        "state": "complete",
        "environment": "Google Colab T4",
        "elapsed_seconds": time.time() - started,
        "training": metrics,
        "input_hashes": {str(path): expected for path, expected in EXPECTED.items()},
        "output_archive_sha256": sha256(FINAL_ARCHIVE),
        "output_archive_bytes": FINAL_ARCHIVE.stat().st_size,
    }
    STATUS.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
