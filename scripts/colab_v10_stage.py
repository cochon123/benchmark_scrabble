#!/usr/bin/env python3
"""Crash-tolerant stage runner used through the Google Colab CLI."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path("/content/v10")
CONFIG = Path("/content/v10_stage.json")
MODEL = "Qwen/Qwen3-4B-Thinking-2507"


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["HF_HOME"] = "/content/hf_cache"
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def setup() -> None:
    archive = Path("/content/v10_bundle.tar.gz")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as source:
        source.extractall(ROOT, filter="data")
    print(json.dumps({"action": "setup", "root": str(ROOT), "files": len(list(ROOT.rglob("*")))}))


def restore(archive_name: str) -> None:
    archive = Path("/content") / archive_name
    destination = ROOT / "artifacts/v10_colab"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as source:
        source.extractall(destination, filter="data")
    print(
        json.dumps(
            {
                "action": "restore",
                "archive": str(archive),
                "adapter_files": len(list((destination / "adapter").rglob("*"))),
            }
        )
    )


def restore_parts(prefix: str, expected_sha256: str) -> None:
    parts = sorted(Path("/content").glob(f"{prefix}*"))
    if not parts:
        raise FileNotFoundError(f"No checkpoint parts for {prefix}")
    archive = Path("/content/adapter_restore.tar.gz")
    digest = hashlib.sha256()
    with archive.open("wb") as output:
        for part in parts:
            payload = part.read_bytes()
            output.write(payload)
            digest.update(payload)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(f"Checkpoint hash mismatch: {observed} != {expected_sha256}")
    restore(archive.name)


def evaluate(
    label: str,
    adapter: str | None,
    *,
    start_index: int = 0,
    boards: int | None = None,
) -> None:
    output = ROOT / "artifacts/v10_colab" / f"{label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_hf.py"),
        "--model",
        MODEL,
        "--dataset",
        str(ROOT / "data/v10_compact_reasoning/gate_positions.json"),
        "--output",
        str(output),
        "--max-new-tokens",
        "512",
        "--max-attempts",
        "1",
        "--batch-size",
        "2",
        "--board-encoding",
        "dense",
        "--seed",
        "10101",
        "--start-index",
        str(start_index),
    ]
    if boards is not None:
        command.extend(["--boards", str(boards)])
    if adapter:
        command.extend(["--adapter", adapter])
    run(command)


def train(stop_after: int, resume: str | None) -> None:
    output = ROOT / "artifacts/v10_colab/adapter"
    command = [
        sys.executable,
        str(ROOT / "scripts/train_qlora.py"),
        "--model",
        MODEL,
        "--train-file",
        str(ROOT / "data/v10_compact_reasoning/train.jsonl"),
        "--validation-file",
        str(ROOT / "data/v10_compact_reasoning/validation.jsonl"),
        "--output-dir",
        str(output),
        "--max-length",
        "1536",
        "--max-steps",
        "200",
        "--stop-after-step",
        str(stop_after),
        "--learning-rate",
        "4e-5",
        "--batch-size",
        "1",
        "--gradient-accumulation",
        "4",
        "--lora-r",
        "16",
        "--lora-alpha",
        "32",
        "--json-loss-weight",
        "1.5",
        "--placement-loss-weight",
        "2.0",
        "--checkpoint-every",
        "25",
        "--logging-steps",
        "5",
        "--save-total-limit",
        "8",
        "--skip-final-eval",
        "--gradient-checkpointing",
        "--seed",
        "10101",
    ]
    if resume:
        command.extend(["--resume-from-checkpoint", resume])
    run(command)
    exports = Path("/content/v10_exports")
    exports.mkdir(exist_ok=True)
    export = exports / f"adapter_step_{stop_after}.tar.gz"
    with tarfile.open(export, "w:gz") as target:
        for relative in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "trainer_state.json",
            "training_metrics.json",
            "segment_metrics.json",
        ):
            path = output / relative
            if path.exists():
                target.add(path, arcname=f"adapter/{relative}")
        checkpoint = output / f"checkpoint-{stop_after}"
        if checkpoint.exists():
            target.add(checkpoint, arcname=f"adapter/checkpoint-{stop_after}")
    print(json.dumps({"export": str(export), "bytes": export.stat().st_size}), flush=True)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    action = config["action"]
    if action == "setup":
        setup()
    elif action == "restore":
        restore(str(config["archive_name"]))
    elif action == "restore_parts":
        restore_parts(str(config["prefix"]), str(config["sha256"]))
    elif action == "evaluate":
        evaluate(
            str(config["label"]),
            config.get("adapter"),
            start_index=int(config.get("start_index", 0)),
            boards=(int(config["boards"]) if config.get("boards") is not None else None),
        )
    elif action == "train":
        train(int(config["stop_after"]), config.get("resume"))
    else:
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    main()
