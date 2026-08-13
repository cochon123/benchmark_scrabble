#!/usr/bin/env python3
"""Crash-tolerant V11 curriculum runner for the Google Colab CLI."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path("/content/v11")
CONFIG = Path("/content/v11_stage.json")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["HF_HOME"] = "/content/hf_cache"
    print("[run] " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    log = completed.stdout + "\n" + completed.stderr
    Path("/content/v11_last_command.log").write_text(log, encoding="utf-8")
    if log:
        print(log[-20000:], flush=True)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def setup() -> None:
    archive = Path("/content/v11_bundle.tar.gz")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as source:
        source.extractall(ROOT, filter="data")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "accelerate==1.14.0",
            "bitsandbytes==0.50.0",
            "datasets==5.0.1",
            "peft==0.20.0",
            "transformers==5.14.1",
        ],
        check=True,
    )
    print(json.dumps({"action": "setup", "files": len(list(ROOT.rglob("*")))}), flush=True)


def restore_parts(prefix: str, expected_sha256: str) -> None:
    parts = sorted(Path("/content").glob(f"{prefix}*"))
    if not parts:
        raise FileNotFoundError(prefix)
    archive = Path("/content/v11_restore.tar.gz")
    digest = hashlib.sha256()
    with archive.open("wb") as output:
        for part in parts:
            payload = part.read_bytes()
            output.write(payload)
            digest.update(payload)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("Checkpoint archive SHA256 mismatch")
    destination = ROOT / "artifacts/v11_colab"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as source:
        source.extractall(destination, filter="data")
    print(json.dumps({"action": "restore", "parts": len(parts)}), flush=True)


def train(
    stage: str,
    data_stage: str,
    stop_after: int,
    resume: str | None,
    init_adapter: str | None,
    gradient_accumulation: int,
    max_steps: int,
    learning_rate: float,
    seed: int,
    train_file: str | None,
    validation_file: str | None,
    epochs: float,
    multi_positive: bool,
    multi_positive_count: int | None,
) -> None:
    output = ROOT / "artifacts/v11_colab" / stage
    command = [
        sys.executable,
        str(ROOT / "scripts/train_qlora.py"),
        "--model",
        MODEL,
        "--train-file",
        str(ROOT / (train_file or f"data/v11_atomic_policy/{data_stage}_train.jsonl")),
        "--validation-file",
        str(ROOT / (validation_file or f"data/v11_atomic_policy/{data_stage}_validation.jsonl")),
        "--output-dir",
        str(output),
        "--max-length",
        "1280",
        "--max-steps",
        str(max_steps),
        "--epochs",
        str(epochs),
        "--stop-after-step",
        str(stop_after),
        "--learning-rate",
        str(learning_rate),
        "--batch-size",
        "1",
        "--gradient-accumulation",
        str(gradient_accumulation),
        "--lora-r",
        "16",
        "--lora-alpha",
        "32",
        "--checkpoint-every",
        "25",
        "--logging-steps",
        "5",
        "--save-total-limit",
        "8",
        "--skip-final-eval",
        "--gradient-checkpointing",
        "--disable-thinking",
        "--seed",
        str(seed),
    ]
    if multi_positive:
        command.append("--multi-positive")
    if multi_positive_count is not None:
        command.extend(["--multi-positive-count", str(multi_positive_count)])
    if resume:
        command.extend(["--resume-from-checkpoint", resume])
    elif init_adapter:
        command.extend(["--init-adapter", init_adapter])
    run(command)
    export_adapter(stage, stop_after)


def export_adapter(stage: str, step: int) -> None:
    source = ROOT / "artifacts/v11_colab" / stage
    exports = Path("/content/v11_exports")
    exports.mkdir(exist_ok=True)
    archive = exports / f"{stage}_step_{step}.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        for relative in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "trainer_state.json",
            "training_metrics.json",
            "segment_metrics.json",
        ):
            path = source / relative
            if path.exists():
                target.add(path, arcname=f"{stage}/{relative}")
        checkpoint = source / f"checkpoint-{step}"
        if checkpoint.exists():
            target.add(checkpoint, arcname=f"{stage}/checkpoint-{step}")
        else:
            for relative in ("adapter_config.json", "adapter_model.safetensors"):
                path = source / relative
                if path.exists():
                    target.add(path, arcname=f"{stage}/{relative}")
    print(json.dumps({"export": str(archive), "bytes": archive.stat().st_size}), flush=True)


def evaluate(
    label: str,
    adapter: str | None,
    categories: list[str],
    limit: int,
    attempts: int,
    dataset: str | None,
    sample_attempts: int,
    temperature: float,
    sample_batch_size: int,
) -> None:
    output = ROOT / "artifacts/v11_colab/evaluations" / f"{label}.json"
    retry_output = ROOT / "artifacts/v11_colab/evaluations" / f"{label}_retry_success.jsonl"
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_v11_policy.py"),
        "--model",
        MODEL,
        "--dataset",
        str(ROOT / (dataset or "data/v11_atomic_policy/frozen_gate.json")),
        "--output",
        str(output),
        "--retry-output",
        str(retry_output),
        "--batch-size",
        "8",
        "--max-new-tokens",
        "16",
        "--max-attempts",
        str(attempts),
        "--sample-attempts",
        str(sample_attempts),
        "--temperature",
        str(temperature),
        "--sample-batch-size",
        str(sample_batch_size),
        "--limit-per-category",
        str(limit),
        "--categories",
        *categories,
    ]
    if adapter:
        command.extend(["--adapter", adapter])
    run(command)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    action = config["action"]
    if action == "setup":
        setup()
    elif action == "restore_parts":
        restore_parts(str(config["prefix"]), str(config["sha256"]))
    elif action == "train":
        train(
            str(config["stage"]),
            str(config.get("data_stage", config["stage"])),
            int(config["stop_after"]),
            config.get("resume"),
            config.get("init_adapter"),
            int(config.get("gradient_accumulation", 4)),
            int(config.get("max_steps", 200)),
            float(config.get("learning_rate", 1e-4)),
            int(config.get("seed", 11001)),
            config.get("train_file"),
            config.get("validation_file"),
            float(config.get("epochs", 1.0)),
            bool(config.get("multi_positive", False)),
            config.get("multi_positive_count"),
        )
    elif action == "evaluate":
        evaluate(
            str(config["label"]),
            config.get("adapter"),
            list(config["categories"]),
            int(config.get("limit", 100)),
            int(config.get("attempts", 1)),
            config.get("dataset"),
            int(config.get("sample_attempts", 0)),
            float(config.get("temperature", 0.8)),
            int(config.get("sample_batch_size", 2)),
        )
    else:
        raise ValueError(action)


if __name__ == "__main__":
    main()
