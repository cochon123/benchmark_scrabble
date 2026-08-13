# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.2",
#   "bitsandbytes>=0.45",
#   "datasets>=3.2",
#   "huggingface-hub>=0.27",
#   "jinja2>=3.1",
#   "peft>=0.14",
#   "safetensors>=0.4",
#   "torch>=2.5",
#   "transformers>=4.48",
# ]
# ///
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


INPUT = Path("/input")
WORK = Path("/workspace")
SOURCE = WORK / "scrabble_ai"
DATA = SOURCE / "data" / "general_v3_sft"
OUTPUT = WORK / "scrabble-v3-adapter"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    print(f"[verify] {path.name} sha256={actual}", flush=True)
    if actual != expected:
        raise RuntimeError(f"Hash mismatch for {path}: {actual} != {expected}")


def extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(destination, filter="data")


def checkpoint_step(path: Path) -> int:
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state["global_step"])


def latest_checkpoint() -> Path | None:
    checkpoints = list(OUTPUT.glob("checkpoint-*"))
    return max(checkpoints, key=checkpoint_step) if checkpoints else None


def upload_json(api: HfApi, repo: str, payload: dict[str, Any], name: str) -> None:
    path = WORK / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=name,
        repo_id=repo,
        commit_message=f"Update v3 job state: {payload.get('state')}",
    )


def upload_checkpoint(api: HfApi, repo: str, checkpoint: Path) -> None:
    step = checkpoint_step(checkpoint)
    api.upload_folder(
        folder_path=str(checkpoint),
        path_in_repo=f"resume/checkpoint-{step}",
        repo_id=repo,
        commit_message=f"Upload resumable v3 checkpoint {step}",
    )


def train_segment(config: dict[str, Any], target: int, resume: Path | None) -> None:
    training = config["training"]
    command = [
        sys.executable,
        str(SOURCE / "scripts" / "train_qlora.py"),
        "--model", str(config["model"]),
        "--train-file", str(DATA / "train.jsonl"),
        "--validation-file", str(DATA / "validation.jsonl"),
        "--output-dir", str(OUTPUT),
        "--max-length", str(training["max_length"]),
        "--epochs", str(training["epochs"]),
        "--learning-rate", str(training["learning_rate"]),
        "--batch-size", str(training["batch_size"]),
        "--gradient-accumulation", str(training["gradient_accumulation"]),
        "--lora-r", str(training["lora_r"]),
        "--lora-alpha", str(training["lora_alpha"]),
        "--json-loss-weight", str(training["json_loss_weight"]),
        "--placement-loss-weight", str(training["placement_loss_weight"]),
        "--checkpoint-every", str(training["checkpoint_every"]),
        "--stop-after-step", str(target),
        "--save-total-limit", "3",
        "--logging-steps", "10",
    ]
    command.append(
        "--gradient-checkpointing"
        if bool(training.get("gradient_checkpointing", True))
        else "--no-gradient-checkpointing"
    )
    if resume is not None:
        command.extend(["--resume-from-checkpoint", str(resume)])
    if target < int(training["total_optimizer_steps"]):
        command.append("--skip-final-eval")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{SOURCE}{os.pathsep}{environment['PYTHONPATH']}"
        if environment.get("PYTHONPATH")
        else str(SOURCE)
    )
    print("[train] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=SOURCE, env=environment, check=True)


def main() -> None:
    started = time.time()
    manifest_path = INPUT / "v3_job_manifest.json"
    config = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo = str(config["output_repo"])
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, private=True, exist_ok=True)
    status: dict[str, Any] = {
        "state": "starting",
        "started_at_utc": utc_now(),
        "model": config["model"],
        "target_step": config["training"]["total_optimizer_steps"],
        "dataset_train_examples": config["dataset"]["train_examples"],
    }
    try:
        source_archive = INPUT / config["source_archive"]["filename"]
        data_archive = INPUT / config["data_archive"]["filename"]
        verify(source_archive, config["source_archive"]["sha256"])
        verify(data_archive, config["data_archive"]["sha256"])
        extract(source_archive, SOURCE)
        extract(data_archive, SOURCE / "data")
        verify(DATA / "manifest.json", config["dataset"]["manifest_sha256"])
        verify(DATA / "train.jsonl", config["dataset"]["train_sha256"])
        verify(DATA / "validation.jsonl", config["dataset"]["validation_sha256"])
        upload_json(api, repo, status, "job_status.json")

        total = int(config["training"]["total_optimizer_steps"])
        segment = int(config["training"]["segment_steps"])
        targets = list(range(segment, total, segment)) + [total]
        resume: Path | None = None
        for target in targets:
            status.update(state="training", step=checkpoint_step(resume) if resume else 0, target_segment=target)
            upload_json(api, repo, status, "job_status.json")
            train_segment(config, target, resume)
            candidate = OUTPUT / f"checkpoint-{target}"
            if target < total:
                if not candidate.exists() or checkpoint_step(candidate) != target:
                    raise RuntimeError(f"Expected resumable checkpoint {target} was not written")
                resume = candidate
                upload_checkpoint(api, repo, resume)
            status.update(
                state="segment_complete",
                step=target,
                elapsed_seconds=time.time() - started,
                estimated_cost_usd=(time.time() - started) / 3600 * float(config["cost_per_hour_usd"]),
            )
            upload_json(api, repo, status, "job_status.json")

        api.upload_folder(
            folder_path=str(OUTPUT),
            path_in_repo="final",
            allow_patterns=[
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer*",
                "special_tokens_map.json",
                "training_args.bin",
                "training_metrics*.json",
                "segment_metrics.json",
                "trainer_state.json",
            ],
            repo_id=repo,
            commit_message="Upload final Scrabble v3 adapter and metrics",
        )
        status.update(
            state="complete",
            step=total,
            completed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(config["cost_per_hour_usd"]),
        )
        upload_json(api, repo, status, "job_status.json")
        print(json.dumps(status, indent=2), flush=True)
    except Exception as error:
        status.update(
            state="error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            failed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(config.get("cost_per_hour_usd", 0)),
        )
        try:
            resume = latest_checkpoint()
            if resume is not None:
                status["latest_local_checkpoint"] = checkpoint_step(resume)
                upload_checkpoint(api, repo, resume)
            upload_json(api, repo, status, "job_status.json")
        finally:
            print(json.dumps(status, indent=2), flush=True)
        raise


if __name__ == "__main__":
    main()
