# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.2",
#   "bitsandbytes>=0.45",
#   "datasets>=3.2",
#   "huggingface-hub>=1.0",
#   "jinja2>=3.1",
#   "peft>=0.14",
#   "safetensors>=0.4",
#   "torch>=2.6",
#   "transformers>=4.55",
# ]
# ///
from __future__ import annotations

import hashlib
import json
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
OUTPUT = WORK / "v3-probe-adapter"
PROBE_STEPS = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_status(api: HfApi, repo: str, status: dict[str, Any]) -> None:
    path = WORK / "probe_status.json"
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo="probe_status.json",
        repo_id=repo,
        commit_message=f"Update v3 probe status: {status['state']}",
    )


def main() -> None:
    started = time.time()
    manifest = json.loads((INPUT / "v3_job_manifest.json").read_text(encoding="utf-8"))
    repo = str(manifest["output_repo"])
    cost_per_hour = float(manifest["cost_per_hour_usd"])
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, private=True, exist_ok=True)
    status: dict[str, Any] = {
        "state": "starting",
        "started_at_utc": utc_now(),
        "probe_steps": PROBE_STEPS,
        "full_steps": int(manifest["training"]["total_optimizer_steps"]),
    }
    try:
        for archive_key, destination in (("source_archive", SOURCE), ("data_archive", SOURCE / "data")):
            archive = INPUT / str(manifest[archive_key]["filename"])
            if sha256(archive) != str(manifest[archive_key]["sha256"]):
                raise RuntimeError(f"{archive_key} hash mismatch")
            destination.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:gz") as handle:
                handle.extractall(destination, filter="data")
        data = SOURCE / "data/general_v3_sft"
        if sha256(data / "train.jsonl") != str(manifest["dataset"]["train_sha256"]):
            raise RuntimeError("Probe train data hash mismatch")
        upload_status(api, repo, status)
        training = manifest["training"]
        command = [
            sys.executable,
            str(SOURCE / "scripts/train_qlora.py"),
            "--model", str(manifest["model"]),
            "--train-file", str(data / "train.jsonl"),
            "--validation-file", str(data / "validation.jsonl"),
            "--output-dir", str(OUTPUT),
            "--max-length", str(training["max_length"]),
            "--max-steps", str(PROBE_STEPS),
            "--learning-rate", str(training["learning_rate"]),
            "--batch-size", str(training["batch_size"]),
            "--gradient-accumulation", str(training["gradient_accumulation"]),
            "--lora-r", str(training["lora_r"]),
            "--lora-alpha", str(training["lora_alpha"]),
            "--json-loss-weight", str(training["json_loss_weight"]),
            "--placement-loss-weight", str(training["placement_loss_weight"]),
            "--checkpoint-every", str(PROBE_STEPS),
            "--logging-steps", "5",
            "--save-total-limit", "1",
            "--skip-final-eval",
        ]
        command.append(
            "--gradient-checkpointing"
            if bool(training.get("gradient_checkpointing", True))
            else "--no-gradient-checkpointing"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE)
        status["state"] = "running"
        upload_status(api, repo, status)
        print("[probe] " + " ".join(command), flush=True)
        subprocess.run(command, cwd=SOURCE, env=environment, check=True)
        metrics_path = OUTPUT / "training_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        seconds_per_step = float(metrics["train_runtime"]) / int(metrics["global_step"])
        projected_training_seconds = seconds_per_step * int(training["total_optimizer_steps"])
        api.upload_folder(
            folder_path=str(OUTPUT),
            repo_id=repo,
            allow_patterns=["training_metrics*.json", "segment_metrics.json", "trainer_state.json"],
            commit_message="Upload optimized v3 L40S probe metrics",
        )
        status.update(
            state="complete",
            completed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * cost_per_hour,
            metrics=metrics,
            projected_training_seconds=projected_training_seconds,
            projected_training_cost_usd=projected_training_seconds / 3600 * cost_per_hour,
        )
    except Exception as error:
        status.update(
            state="error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            failed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * cost_per_hour,
        )
        raise
    finally:
        upload_status(api, repo, status)
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
