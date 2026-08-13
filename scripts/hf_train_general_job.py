#!/usr/bin/env python3
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

from huggingface_hub import HfApi


INPUT = Path("/input")
WORK = Path("/workspace")
SOURCE = WORK / "scrabble_ai"
OUTPUT = WORK / "qwen3-4b-scrabble-general-lora"
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
OUTPUT_REPO = "Cochon123/Qwen3-4B-Scrabble-General-v2-work"
EXPECTED_HASHES = {
    "scrabble_general_source_v2.tar.gz": "0b07f0bea8695b1b62249164775ffb5558a3243cab07ed430cbd3f0fc0d11b98",
    "scrabble_general_data_v2.tar.gz": "a19d0706bc91dce5e98a1d1092dd45a1d2115dc729b886a548849558bd023325",
    "scrabble_checkpoint_100.tar.gz": "5c3632fff24a8cec326fcc8ddbba65895e8526df5f10f62dfaefc8fc1bcd61cc",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs() -> None:
    for filename, expected in EXPECTED_HASHES.items():
        path = INPUT / filename
        actual = sha256(path)
        print(f"[input] {filename} sha256={actual}", flush=True)
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {filename}: {actual}")


def extract_inputs() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(INPUT / "scrabble_general_source_v2.tar.gz", "r:gz") as archive:
        archive.extractall(SOURCE, filter="data")
    data_destination = SOURCE / "data"
    data_destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(INPUT / "scrabble_general_data_v2.tar.gz", "r:gz") as archive:
        archive.extractall(data_destination, filter="data")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(INPUT / "scrabble_checkpoint_100.tar.gz", "r:gz") as archive:
        archive.extractall(OUTPUT, filter="data")


def verify_dataset() -> dict[str, object]:
    root = SOURCE / "data" / "general_v2_sft"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checks: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        expected = manifest["splits"][split]
        jsonl = root / f"{split}.jsonl"
        positions = root / f"{split}_positions.json"
        records = sum(1 for _ in jsonl.open(encoding="utf-8"))
        passed = (
            records == int(expected["records"])
            and sha256(jsonl) == expected["jsonl_sha256"]
            and sha256(positions) == expected["positions_sha256"]
        )
        checks[split] = {"records": records, "passed": passed}
    audit = json.loads((root / "source_audit.json").read_text(encoding="utf-8"))
    checks["source_audit_passed"] = bool(audit["passed"])
    checks["passed"] = bool(audit["passed"]) and all(
        bool(checks[split]["passed"]) for split in ("train", "validation", "test")
    )
    if not checks["passed"]:
        raise RuntimeError(f"Dataset verification failed: {checks}")
    print(json.dumps({"dataset_verification": checks}, indent=2), flush=True)
    return checks


def checkpoint_step(path: Path) -> int:
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state["global_step"])


def latest_checkpoint() -> Path:
    checkpoints = sorted(
        OUTPUT.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    if not checkpoints:
        raise RuntimeError("No checkpoint found")
    return checkpoints[-1]


def write_status(payload: dict[str, object]) -> Path:
    path = WORK / "job_status.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def upload_status(api: HfApi, status: dict[str, object]) -> None:
    path = write_status(status)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo="job_status.json",
        repo_id=OUTPUT_REPO,
        repo_type="model",
        commit_message=f"Update training status: {status.get('state')} step {status.get('step')}",
    )


def upload_checkpoint(api: HfApi, checkpoint: Path, step: int) -> None:
    api.upload_folder(
        folder_path=str(checkpoint),
        path_in_repo=f"resume/checkpoint-{step}",
        repo_id=OUTPUT_REPO,
        repo_type="model",
        commit_message=f"Archive resumable optimizer checkpoint {step}",
    )


def upload_selection_checkpoint(api: HfApi, step: int) -> None:
    checkpoint = OUTPUT / f"checkpoint-{step}"
    if not checkpoint.exists():
        raise RuntimeError(f"Selection checkpoint {step} is missing")
    allowed = {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "trainer_state.json",
    }
    api.upload_folder(
        folder_path=str(checkpoint),
        path_in_repo=f"selection/checkpoint-{step}",
        allow_patterns=sorted(allowed),
        repo_id=OUTPUT_REPO,
        repo_type="model",
        commit_message=f"Preserve validation-selection checkpoint {step}",
    )


def train_segment(target: int, resume: Path) -> None:
    command = [
        sys.executable,
        str(SOURCE / "scripts" / "train_qlora.py"),
        "--model",
        MODEL_ID,
        "--train-file",
        str(SOURCE / "data" / "general_v2_sft" / "train.jsonl"),
        "--validation-file",
        str(SOURCE / "data" / "general_v2_sft" / "validation.jsonl"),
        "--output-dir",
        str(OUTPUT),
        "--max-length",
        "2816",
        "--lora-r",
        "16",
        "--lora-alpha",
        "32",
        "--epochs",
        "1",
        "--learning-rate",
        "0.0001",
        "--gradient-accumulation",
        "4",
        "--logging-steps",
        "25",
        "--checkpoint-every",
        "50",
        "--save-total-limit",
        "30",
        "--stop-after-step",
        str(target),
        "--resume-from-checkpoint",
        str(resume),
    ]
    if target < 1265:
        command.append("--skip-final-eval")
    print(f"[train] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=SOURCE, check=True)


def upload_final(api: HfApi) -> None:
    root_files = [path for path in OUTPUT.iterdir() if path.is_file()]
    for path in root_files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"final/{path.name}",
            repo_id=OUTPUT_REPO,
            repo_type="model",
            commit_message=f"Upload final training artifact {path.name}",
        )
    upload_selection_checkpoint(api, 1250)


def main() -> None:
    started = time.time()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret is required")
    api = HfApi(token=token)
    status: dict[str, object] = {
        "state": "starting",
        "step": 100,
        "started_at_utc": utc_now(),
        "model": MODEL_ID,
        "source_checkpoint": 100,
        "target_step": 1265,
        "hardware": "t4-small",
        "cost_per_hour_usd": 0.40,
    }
    try:
        verify_inputs()
        extract_inputs()
        status["dataset_verification"] = verify_dataset()
        resume = latest_checkpoint()
        if checkpoint_step(resume) != 100:
            raise RuntimeError(f"Initial checkpoint is not step 100: {resume}")
        upload_status(api, status)

        selection_250_uploaded = False
        for target in list(range(200, 1201, 100)) + [1265]:
            status.update(state="training", step=checkpoint_step(resume), target_segment=target)
            upload_status(api, status)
            train_segment(target, resume)
            if target <= 1200:
                resume = OUTPUT / f"checkpoint-{target}"
                if checkpoint_step(resume) != target:
                    raise RuntimeError(f"Checkpoint {target} was not written correctly")
                upload_checkpoint(api, resume, target)
            if target >= 300 and not selection_250_uploaded:
                upload_selection_checkpoint(api, 250)
                selection_250_uploaded = True
            if target == 1000:
                upload_selection_checkpoint(api, 1000)
            status.update(
                state="segment_complete",
                step=target,
                elapsed_seconds=time.time() - started,
                estimated_cost_usd=(time.time() - started) / 3600 * 0.40,
            )
            upload_status(api, status)

        upload_final(api)
        status.update(
            state="complete",
            step=1265,
            completed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * 0.40,
        )
        upload_status(api, status)
        print(json.dumps(status, indent=2), flush=True)
    except Exception as error:
        status.update(
            state="error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            failed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * 0.40,
        )
        try:
            upload_status(api, status)
        finally:
            print(json.dumps(status, indent=2), flush=True)
        raise


if __name__ == "__main__":
    main()
