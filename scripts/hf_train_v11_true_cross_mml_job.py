# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.2", "bitsandbytes>=0.45", "datasets>=3.2",
#   "huggingface-hub>=1.0", "jinja2>=3.1", "peft>=0.14",
#   "safetensors>=0.4", "torch>=2.6", "transformers>=4.55",
# ]
# ///
"""Run the frozen V11 true-cross multi-positive experiment on Hugging Face."""
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
WORK = Path("/workspace/v11_true_cross_mml")
SOURCE = WORK / "source"
ADAPTERS = WORK / "adapters"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE)
    environment["HF_HOME"] = str(WORK / "hf_cache")
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=SOURCE, env=environment, check=True)


def upload_json(api: HfApi, repo: str, payload: Any, name: str, message: str) -> None:
    path = WORK / name.replace("/", "-")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(path), path_in_repo=name, repo_id=repo, commit_message=message
    )


def upload_results(api: HfApi, repo: str, output: Path) -> None:
    api.upload_folder(
        folder_path=str(output),
        path_in_repo="stage1d_true_cross_mml",
        repo_id=repo,
        allow_patterns=[
            "adapter_config.json", "adapter_model.safetensors", "tokenizer*",
            "trainer_state.json", "training_metrics*.json", "segment_metrics.json",
        ],
        commit_message="Upload V11 true-cross MML adapter",
    )


def main() -> None:
    started = time.time()
    manifest = json.loads((INPUT / "job_manifest.json").read_text(encoding="utf-8"))
    repo = str(manifest["output_repo"])
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, private=True, exist_ok=True)
    status: dict[str, Any] = {"state": "starting", "started_at_utc": now(), "manifest": manifest}
    try:
        for filename, expected in manifest["input_sha256"].items():
            observed = sha256(INPUT / filename)
            if observed != expected:
                raise RuntimeError(f"Input hash mismatch for {filename}: {observed} != {expected}")
        SOURCE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(INPUT / "source.tar.gz", "r:gz") as archive:
            archive.extractall(SOURCE, filter="data")
        ADAPTERS.mkdir(parents=True, exist_ok=True)
        with tarfile.open(INPUT / "stage1d_adapter.tar.gz", "r:gz") as archive:
            archive.extractall(ADAPTERS, filter="data")
        init_adapter = ADAPTERS / "stage1d_anchor" / "checkpoint-25"
        if not (init_adapter / "adapter_model.safetensors").exists():
            raise FileNotFoundError(init_adapter)

        output = WORK / "stage1d_true_cross_mml"
        status["state"] = "training"
        upload_json(api, repo, status, "status.json", "Start V11 true-cross MML training")
        run([
            sys.executable, str(SOURCE / "scripts/train_qlora.py"),
            "--model", str(manifest["model"]),
            "--init-adapter", str(init_adapter),
            "--train-file", str(SOURCE / "data/v11_true_cross_mml/train.jsonl"),
            "--validation-file", str(SOURCE / "data/v11_true_cross_mml/validation.jsonl"),
            "--output-dir", str(output), "--max-length", "1280",
            "--max-steps", str(manifest["max_steps"]), "--epochs", "1",
            "--learning-rate", str(manifest["learning_rate"]),
            "--batch-size", "1", "--gradient-accumulation", "4",
            "--lora-r", "16", "--lora-alpha", "32", "--checkpoint-every", "25",
            "--logging-steps", "5", "--save-total-limit", "6", "--skip-final-eval",
            "--gradient-checkpointing", "--disable-thinking", "--multi-positive",
            "--multi-positive-count", "8", "--seed", str(manifest["seed"]),
        ])
        status["training"] = json.loads((output / "training_metrics.json").read_text())
        upload_results(api, repo, output)

        evaluation_dir = WORK / "evaluations"
        evaluation_dir.mkdir(exist_ok=True)
        gate_output = evaluation_dir / "true-cross-gate.json"
        status["state"] = "evaluating_true_cross_gate"
        upload_json(api, repo, status, "status.json", "Evaluate V11 true-cross MML gate")
        run([
            sys.executable, str(SOURCE / "scripts/evaluate_v11_policy.py"),
            "--model", str(manifest["model"]), "--adapter", str(output),
            "--dataset", str(SOURCE / "data/v11_true_cross_mml/gate.json"),
            "--output", str(gate_output), "--categories", "true_cross",
            "--limit-per-category", "400", "--batch-size", "8", "--max-new-tokens", "16",
            "--max-attempts", "1", "--sample-attempts", "0", "--seed", str(manifest["seed"]),
        ])
        status["true_cross_gate"] = json.loads(gate_output.read_text())
        api.upload_file(path_or_fileobj=str(gate_output), path_in_repo="evaluations/true-cross-gate.json", repo_id=repo, commit_message="Upload true-cross gate")
        status["state"] = "completed"
    except Exception as error:
        status["state"] = "failed"
        status["error"] = repr(error)
        status["traceback"] = traceback.format_exc()
        raise
    finally:
        status["finished_at_utc"] = now()
        status["elapsed_seconds"] = time.time() - started
        upload_json(api, repo, status, "status.json", "Update V11 true-cross MML status")


if __name__ == "__main__":
    main()
