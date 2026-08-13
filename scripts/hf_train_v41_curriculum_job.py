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
MODEL_REPO = Path("/model")
WORK = Path("/workspace")
SOURCE = WORK / "scrabble_ai"
DATA = SOURCE / "data" / "general_v41"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_json(api: HfApi, repo: str, value: Any, repo_path: str, message: str) -> None:
    path = WORK / Path(repo_path).name
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(path), path_in_repo=repo_path, repo_id=repo, commit_message=message)


def run(command: list[str], env: dict[str, str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=SOURCE, env=env, check=True)


def train_stage(stage: dict[str, Any], init_adapter: Path, output: Path, env: dict[str, str]) -> dict[str, Any]:
    run(
        [
            sys.executable, str(SOURCE / "scripts" / "train_qlora.py"),
            "--model", "Qwen/Qwen3-4B-Instruct-2507",
            "--init-adapter", str(init_adapter),
            "--train-file", str(DATA / stage["file"]),
            "--validation-file", str(DATA / "validation.jsonl"),
            "--output-dir", str(output),
            "--max-length", "1856",
            "--max-steps", str(stage["steps"]),
            "--learning-rate", str(stage["learning_rate"]),
            "--batch-size", "1", "--gradient-accumulation", "8",
            "--checkpoint-every", "100", "--logging-steps", "10",
            "--save-total-limit", "2", "--skip-final-eval",
            "--no-gradient-checkpointing", "--seed", "6419",
        ],
        env,
    )
    return json.loads((output / "training_metrics.json").read_text(encoding="utf-8"))


def evaluate(adapter: Path, dataset: str, output: Path, env: dict[str, str]) -> dict[str, Any]:
    run(
        [
            sys.executable, str(SOURCE / "scripts" / "evaluate_v41.py"),
            "--model", "Qwen/Qwen3-4B-Instruct-2507",
            "--adapter", str(adapter),
            "--dataset", str(DATA / dataset),
            "--output", str(output),
            "--batch-size", "4", "--seed", "6419",
        ],
        env,
    )
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def main() -> None:
    started = time.time()
    manifest = json.loads((INPUT / "job_manifest.json").read_text(encoding="utf-8"))
    output_repo = str(manifest["output_repo"])
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(output_repo, private=True, exist_ok=True)
    status: dict[str, Any] = {"state": "starting", "started_at_utc": now(), "manifest": manifest, "stages": {}}
    env = os.environ.copy()
    try:
        for filename, expected in manifest["sha256"].items():
            if sha256(INPUT / filename) != expected:
                raise RuntimeError(f"Input hash mismatch: {filename}")
        SOURCE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(INPUT / "source.tar.gz", "r:gz") as archive:
            archive.extractall(SOURCE, filter="data")
        with tarfile.open(INPUT / "data.tar.gz", "r:gz") as archive:
            archive.extractall(SOURCE / "data", filter="data")
        env["PYTHONPATH"] = str(SOURCE)
        init_adapter = MODEL_REPO / "sft" / "final"
        if not (init_adapter / "adapter_model.safetensors").exists():
            raise FileNotFoundError(init_adapter / "adapter_model.safetensors")
        evaluations = WORK / "evaluations"
        evaluations.mkdir(parents=True, exist_ok=True)

        status.update(state="baseline_evaluation")
        upload_json(api, output_repo, status, "status.json", "Start V4.1 baseline")
        baseline_path = evaluations / "baseline.json"
        status["baseline"] = evaluate(init_adapter, "baseline_gate.json", baseline_path, env)
        api.upload_file(path_or_fileobj=str(baseline_path), path_in_repo="evaluations/baseline.json", repo_id=output_repo, commit_message="Upload V4.1 baseline")

        current = init_adapter
        stopped = None
        stages = manifest["stages"]
        for stage in stages:
            name = str(stage["name"])
            status.update(state="training", current_stage=name)
            upload_json(api, output_repo, status, "status.json", f"Start V4.1 {name}")
            output = WORK / name
            metrics = train_stage(stage, current, output, env)
            api.upload_folder(
                folder_path=str(output), path_in_repo=f"{name}/adapter", repo_id=output_repo,
                allow_patterns=["adapter_config.json", "adapter_model.safetensors", "tokenizer*", "trainer_state.json", "training_metrics*.json", "segment_metrics.json"],
                commit_message=f"Upload V4.1 {name} adapter",
            )
            evaluation_path = evaluations / f"{name}.json"
            summary = evaluate(output, str(stage["gate_file"]), evaluation_path, env)
            api.upload_file(path_or_fileobj=str(evaluation_path), path_in_repo=f"evaluations/{name}.json", repo_id=output_repo, commit_message=f"Upload V4.1 {name} gate")
            gate_passed = all(
                float(summary["by_task"][task][metric]) >= float(threshold)
                for task, metric, threshold in stage.get("gate", [])
            )
            status["stages"][name] = {"training": metrics, "evaluation": summary, "gate_passed": gate_passed}
            current = output
            upload_json(api, output_repo, status, "status.json", f"Complete V4.1 {name} gate")
            if not gate_passed:
                stopped = name
                break

        status.update(
            state="complete",
            completed_at_utc=now(),
            stopped_after_stage=stopped,
            next_stage="redesign" if stopped else "analyze_final_free_gate",
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(manifest["cost_per_hour_usd"]),
        )
    except Exception as error:
        status.update(
            state="error", failed_at_utc=now(), error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(), elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(manifest["cost_per_hour_usd"]),
        )
        raise
    finally:
        upload_json(api, output_repo, status, "status.json", "Update V4.1 final status")
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
