# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.2", "bitsandbytes>=0.45", "datasets>=3.2",
#   "huggingface-hub>=1.0", "jinja2>=3.1", "peft>=0.14",
#   "safetensors>=0.4", "torch>=2.6", "transformers>=4.55",
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
V41 = Path("/v41") / "stage2" / "adapter"
V42 = Path("/v42") / "stage1" / "adapter"
COST_PER_HOUR = 1.8


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE)
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=SOURCE, env=environment, check=True)


def upload_json(api: HfApi, repo: str, payload: Any, name: str, message: str) -> None:
    path = WORK / Path(name).name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo, commit_message=message)


def sample(adapter: Path, dataset: Path, output: Path, count: int, *, greedy: bool = False) -> dict[str, Any]:
    command = [
        sys.executable, str(SOURCE / "scripts" / "evaluate_pass_at_k.py"),
        "--adapter", str(adapter), "--dataset", str(dataset), "--output", str(output),
        "--temperature", "0.7", "--samples", str(count), "--sample-chunk", "8",
        "--max-new-tokens", "96", "--seed", "8432",
    ]
    if greedy:
        command.append("--greedy")
    run(command)
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def main() -> None:
    started = time.time()
    manifest = json.loads((INPUT / "job_manifest.json").read_text(encoding="utf-8"))
    repo = str(manifest["output_repo"])
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, private=True, exist_ok=True)
    status: dict[str, Any] = {"state": "starting", "started_at_utc": now(), "manifest": manifest}
    try:
        if sha256(INPUT / "source.tar.gz") != manifest["source_sha256"]:
            raise RuntimeError("Source archive hash mismatch")
        SOURCE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(INPUT / "source.tar.gz", "r:gz") as archive:
            archive.extractall(SOURCE, filter="data")
        for adapter in (V41, V42):
            if not (adapter / "adapter_model.safetensors").exists():
                raise FileNotFoundError(adapter)

        rollout_data = SOURCE / "data/general_v43/rollout_positions.json"
        gate_data = SOURCE / "data/general_v42_passk/positions.json"
        evaluations = WORK / "evaluations"
        evaluations.mkdir()

        status["state"] = "baseline_greedy"
        upload_json(api, repo, status, "status.json", "Start V4.3 baseline")
        baseline_path = evaluations / "baseline-greedy.json"
        status["baseline_greedy"] = sample(V42, gate_data, baseline_path, 1, greedy=True)
        api.upload_file(path_or_fileobj=str(baseline_path), path_in_repo="evaluations/baseline-greedy.json", repo_id=repo, commit_message="Upload V4.3 greedy baseline")

        status["state"] = "rollout_v42"
        upload_json(api, repo, status, "status.json", "Start V4.2 train-only rollouts")
        v42_path = evaluations / "train-rollouts-v42.json"
        status["rollout_v42"] = sample(V42, rollout_data, v42_path, 16)
        api.upload_file(path_or_fileobj=str(v42_path), path_in_repo="rollouts/train-v42.json", repo_id=repo, commit_message="Upload V4.2 train-only rollouts")

        status["state"] = "rollout_v41"
        upload_json(api, repo, status, "status.json", "Start V4.1 train-only rollouts")
        v41_path = evaluations / "train-rollouts-v41.json"
        status["rollout_v41"] = sample(V41, rollout_data, v41_path, 16)
        api.upload_file(path_or_fileobj=str(v41_path), path_in_repo="rollouts/train-v41.json", repo_id=repo, commit_message="Upload V4.1 train-only rollouts")

        corpus = WORK / "corpus"
        run(
            [
                sys.executable, str(SOURCE / "scripts/build_v43_selftrain_corpus.py"),
                "--positions", str(rollout_data), "--v42-rollouts", str(v42_path),
                "--v41-rollouts", str(v41_path),
                "--v42-rehearsal", str(SOURCE / "data/general_v42/stage1.jsonl"),
                "--v41-rehearsal", str(SOURCE / "data/general_v41/stage2.jsonl"),
                "--validation", str(SOURCE / "data/general_v42/validation.jsonl"),
                "--output-dir", str(corpus),
            ]
        )
        status["corpus"] = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
        api.upload_folder(folder_path=str(corpus), path_in_repo="corpus", repo_id=repo, allow_patterns=["*.json", "*.jsonl"], commit_message="Upload V4.3 verifier-filtered corpus")

        status["state"] = "training"
        upload_json(api, repo, status, "status.json", "Start V4.3 offline self-training")
        adapter = WORK / "adapter"
        run(
            [
                sys.executable, str(SOURCE / "scripts/train_qlora.py"),
                "--model", "Qwen/Qwen3-4B-Instruct-2507", "--init-adapter", str(V42),
                "--train-file", str(corpus / "train.jsonl"),
                "--validation-file", str(corpus / "validation.jsonl"),
                "--output-dir", str(adapter), "--max-length", "1216", "--max-steps", "200",
                "--learning-rate", "1e-5", "--batch-size", "1", "--gradient-accumulation", "8",
                "--json-loss-weight", "1.5", "--placement-loss-weight", "2.0",
                "--checkpoint-every", "50", "--logging-steps", "10", "--save-total-limit", "2",
                "--skip-final-eval", "--no-gradient-checkpointing", "--seed", "9431",
            ]
        )
        status["training"] = json.loads((adapter / "training_metrics.json").read_text(encoding="utf-8"))
        api.upload_folder(
            folder_path=str(adapter), path_in_repo="adapter", repo_id=repo,
            allow_patterns=["adapter_config.json", "adapter_model.safetensors", "tokenizer*", "trainer_state.json", "training_metrics*.json", "segment_metrics.json"],
            commit_message="Upload V4.3 offline self-training adapter",
        )

        status["state"] = "final_evaluation"
        upload_json(api, repo, status, "status.json", "Start V4.3 final gates")
        greedy_path = evaluations / "v43-greedy.json"
        pass16_path = evaluations / "v43-pass16.json"
        status["final_greedy"] = sample(adapter, gate_data, greedy_path, 1, greedy=True)
        status["final_pass16"] = sample(adapter, gate_data, pass16_path, 16)
        localization_path = evaluations / "v43-localization.json"
        run(
            [
                sys.executable, str(SOURCE / "scripts/evaluate_v42.py"), "--adapter", str(adapter),
                "--dataset", str(SOURCE / "data/general_v42/localization_gate.json"),
                "--output", str(localization_path), "--batch-size", "8", "--seed", "7421",
            ]
        )
        status["final_localization"] = json.loads(localization_path.read_text(encoding="utf-8"))["summary"]
        for path in (greedy_path, pass16_path, localization_path):
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"evaluations/{path.name}", repo_id=repo, commit_message=f"Upload {path.stem}")

        status.update(
            state="complete", completed_at_utc=now(), elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * COST_PER_HOUR,
        )
    except Exception as exc:
        status.update(
            state="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * COST_PER_HOUR,
        )
        raise
    finally:
        upload_json(api, repo, status, "status.json", "Update V4.3 status")
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
