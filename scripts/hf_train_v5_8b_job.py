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
MODEL = "Qwen/Qwen3-8B"


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


def adapter_args(adapter: Path | None) -> list[str]:
    return ["--adapter", str(adapter)] if adapter is not None else []


def passk(adapter: Path | None, dataset: Path, output: Path, samples: int, *, greedy: bool = False) -> dict[str, Any]:
    command = [
        sys.executable, str(SOURCE / "scripts/evaluate_pass_at_k.py"),
        "--model", MODEL, *adapter_args(adapter), "--dataset", str(dataset),
        "--output", str(output), "--temperature", "0.7", "--samples", str(samples),
        "--sample-chunk", "4", "--max-new-tokens", "96", "--seed", "8501",
        "--disable-thinking",
    ]
    if greedy:
        command.append("--greedy")
    run(command)
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def localization(adapter: Path | None, output: Path) -> dict[str, Any]:
    run([
        sys.executable, str(SOURCE / "scripts/evaluate_v42.py"), "--model", MODEL,
        *adapter_args(adapter), "--dataset", str(SOURCE / "data/general_v42/localization_gate.json"),
        "--output", str(output), "--batch-size", "4", "--seed", "8501", "--disable-thinking",
    ])
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def plans(adapter: Path | None, output: Path) -> dict[str, Any]:
    run([
        sys.executable, str(SOURCE / "scripts/evaluate_v41.py"), "--model", MODEL,
        *adapter_args(adapter), "--dataset", str(SOURCE / "data/general_v41/plan_gate.json"),
        "--output", str(output), "--batch-size", "4", "--seed", "8501", "--disable-thinking",
    ])
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

        evaluations = WORK / "evaluations"
        evaluations.mkdir()
        pass_data = SOURCE / "data/general_v42_passk/positions.json"

        status["state"] = "base_capacity_probe"
        upload_json(api, repo, status, "status.json", "Start 8B base capacity probe")
        base_greedy = evaluations / "base-greedy.json"
        base_pass16 = evaluations / "base-pass16.json"
        base_localization = evaluations / "base-localization.json"
        base_plans = evaluations / "base-plans.json"
        status["base"] = {
            "greedy": passk(None, pass_data, base_greedy, 1, greedy=True),
            "pass16": passk(None, pass_data, base_pass16, 16),
            "localization": localization(None, base_localization),
            "plans": plans(None, base_plans),
        }
        for path in (base_greedy, base_pass16, base_localization, base_plans):
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"evaluations/{path.name}", repo_id=repo, commit_message=f"Upload {path.stem}")

        status["state"] = "training"
        upload_json(api, repo, status, "status.json", "Start balanced 8B QLoRA")
        adapter = WORK / "adapter"
        run([
            sys.executable, str(SOURCE / "scripts/train_qlora.py"), "--model", MODEL,
            "--train-file", str(SOURCE / "data/general_v5_8b/train.jsonl"),
            "--validation-file", str(SOURCE / "data/general_v5_8b/validation.jsonl"),
            "--output-dir", str(adapter), "--max-length", "1856", "--max-steps", "400",
            "--learning-rate", "2e-5", "--batch-size", "1", "--gradient-accumulation", "8",
            "--lora-r", "16", "--lora-alpha", "32", "--json-loss-weight", "1.5",
            "--placement-loss-weight", "2.0", "--checkpoint-every", "100", "--logging-steps", "10",
            "--save-total-limit", "2", "--skip-final-eval", "--gradient-checkpointing",
            "--disable-thinking", "--seed", "8501",
        ])
        status["training"] = json.loads((adapter / "training_metrics.json").read_text(encoding="utf-8"))
        api.upload_folder(
            folder_path=str(adapter), path_in_repo="adapter", repo_id=repo,
            allow_patterns=["adapter_config.json", "adapter_model.safetensors", "tokenizer*", "trainer_state.json", "training_metrics*.json", "segment_metrics.json"],
            commit_message="Upload balanced 8B adapter",
        )

        status["state"] = "final_frozen_gates"
        upload_json(api, repo, status, "status.json", "Start final 8B promotion gates")
        final_greedy = evaluations / "final-greedy.json"
        final_pass16 = evaluations / "final-pass16.json"
        final_localization = evaluations / "final-localization.json"
        final_plans = evaluations / "final-plans.json"
        status["final"] = {
            "greedy": passk(adapter, pass_data, final_greedy, 1, greedy=True),
            "pass16": passk(adapter, pass_data, final_pass16, 16),
            "localization": localization(adapter, final_localization),
            "plans": plans(adapter, final_plans),
        }
        for path in (final_greedy, final_pass16, final_localization, final_plans):
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"evaluations/{path.name}", repo_id=repo, commit_message=f"Upload {path.stem}")

        final = status["final"]
        observed = {
            "start_choice_pct": final["localization"]["by_task"]["start-choice"]["success_pct"],
            "start_location_pct": final["localization"]["by_task"]["start-location"]["success_pct"],
            "pass16_legal_pct": final["pass16"]["board_legal_pass_pct"]["16"],
            "pass16_oracle_score_pct": final["pass16"]["oracle_best_score_pct"],
            "plan_copy_legal_pct": final["plans"]["by_task"]["plan-copy"]["legal_pct"],
            "plan_ranking_legal_pct": final["plans"]["by_task"]["plan-ranking"]["legal_pct"],
        }
        thresholds = manifest["promotion_thresholds"]
        checks = {name: float(observed[name]) >= float(value) for name, value in thresholds.items()}
        status["promotion"] = {
            "thresholds": thresholds, "observed": observed, "checks": checks,
            "passed": all(checks.values()),
            "decision": "promote_to_longer_8b_training" if all(checks.values()) else "stop_8b_pilot",
        }
        status.update(
            state="complete", completed_at_utc=now(), elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(manifest["cost_per_hour_usd"]),
        )
    except Exception as exc:
        status.update(
            state="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(manifest["cost_per_hour_usd"]),
        )
        raise
    finally:
        upload_json(api, repo, status, "status.json", "Update 8B capacity pilot status")
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
