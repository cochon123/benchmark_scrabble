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
DATA = SOURCE / "data/general_v7"
V42 = Path("/v42/stage1/adapter")
PRIOR_STAGE_A = Path("/prior/stage_a/adapter")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE)
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=SOURCE, env=environment, check=True)


def upload_json(
    api: HfApi, repo: str, payload: Any, path_in_repo: str, message: str
) -> None:
    path = WORK / path_in_repo.replace("/", "-")
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=repo,
        commit_message=message,
    )


def evaluate_factorized(adapter: Path, dataset: Path, output: Path) -> dict[str, Any]:
    run(
        [
            sys.executable,
            str(SOURCE / "scripts/evaluate_v7_factorized.py"),
            "--adapter",
            str(adapter),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--batch-size",
            "8",
            "--max-new-tokens",
            "40",
            "--disable-thinking",
            "--seed",
            "10703",
        ]
    )
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def train(
    init_adapter: Path,
    train_file: Path,
    validation_file: Path,
    output: Path,
    *,
    steps: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    run(
        [
            sys.executable,
            str(SOURCE / "scripts/train_qlora.py"),
            "--model",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--init-adapter",
            str(init_adapter),
            "--train-file",
            str(train_file),
            "--validation-file",
            str(validation_file),
            "--output-dir",
            str(output),
            "--max-length",
            "1408",
            "--max-steps",
            str(steps),
            "--learning-rate",
            str(learning_rate),
            "--batch-size",
            "1",
            "--gradient-accumulation",
            "8",
            "--checkpoint-every",
            "50",
            "--logging-steps",
            "10",
            "--save-total-limit",
            "2",
            "--skip-final-eval",
            "--no-gradient-checkpointing",
            "--disable-thinking",
            "--seed",
            str(seed),
        ]
    )
    return json.loads((output / "training_metrics.json").read_text(encoding="utf-8"))


def evaluate_legacy(
    script: str, adapter: Path, dataset: Path, output: Path, seed: int
) -> dict[str, Any]:
    run(
        [
            sys.executable,
            str(SOURCE / "scripts" / script),
            "--adapter",
            str(adapter),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--batch-size",
            "8",
            "--disable-thinking",
            "--seed",
            str(seed),
        ]
    )
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def evaluate_pass16(adapter: Path, dataset: Path, output: Path) -> dict[str, Any]:
    run(
        [
            sys.executable,
            str(SOURCE / "scripts/evaluate_pass_at_k.py"),
            "--adapter",
            str(adapter),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--samples",
            "16",
            "--sample-chunk",
            "8",
            "--temperature",
            "0.7",
            "--max-new-tokens",
            "96",
            "--disable-thinking",
            "--seed",
            "8432",
        ]
    )
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def upload_adapter(api: HfApi, repo: str, adapter: Path, name: str) -> None:
    api.upload_folder(
        folder_path=str(adapter),
        path_in_repo=f"{name}/adapter",
        repo_id=repo,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer*",
            "trainer_state.json",
            "training_metrics*.json",
            "segment_metrics.json",
        ],
        commit_message=f"Upload V7 {name} adapter",
    )


def main() -> None:
    started = time.time()
    manifest = json.loads((INPUT / "job_manifest.json").read_text(encoding="utf-8"))
    repo = str(manifest["output_repo"])
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, private=True, exist_ok=True)
    status: dict[str, Any] = {
        "state": "starting",
        "started_at_utc": now(),
        "manifest": manifest,
    }
    evaluations = WORK / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    try:
        if sha256(INPUT / "source.tar.gz") != manifest["source_sha256"]:
            raise RuntimeError("Source archive hash mismatch")
        SOURCE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(INPUT / "source.tar.gz", "r:gz") as archive:
            archive.extractall(SOURCE, filter="data")
        if not (V42 / "adapter_model.safetensors").exists():
            raise FileNotFoundError(V42)

        if manifest.get("resume_from_stage_a"):
            if not (PRIOR_STAGE_A / "adapter_model.safetensors").exists():
                raise FileNotFoundError(PRIOR_STAGE_A)
            stage_a = PRIOR_STAGE_A
            prior_status_path = Path("/prior/status.json")
            if prior_status_path.exists():
                prior = json.loads(prior_status_path.read_text(encoding="utf-8"))
                status["reused_stage_a"] = {
                    "source_job": manifest.get("source_job"),
                    "training": prior.get("stage_a_training"),
                    "original_factorized_evaluation": prior.get("stage_a_factorized"),
                    "policy": "reuse completed Stage A; rerun corrected component attribution and do not retrain",
                }
        else:
            status["state"] = "baseline_factorized_evaluation"
            upload_json(api, repo, status, "status.json", "Start V7 baseline")
            baseline_path = evaluations / "baseline-factorized.json"
            status["baseline_factorized"] = evaluate_factorized(
                V42, DATA / "factorized_gate.json", baseline_path
            )

            status["state"] = "stage_a_training"
            upload_json(api, repo, status, "status.json", "Start V7 Stage A")
            stage_a = WORK / "stage-a"
            status["stage_a_training"] = train(
                V42,
                DATA / "stage_a.jsonl",
                DATA / "validation.jsonl",
                stage_a,
                steps=int(manifest["stage_a_steps"]),
                learning_rate=float(manifest["stage_a_learning_rate"]),
                seed=10703,
            )
            upload_adapter(api, repo, stage_a, "stage_a")

        status["state"] = "stage_a_corrected_evaluation"
        upload_json(api, repo, status, "status.json", "Evaluate V7 Stage A")
        stage_a_gate_path = evaluations / "stage-a-factorized.json"
        status["stage_a_factorized"] = evaluate_factorized(
            stage_a, DATA / "factorized_gate.json", stage_a_gate_path
        )

        status["state"] = "onpolicy_rollout"
        upload_json(api, repo, status, "status.json", "Start V7 on-policy rollout")
        rollout_path = evaluations / "stage-a-rollout.json"
        status["stage_a_rollout"] = evaluate_factorized(
            stage_a, DATA / "rollout_gate.json", rollout_path
        )
        corpus = WORK / "onpolicy-corpus"
        run(
            [
                sys.executable,
                str(SOURCE / "scripts/build_v7_onpolicy_corpus.py"),
                "--rollout-dataset",
                str(DATA / "rollout_gate.json"),
                "--rollout-evaluation",
                str(rollout_path),
                "--stage-a",
                str(DATA / "stage_a.jsonl"),
                "--validation",
                str(DATA / "validation.jsonl"),
                "--output-dir",
                str(corpus),
            ]
        )
        status["onpolicy_corpus"] = json.loads(
            (corpus / "manifest.json").read_text(encoding="utf-8")
        )

        status["state"] = "stage_b_training"
        upload_json(api, repo, status, "status.json", "Start V7 Stage B")
        stage_b = WORK / "stage-b"
        status["stage_b_training"] = train(
            stage_a,
            corpus / "stage_b.jsonl",
            corpus / "validation.jsonl",
            stage_b,
            steps=int(manifest["stage_b_steps"]),
            learning_rate=float(manifest["stage_b_learning_rate"]),
            seed=11703,
        )
        upload_adapter(api, repo, stage_b, "stage_b")

        status["state"] = "final_evaluation"
        upload_json(api, repo, status, "status.json", "Start V7 final gates")
        final_factorized_path = evaluations / "stage-b-factorized.json"
        localization_path = evaluations / "stage-b-localization.json"
        plans_path = evaluations / "stage-b-plans.json"
        pass16_path = evaluations / "stage-b-pass16.json"
        status["final_factorized"] = evaluate_factorized(
            stage_b, DATA / "factorized_gate.json", final_factorized_path
        )
        status["final_localization"] = evaluate_legacy(
            "evaluate_v42.py",
            stage_b,
            SOURCE / "data/general_v42/localization_gate.json",
            localization_path,
            7421,
        )
        status["final_plans"] = evaluate_legacy(
            "evaluate_v41.py",
            stage_b,
            SOURCE / "data/general_v41/plan_gate.json",
            plans_path,
            6419,
        )
        status["final_pass16"] = evaluate_pass16(
            stage_b,
            SOURCE / "data/general_v42_passk/positions.json",
            pass16_path,
        )

        for path in evaluations.glob("*.json"):
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"evaluations/{path.name}",
                repo_id=repo,
                commit_message=f"Upload {path.stem}",
            )
        api.upload_folder(
            folder_path=str(corpus),
            path_in_repo="onpolicy_corpus",
            repo_id=repo,
            allow_patterns=["*.json", "*.jsonl"],
            commit_message="Upload V7 on-policy corpus",
        )

        factorized = status["final_factorized"]["by_task"]
        localization = status["final_localization"]["by_task"]
        plans = status["final_plans"]["by_task"]
        pass16 = status["final_pass16"]
        measured = {
            "factorized_location_optimal_pct": factorized["location"]["optimal_pct"],
            "factorized_free_plan_legal_pct": factorized["free-plan"]["legal_pct"],
            "factorized_free_plan_optimal_pct": factorized["free-plan"]["optimal_pct"],
            "start_choice_pct": localization["start-choice"]["success_pct"],
            "start_location_pct": localization["start-location"]["success_pct"],
            "plan_copy_legal_pct": plans["plan-copy"]["legal_pct"],
            "plan_ranking_legal_pct": plans["plan-ranking"]["legal_pct"],
            "pass16_legal_pct": pass16["board_legal_pass_pct"]["16"],
            "pass16_oracle_score_pct": pass16["oracle_best_score_pct"],
        }
        thresholds = manifest["promotion_thresholds"]
        passed = {
            name: float(measured[name]) >= float(threshold)
            for name, threshold in thresholds.items()
        }
        status["promotion"] = {
            "measured": measured,
            "thresholds": thresholds,
            "passed": passed,
            "promote": all(passed.values()),
            "decision": "promote_stage_b" if all(passed.values()) else "reject_keep_v42",
        }
        status.update(
            state="complete",
            completed_at_utc=now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(manifest["cost_per_hour_usd"]),
            total_pilot_estimated_cost_usd=(time.time() - started)
            / 3600
            * float(manifest["cost_per_hour_usd"])
            + float(manifest.get("prior_estimated_cost_usd", 0.0)),
        )
    except Exception as error:
        status.update(
            state="error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(manifest["cost_per_hour_usd"]),
        )
        raise
    finally:
        upload_json(api, repo, status, "status.json", "Update V7 status")
        print(json.dumps(status, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
