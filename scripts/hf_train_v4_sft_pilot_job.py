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
import statistics
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
DATA = SOURCE / "data" / "general_v4_pilot"
OUTPUT = WORK / "v4-sft-adapter"
EVALUATIONS = WORK / "evaluations"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        archive.extractall(destination, filter="data")


def upload_status(api: HfApi, repo: str, status: dict[str, Any]) -> None:
    path = WORK / "job_status.json"
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo="job_status.json",
        repo_id=repo,
        commit_message=f"Update v4 pilot status: {status['state']}",
    )


def run(command: list[str], environment: dict[str, str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=SOURCE, env=environment, check=True)


def evaluation_health(path: Path, model: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from scrabble_bench.runner import parse_tool_payload

    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload["results"]
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    parseable = 0
    lengths: list[int] = []
    for result in results:
        response = str(result["raw_response"])
        lengths.append(len(tokenizer(response, add_special_tokens=False)["input_ids"]))
        try:
            parse_tool_payload(response)
            parseable += 1
        except Exception:
            pass
    return {
        **payload["summary"],
        "parseable": parseable,
        "parseable_pct": 100 * parseable / len(results),
        "completion_tokens_mean": statistics.mean(lengths),
        "completion_tokens_median": statistics.median(lengths),
        "completion_tokens_max": max(lengths),
    }


def main() -> None:
    started = time.time()
    manifest = json.loads((INPUT / "v4_pilot_job_manifest.json").read_text(encoding="utf-8"))
    repo = str(manifest["output_repo"])
    cost_rate = float(manifest["cost_per_hour_usd"])
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, private=True, exist_ok=True)
    status: dict[str, Any] = {
        "state": "starting",
        "started_at_utc": utc_now(),
        "budget_usd": manifest["pilot_budget_usd"],
        "job_cost_cap_usd": cost_rate * float(manifest["hard_job_timeout_hours"]),
        "model": manifest["model"],
    }
    try:
        for key, destination in (("source_archive", SOURCE), ("data_archive", SOURCE / "data")):
            archive = INPUT / manifest[key]["filename"]
            actual = sha256(archive)
            if actual != manifest[key]["sha256"]:
                raise RuntimeError(f"{key} hash mismatch: {actual}")
            extract(archive, destination)
        for filename, expected in (
            ("manifest.json", manifest["dataset"]["manifest_sha256"]),
            ("train.jsonl", manifest["dataset"]["train_sha256"]),
            ("validation.jsonl", manifest["dataset"]["validation_sha256"]),
            ("gate_positions.json", manifest["dataset"]["gate_sha256"]),
            ("preferences.jsonl", manifest["dataset"]["preferences_sha256"]),
        ):
            actual = sha256(DATA / filename)
            if actual != expected:
                raise RuntimeError(f"Data hash mismatch for {filename}: {actual}")
        sys.path.insert(0, str(SOURCE))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE)
        upload_status(api, repo, status)

        training = manifest["training"]
        status.update(state="training", phase="short_sft", step=0)
        upload_status(api, repo, status)
        command = [
            sys.executable,
            str(SOURCE / "scripts" / "train_qlora.py"),
            "--model", str(manifest["model"]),
            "--train-file", str(DATA / "train.jsonl"),
            "--validation-file", str(DATA / "validation.jsonl"),
            "--output-dir", str(OUTPUT),
            "--max-length", str(training["max_length"]),
            "--max-steps", str(training["max_steps"]),
            "--learning-rate", str(training["learning_rate"]),
            "--batch-size", str(training["batch_size"]),
            "--gradient-accumulation", str(training["gradient_accumulation"]),
            "--lora-r", str(training["lora_r"]),
            "--lora-alpha", str(training["lora_alpha"]),
            "--json-loss-weight", str(training["json_loss_weight"]),
            "--placement-loss-weight", str(training["placement_loss_weight"]),
            "--checkpoint-every", str(training["checkpoint_every"]),
            "--logging-steps", "10",
            "--save-total-limit", "3",
            "--skip-final-eval",
            "--seed", str(training["seed"]),
        ]
        command.append(
            "--gradient-checkpointing"
            if bool(training["gradient_checkpointing"])
            else "--no-gradient-checkpointing"
        )
        run(command, environment)
        metrics = json.loads((OUTPUT / "training_metrics.json").read_text(encoding="utf-8"))
        api.upload_folder(
            folder_path=str(OUTPUT),
            path_in_repo="sft/final",
            repo_id=repo,
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
            commit_message="Upload v4 short-SFT pilot adapter",
        )

        EVALUATIONS.mkdir(parents=True, exist_ok=True)
        evaluations: dict[str, Any] = {}
        for label, adapter in (("sft", str(OUTPUT)), ("base", None)):
            status.update(state="evaluating", phase=label, step=training["max_steps"])
            upload_status(api, repo, status)
            output_path = EVALUATIONS / f"gate-{label}-dense.json"
            eval_command = [
                sys.executable,
                str(SOURCE / "scripts" / "evaluate_hf.py"),
                "--model", str(manifest["model"]),
                "--dataset", str(DATA / "gate_positions.json"),
                "--output", str(output_path),
                "--max-new-tokens", str(manifest["gate"]["max_new_tokens"]),
                "--max-attempts", str(manifest["gate"]["max_attempts"]),
                "--batch-size", "4",
                "--board-encoding", str(manifest["gate"]["board_encoding"]),
                "--seed", str(training["seed"]),
            ]
            if adapter is not None:
                eval_command.extend(["--adapter", adapter])
            run(eval_command, environment)
            evaluations[label] = evaluation_health(output_path, str(manifest["model"]))
            api.upload_file(
                path_or_fileobj=str(output_path),
                path_in_repo=f"evaluations/{output_path.name}",
                repo_id=repo,
                commit_message=f"Upload v4 {label} gate evaluation",
            )

        gate = manifest["gate"]
        sft = evaluations["sft"]
        gate_passed = (
            float(sft["legal_move_pct"]) >= float(gate["legal_move_pct_min"])
            and float(sft["parseable_pct"]) >= float(gate["parseable_pct_min"])
            and float(sft["completion_tokens_median"]) <= float(gate["median_completion_tokens_max"])
        )
        comparison = {
            "protocol": manifest["gate"],
            "base": evaluations["base"],
            "sft": evaluations["sft"],
            "deltas": {
                "legal_move_pct": evaluations["sft"]["legal_move_pct"] - evaluations["base"]["legal_move_pct"],
                "score_pct": evaluations["sft"]["score_pct"] - evaluations["base"]["score_pct"],
                "optimal_move_pct": evaluations["sft"]["optimal_move_pct"] - evaluations["base"]["optimal_move_pct"],
            },
            "gate_passed": gate_passed,
        }
        comparison_path = EVALUATIONS / "gate-comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(comparison_path),
            path_in_repo="evaluations/gate-comparison.json",
            repo_id=repo,
            commit_message="Upload v4 SFT gate decision",
        )
        status.update(
            state="complete",
            completed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * cost_rate,
            training_metrics=metrics,
            gate=comparison,
            next_stage="preference_or_rl_pilot" if gate_passed else "stop_and_redesign_corpus",
        )
    except Exception as error:
        status.update(
            state="error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            failed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * cost_rate,
        )
        if OUTPUT.exists():
            try:
                api.upload_folder(
                    folder_path=str(OUTPUT),
                    path_in_repo="sft/recovery",
                    repo_id=repo,
                    allow_patterns=["checkpoint-*/**", "training_metrics*.json", "trainer_state.json"],
                    commit_message="Upload recoverable v4 SFT state after failure",
                )
            except Exception:
                status["recovery_upload_error"] = traceback.format_exc()
        raise
    finally:
        upload_status(api, repo, status)
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
