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
import random
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


INPUT = Path("/input")
WORK = Path("/workspace")
SOURCE = WORK / "scrabble_ai"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
V42_REPO = "Cochon123/Qwen3-4B-Scrabble-General-v4.2"
VARIANTS = ("direct", "action_value", "compact_search")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE)
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=SOURCE, env=environment, check=True)


def upload_json(api: HfApi, repo: str, payload: Any, path_in_repo: str, message: str) -> None:
    path = WORK / path_in_repo.replace("/", "-")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo, repo_id=repo, commit_message=message)


def adapter_args(adapter: Path | None) -> list[str]:
    return ["--adapter", str(adapter)] if adapter else []


def passk(
    adapter: Path,
    dataset: Path,
    output: Path,
    samples: int,
    *,
    greedy: bool = False,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SOURCE / "scripts/evaluate_pass_at_k.py"),
        "--model", MODEL,
        *adapter_args(adapter),
        "--dataset", str(dataset),
        "--output", str(output),
        "--temperature", "0.7",
        "--samples", str(samples),
        "--sample-chunk", "4",
        "--max-new-tokens", "192",
        "--seed", "8601",
        "--disable-thinking",
    ]
    if greedy:
        command.append("--greedy")
    run(command)
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def localization(adapter: Path, output: Path) -> dict[str, Any]:
    run([
        sys.executable,
        str(SOURCE / "scripts/evaluate_v42.py"),
        "--model", MODEL,
        "--adapter", str(adapter),
        "--dataset", str(SOURCE / "data/general_v42/localization_gate.json"),
        "--output", str(output),
        "--batch-size", "8",
        "--seed", "8601",
        "--disable-thinking",
    ])
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def plans(adapter: Path, output: Path) -> dict[str, Any]:
    run([
        sys.executable,
        str(SOURCE / "scripts/evaluate_v41.py"),
        "--model", MODEL,
        "--adapter", str(adapter),
        "--dataset", str(SOURCE / "data/general_v41/plan_gate.json"),
        "--output", str(output),
        "--batch-size", "8",
        "--seed", "8601",
        "--disable-thinking",
    ])
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def train(
    variant: str,
    train_file: Path,
    validation_file: Path,
    output: Path,
    max_length: int,
    steps: int,
    learning_rate: float,
    *,
    init_adapter: Path | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SOURCE / "scripts/train_qlora.py"),
        "--model", MODEL,
        "--train-file", str(train_file),
        "--validation-file", str(validation_file),
        "--output-dir", str(output),
        "--max-length", str(max_length),
        "--max-steps", str(steps),
        "--learning-rate", str(learning_rate),
        "--batch-size", "1",
        "--gradient-accumulation", "8",
        "--lora-r", "16",
        "--lora-alpha", "32",
        "--json-loss-weight", "1.5",
        "--placement-loss-weight", "2.0",
        "--checkpoint-every", str(steps),
        "--logging-steps", "10",
        "--save-total-limit", "1",
        "--skip-final-eval",
        "--no-gradient-checkpointing",
        "--disable-thinking",
        "--seed", "8601",
    ]
    if init_adapter:
        command.extend(["--init-adapter", str(init_adapter)])
    run(command)
    metrics = json.loads((output / "training_metrics.json").read_text(encoding="utf-8"))
    metrics["variant"] = variant
    return metrics


def selection_key(summary: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(summary["board_legal_pass_pct"]["8"]),
        float(summary["oracle_best_score_pct"]),
        float(summary["sample_legal_pct"]),
        float(summary["sample_parseable_pct"]),
    )


def continuation_mix(winner_file: Path, rehearsal_file: Path, output: Path) -> dict[str, int]:
    winner = [json.loads(line) for line in winner_file.read_text(encoding="utf-8").splitlines() if line]
    rehearsal = [json.loads(line) for line in rehearsal_file.read_text(encoding="utf-8").splitlines() if line]
    rng = random.Random(8601)
    rng.shuffle(rehearsal)
    rehearsal = rehearsal[: min(2000, len(rehearsal))]
    # Arrow's JSON loader requires every record to expose the same columns.  V6
    # keeps source_game_id for corpus auditing, while the older V4.2 rehearsal
    # records predate that field.  Training only needs this common schema.
    columns = (
        "id",
        "source_id",
        "source_game_id",
        "transform",
        "position_key",
        "optimal_score",
        "record_type",
        "messages",
    )
    rows = [
        {
            key: (row.get(key, "") if key == "source_game_id" else row[key])
            for key in columns
        }
        for row in winner + rehearsal
    ]
    rng.shuffle(rows)
    output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    return {"winner_records": len(winner), "rehearsal_records": len(rehearsal), "total": len(rows)}


def full_gates(adapter: Path, label: str, evaluations: Path) -> dict[str, Any]:
    greedy_path = evaluations / f"{label}-greedy.json"
    pass_path = evaluations / f"{label}-pass16.json"
    localization_path = evaluations / f"{label}-localization.json"
    plans_path = evaluations / f"{label}-plans.json"
    return {
        "greedy": passk(adapter, SOURCE / "data/general_v42_passk/positions.json", greedy_path, 1, greedy=True),
        "pass16": passk(adapter, SOURCE / "data/general_v42_passk/positions.json", pass_path, 16),
        "localization": localization(adapter, localization_path),
        "plans": plans(adapter, plans_path),
        "files": [str(path) for path in (greedy_path, pass_path, localization_path, plans_path)],
    }


def observed(gates: dict[str, Any]) -> dict[str, float]:
    return {
        "start_choice_pct": gates["localization"]["by_task"]["start-choice"]["success_pct"],
        "start_location_pct": gates["localization"]["by_task"]["start-location"]["success_pct"],
        "pass16_legal_pct": gates["pass16"]["board_legal_pass_pct"]["16"],
        "pass16_oracle_score_pct": gates["pass16"]["oracle_best_score_pct"],
        "plan_copy_legal_pct": gates["plans"]["by_task"]["plan-copy"]["legal_pct"],
        "plan_ranking_legal_pct": gates["plans"]["by_task"]["plan-ranking"]["legal_pct"],
    }


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
        data = SOURCE / "data/general_v6_pilot"
        audit = json.loads((data / "token_audit.json").read_text(encoding="utf-8"))
        max_length = int(audit["recommended_max_length"])
        evaluations = WORK / "evaluations"
        evaluations.mkdir()

        resume_repo = manifest.get("resume_from_fresh_repo")
        if resume_repo:
            winner = str(manifest["winner"])
            fresh_root = Path(snapshot_download(
                str(resume_repo),
                token=os.environ.get("HF_TOKEN"),
                local_dir=WORK / "prior-fresh",
                allow_patterns=[f"fresh/{winner}/adapter/*"],
            ))
            selected_fresh = fresh_root / f"fresh/{winner}/adapter"
            status["fresh"] = manifest.get("fresh_results", {})
            status["selection"] = {
                "winner": winner,
                "policy": "reused completed fresh ablation; no fresh GPU retraining",
                "source_job": manifest.get("source_job"),
            }
        else:
            status["state"] = "fresh_ablation_training"
            upload_json(api, repo, status, "status.json", "Start V6 fresh 4B ablations")
            fresh: dict[str, Any] = {}
            for variant in VARIANTS:
                adapter = WORK / f"fresh-{variant}"
                training = train(
                    variant,
                    data / f"{variant}_train.jsonl",
                    data / f"{variant}_validation.jsonl",
                    adapter,
                    max_length,
                    int(manifest["fresh_steps"]),
                    float(manifest["fresh_learning_rate"]),
                )
                selection_path = evaluations / f"fresh-{variant}-selection-pass8.json"
                selection = passk(adapter, data / "selection_passk.json", selection_path, 8)
                fresh[variant] = {"training": training, "selection": selection}
                api.upload_folder(
                    folder_path=str(adapter),
                    path_in_repo=f"fresh/{variant}/adapter",
                    repo_id=repo,
                    allow_patterns=["adapter_config.json", "adapter_model.safetensors", "tokenizer*", "trainer_state.json", "training_metrics*.json", "segment_metrics.json"],
                    commit_message=f"Upload fresh {variant} adapter",
                )
                api.upload_file(
                    path_or_fileobj=str(selection_path),
                    path_in_repo=f"evaluations/{selection_path.name}",
                    repo_id=repo,
                    commit_message=f"Upload {variant} selection result",
                )
                status["fresh"] = fresh
                upload_json(api, repo, status, "status.json", f"Complete fresh {variant}")

            winner = max(VARIANTS, key=lambda name: selection_key(fresh[name]["selection"]))
            selected_fresh = WORK / f"fresh-{winner}"
            status["selection"] = {
                "winner": winner,
                "keys": {name: selection_key(fresh[name]["selection"]) for name in VARIANTS},
                "policy": "lexicographic pass@8 legality, oracle score, sample legality, parseability",
            }

        status["state"] = "v42_continuation_training"
        upload_json(api, repo, status, "status.json", f"Start V4.2 continuation with {winner}")
        v42_root = Path(snapshot_download(
            V42_REPO,
            token=os.environ.get("HF_TOKEN"),
            local_dir=WORK / "v42",
            allow_patterns=["stage1/adapter/*"],
        ))
        continuation_file = WORK / "continuation_train.jsonl"
        mix = continuation_mix(
            data / f"{winner}_train.jsonl",
            SOURCE / "data/general_v42/stage1.jsonl",
            continuation_file,
        )
        continuation = WORK / "v42-continuation"
        continuation_training = train(
            f"v42-{winner}",
            continuation_file,
            data / f"{winner}_validation.jsonl",
            continuation,
            max_length,
            int(manifest["continuation_steps"]),
            float(manifest["continuation_learning_rate"]),
            init_adapter=v42_root / "stage1/adapter",
        )
        api.upload_folder(
            folder_path=str(continuation),
            path_in_repo="v42_continuation/adapter",
            repo_id=repo,
            allow_patterns=["adapter_config.json", "adapter_model.safetensors", "tokenizer*", "trainer_state.json", "training_metrics*.json", "segment_metrics.json"],
            commit_message="Upload V4.2-initialized winner",
        )
        status["continuation"] = {"mix": mix, "training": continuation_training}

        status["state"] = "untouched_hard_gates"
        upload_json(api, repo, status, "status.json", "Start selected models on hard gates")
        finalists = {
            f"fresh_{winner}": full_gates(selected_fresh, f"fresh-{winner}", evaluations),
            f"v42_{winner}": full_gates(continuation, f"v42-{winner}", evaluations),
        }
        for result in finalists.values():
            for raw in result.pop("files"):
                path = Path(raw)
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=f"evaluations/{path.name}",
                    repo_id=repo,
                    commit_message=f"Upload {path.stem}",
                )
        thresholds = manifest["promotion_thresholds"]
        promotion: dict[str, Any] = {}
        for label, gates in finalists.items():
            values = observed(gates)
            checks = {name: float(values[name]) >= float(target) for name, target in thresholds.items()}
            promotion[label] = {
                "observed": values,
                "checks": checks,
                "passed_count": sum(checks.values()),
                "passed": all(checks.values()),
            }
        status["finalists"] = finalists
        status["promotion"] = promotion
        promoted = [label for label, result in promotion.items() if result["passed"]]
        status.update(
            state="complete",
            decision=("promote:" + ",".join(promoted)) if promoted else "stop_before_rl",
            completed_at_utc=now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(manifest["cost_per_hour_usd"]),
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
        upload_json(api, repo, status, "status.json", "Update V6 pilot status")
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
