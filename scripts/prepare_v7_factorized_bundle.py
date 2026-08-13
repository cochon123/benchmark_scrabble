#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfApi


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(output: Path, root: Path, entries: list[Path]) -> None:
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as target:
                for path in sorted(set(entries)):
                    name = str(path.relative_to(root))
                    info = target.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        target.addfile(info, handle)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/v7_factorized_bundle"
    output.mkdir(parents=True, exist_ok=True)
    entries = [root / "pyproject.toml", root / "data/lexicon/ENABLE.txt"]
    entries += list((root / "scrabble_bench").rglob("*.py"))
    entries += [
        root / "scripts/train_qlora.py",
        root / "scripts/evaluate_pass_at_k.py",
        root / "scripts/evaluate_v41.py",
        root / "scripts/evaluate_v42.py",
        root / "scripts/evaluate_v7_factorized.py",
        root / "scripts/build_v7_onpolicy_corpus.py",
        root / "data/general_v42/localization_gate.json",
        root / "data/general_v41/plan_gate.json",
        root / "data/general_v42_passk/positions.json",
    ]
    entries += list((root / "data/general_v7").glob("*"))
    archive_path = output / "source.tar.gz"
    archive(archive_path, root, entries)
    manifest = {
        "version": "v7-factorized-onpolicy-pilot-1",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "initial_adapter": "Cochon123/Qwen3-4B-Scrabble-General-v4.2/stage1/adapter",
        "output_repo": "Cochon123/Qwen3-4B-Scrabble-General-v7-factorized",
        "hardware": "l40sx1",
        "cost_per_hour_usd": 1.8,
        "timeout_hours": 4.0,
        "max_cost_usd": 7.2,
        "stage_a_steps": 200,
        "stage_a_learning_rate": 5e-6,
        "stage_b_steps": 150,
        "stage_b_learning_rate": 3e-6,
        "effective_batch_size": 8,
        "max_length": 1408,
        "training_examples_seen": {"stage_a": 1600, "stage_b": 1200},
        "promotion_thresholds": {
            "factorized_location_optimal_pct": 25.0,
            "factorized_free_plan_legal_pct": 5.0,
            "factorized_free_plan_optimal_pct": 3.0,
            "start_choice_pct": 80.0,
            "start_location_pct": 10.0,
            "plan_copy_legal_pct": 85.0,
            "plan_ranking_legal_pct": 90.0,
            "pass16_legal_pct": 15.0,
            "pass16_oracle_score_pct": 9.0,
        },
        "baseline_reference": {
            "checkpoint": "v4.2-stage1",
            "start_choice_pct": 87.0,
            "start_location_pct": 12.0,
            "plan_copy_legal_pct": 90.0,
            "plan_ranking_legal_pct": 98.0,
            "pass16_legal_pct": 15.0,
            "pass16_oracle_score_pct": 9.847198641765704,
        },
        "source_sha256": sha256(archive_path),
    }
    (output / "job_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "hf_train_v7_factorized_job.py").write_bytes(
        (root / "scripts/hf_train_v7_factorized_job.py").read_bytes()
    )
    repo = "Cochon123/scrabble-v7-factorized-input"
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.create_repo(manifest["output_repo"], private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(output),
        repo_id=repo,
        repo_type="dataset",
        commit_message="Upload audited V7 factorized on-policy pilot",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
