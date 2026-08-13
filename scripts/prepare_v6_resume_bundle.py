#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi

from prepare_v6_pilot_bundle import archive, sha256


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/v6_pilot_resume_bundle"
    output.mkdir(parents=True, exist_ok=True)
    entries = [root / "pyproject.toml", root / "data/lexicon/ENABLE.txt"]
    entries += list((root / "scrabble_bench").rglob("*.py"))
    entries += [
        root / "scripts/train_qlora.py",
        root / "scripts/evaluate_pass_at_k.py",
        root / "scripts/evaluate_v42.py",
        root / "scripts/evaluate_v41.py",
        root / "data/general_v42/localization_gate.json",
        root / "data/general_v42/stage1.jsonl",
        root / "data/general_v41/plan_gate.json",
        root / "data/general_v42_passk/positions.json",
    ]
    entries += list((root / "data/general_v6_pilot").glob("*"))
    archive_path = output / "source.tar.gz"
    archive(archive_path, root, entries)
    manifest = {
        "version": "v6-4b-compact-search-resume-1",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "output_repo": "Cochon123/Qwen3-4B-Scrabble-General-v6-pilot",
        "hardware": "l40sx1",
        "cost_per_hour_usd": 1.8,
        "timeout_hours": 3.0,
        "max_cost_usd": 5.4,
        "resume_from_fresh_repo": "Cochon123/Qwen3-4B-Scrabble-General-v6-pilot",
        "source_job": "Cochon123/6a7355e66b79c09949c232f1",
        "winner": "compact_search",
        "fresh_results": {
            "direct": {"selection_pass8_legal_pct": 0.0, "oracle_best_score_pct": 0.0},
            "action_value": {"selection_pass8_legal_pct": 5.0, "oracle_best_score_pct": 0.8495145631067961},
            "compact_search": {"selection_pass8_legal_pct": 5.0, "oracle_best_score_pct": 1.4563106796116505},
        },
        "fresh_steps": 0,
        "fresh_learning_rate": 0.00004,
        "continuation_steps": 250,
        "continuation_learning_rate": 0.00001,
        "selection_samples": 8,
        "final_samples": 16,
        "thinking_disabled_in_template": True,
        "promotion_thresholds": {
            "start_choice_pct": 85.0,
            "start_location_pct": 20.0,
            "pass16_legal_pct": 25.0,
            "pass16_oracle_score_pct": 12.0,
            "plan_copy_legal_pct": 90.0,
            "plan_ranking_legal_pct": 98.0,
        },
        "source_sha256": sha256(archive_path),
    }
    (output / "job_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "hf_train_v6_pilot_job.py").write_bytes(
        (root / "scripts/hf_train_v6_pilot_job.py").read_bytes()
    )
    repo = "Cochon123/scrabble-v6-pilot-resume-input"
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(output),
        repo_id=repo,
        repo_type="dataset",
        commit_message="Upload V6 continuation-only resume bundle",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
