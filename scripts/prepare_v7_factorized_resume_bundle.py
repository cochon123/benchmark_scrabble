#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi

from prepare_v7_factorized_bundle import archive, sha256


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/v7_factorized_resume_bundle"
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
    original = json.loads(
        (root / "artifacts/v7_factorized_bundle/job_manifest.json").read_text(encoding="utf-8")
    )
    manifest = {
        **original,
        "version": "v7-factorized-onpolicy-resume-1",
        "hardware": "a100-large",
        "cost_per_hour_usd": 2.5,
        "timeout_hours": 3.0,
        "max_cost_usd": 7.5,
        "resume_from_stage_a": True,
        "source_job": "Cochon123/6a7388af6b79c09949c237b9",
        "recovery_reason": "Preserve completed Stage A; fix first-component attribution before on-policy Stage B",
        "prior_estimated_cost_usd": 0.55,
        "stage_a_steps": 0,
        "source_sha256": sha256(archive_path),
    }
    (output / "job_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "hf_train_v7_factorized_job.py").write_bytes(
        (root / "scripts/hf_train_v7_factorized_job.py").read_bytes()
    )
    repo = "Cochon123/scrabble-v7-factorized-resume-input"
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(output),
        repo_id=repo,
        repo_type="dataset",
        commit_message="Upload corrected V7 Stage-B resume bundle",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
