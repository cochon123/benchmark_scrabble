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
    output = root / "artifacts/v6_pilot_bundle"
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
        "version": "v6-4b-action-value-pilot-1",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "output_repo": "Cochon123/Qwen3-4B-Scrabble-General-v6-pilot",
        "hardware": "l40sx1",
        "cost_per_hour_usd": 1.8,
        "timeout_hours": 4.0,
        "max_cost_usd": 7.2,
        "fresh_steps": 400,
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
    repo = "Cochon123/scrabble-v6-pilot-input"
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(output),
        repo_id=repo,
        repo_type="dataset",
        commit_message="Upload audited V6 4B ablation pilot",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
