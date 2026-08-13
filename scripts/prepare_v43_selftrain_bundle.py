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
    output = root / "artifacts/v43_hf_bundle"
    output.mkdir(parents=True, exist_ok=True)
    entries = [root / "pyproject.toml", root / "data/lexicon/ENABLE.txt"]
    entries += list((root / "scrabble_bench").rglob("*.py"))
    entries += [
        root / "scripts/evaluate_pass_at_k.py", root / "scripts/evaluate_v42.py",
        root / "scripts/build_v43_selftrain_corpus.py", root / "scripts/train_qlora.py",
        root / "data/general_v43/rollout_positions.json",
        root / "data/general_v43/rollout_manifest.json",
        root / "data/general_v42_passk/positions.json",
        root / "data/general_v42/localization_gate.json",
        root / "data/general_v42/stage1.jsonl",
        root / "data/general_v42/validation.jsonl",
        root / "data/general_v41/stage2.jsonl",
    ]
    archive_path = output / "source.tar.gz"
    archive(archive_path, root, entries)
    manifest = {
        "version": "v4.3-selftrain-job-1",
        "output_repo": "Cochon123/Qwen3-4B-Scrabble-General-v4.3-onpolicy",
        "hardware": "l40sx1",
        "cost_per_hour_usd": 1.8,
        "timeout_hours": 2.5,
        "max_cost_usd": 4.5,
        "rollout_positions": 160,
        "rollouts_per_checkpoint": 16,
        "training_steps": 200,
        "source_sha256": sha256(archive_path),
    }
    (output / "job_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "hf_train_v43_selftrain_job.py").write_bytes((root / "scripts/hf_train_v43_selftrain_job.py").read_bytes())
    api = HfApi()
    repo = "Cochon123/scrabble-v43-selftrain-input"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(output), repo_id=repo, repo_type="dataset", commit_message="Upload audited V4.3 self-training bundle")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
