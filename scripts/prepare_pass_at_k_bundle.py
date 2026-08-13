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
                for path in sorted(entries):
                    name = str(path.relative_to(root))
                    info = target.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        target.addfile(info, handle)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "v42_passk_bundle"
    output.mkdir(parents=True, exist_ok=True)
    entries = [root / "pyproject.toml", root / "data" / "lexicon" / "ENABLE.txt"]
    entries += list((root / "scrabble_bench").rglob("*.py"))
    entries += [root / "scripts" / "evaluate_pass_at_k.py"]
    entries += list((root / "data" / "general_v42_passk").glob("*.json"))
    archive_path = output / "source.tar.gz"
    archive(archive_path, root, entries)
    manifest = {
        "version": "v4.2-passk-job-1",
        "output_repo": "Cochon123/scrabble-v42-pass-at-k",
        "cost_per_hour_usd": 0.4,
        "max_runtime_hours": 3,
        "max_cost_usd": 1.2,
        "models": {
            "v41-stage2": "Cochon123/Qwen3-4B-Scrabble-General-v4.1/stage2/adapter",
            "v42-stage1": "Cochon123/Qwen3-4B-Scrabble-General-v4.2/stage1/adapter",
        },
        "source_sha256": sha256(archive_path),
    }
    (output / "job_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "hf_pass_at_k_job.py").write_bytes((root / "scripts" / "hf_pass_at_k_job.py").read_bytes())
    api = HfApi()
    repo = "Cochon123/scrabble-v42-passk-input"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(output), repo_id=repo, repo_type="dataset", commit_message="Upload frozen pass@32 audit bundle")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
