#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_v3_job_bundle import deterministic_archive, sha256, source_entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package source, lexicon, and benchmark hashes for remote v3 data generation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3_generation_bundle"))
    parser.add_argument("--input-repo", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "data" / "lexicon" / "ENABLE.txt",
        root / "data" / "dataset" / "benchmark_positions.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    entries = source_entries(root)
    entries.extend((path, str(path.relative_to(root))) for path in required)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "scrabble_v3_generation_source.tar.gz"
    deterministic_archive(archive, entries)
    manifest = {
        "version": 3,
        "archive": {"filename": archive.name, "sha256": sha256(archive)},
        "lexicon_sha256": sha256(required[0]),
        "benchmark_sha256": sha256(required[1]),
        "official_benchmark_policy": "hash exclusion only; never emitted as training data",
    }
    (args.output_dir / "v3_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if args.input_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.input_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(args.output_dir),
            repo_id=args.input_repo,
            repo_type="dataset",
            commit_message="Upload Scrabble v3 generation source bundle",
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
