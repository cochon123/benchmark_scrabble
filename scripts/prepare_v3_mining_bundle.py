#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_v3_job_bundle import deterministic_archive, sha256, source_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Package train-only v3 hard-negative mining inputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3_mining_bundle"))
    parser.add_argument("--input-repo", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    positions = root / "data/general_v3_sft/train_mining_positions.json"
    lexicon = root / "data/lexicon/ENABLE.txt"
    for path in (positions, lexicon):
        if not path.exists():
            raise FileNotFoundError(path)
    entries = source_entries(root)
    entries.extend(
        [
            (positions, "data/general_v3_sft/train_mining_positions.json"),
            (lexicon, "data/lexicon/ENABLE.txt"),
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "scrabble_v3_mining.tar.gz"
    deterministic_archive(archive, entries)
    manifest = {
        "version": 3,
        "archive": {"filename": archive.name, "sha256": sha256(archive)},
        "positions": {
            "count": len(json.loads(positions.read_text(encoding="utf-8"))),
            "sha256": sha256(positions),
            "scope": "predesignated train-only mining subset",
        },
        "lexicon_sha256": sha256(lexicon),
        "base_model": "Qwen/Qwen3-4B-Thinking-2507",
        "adapter_repo": "Cochon123/Qwen3-4B-Scrabble-General-v2-work",
        "adapter_path": "final",
        "output_repo": "Cochon123/scrabble-v3-hard-negatives",
    }
    manifest_path = args.output_dir / "v3_mining_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.input_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.input_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(args.output_dir),
            repo_id=args.input_repo,
            repo_type="dataset",
            commit_message="Upload immutable v3 hard-negative mining bundle",
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
