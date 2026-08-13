#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_v3_job_bundle import deterministic_archive, sha256, source_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the immutable v2 final evaluation inputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v2_final_eval_bundle"))
    parser.add_argument("--input-repo", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    official = root / "data/dataset/benchmark_positions.json"
    generated = root / "data/general_v2_sft/test_positions.json"
    lexicon = root / "data/lexicon/ENABLE.txt"
    for path in (official, generated, lexicon):
        if not path.exists():
            raise FileNotFoundError(path)
    entries = source_entries(root)
    entries.extend(
        [
            (official, "data/dataset/benchmark_positions.json"),
            (generated, "data/evaluation/generated_test_positions.json"),
            (lexicon, "data/lexicon/ENABLE.txt"),
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "scrabble_v2_final_eval.tar.gz"
    deterministic_archive(archive, entries)
    manifest = {
        "version": 2,
        "archive": {"filename": archive.name, "sha256": sha256(archive)},
        "official": {
            "positions": 100,
            "sha256": sha256(official),
            "usage": "final benchmark only",
        },
        "generated_test": {
            "positions": 440,
            "sha256": sha256(generated),
            "usage": "one final evaluation after training",
        },
        "lexicon_sha256": sha256(lexicon),
        "adapter_repo": "Cochon123/Qwen3-4B-Scrabble-General-v2-work",
        "adapter_path": "final",
        "base_model": "Qwen/Qwen3-4B-Thinking-2507",
    }
    (args.output_dir / "v2_final_eval_manifest.json").write_text(
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
            commit_message="Upload immutable v2 final benchmark bundle",
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
