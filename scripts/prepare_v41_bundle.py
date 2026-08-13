#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive(output: Path, entries: list[tuple[Path, str]]) -> None:
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as target:
                for path, name in sorted(entries, key=lambda item: item[1]):
                    info = target.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        target.addfile(info, handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v41_job_bundle"))
    parser.add_argument("--input-repo", default="Cochon123/scrabble-v41-curriculum")
    parser.add_argument("--output-repo", default="Cochon123/Qwen3-4B-Scrabble-General-v4.1")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / "data" / "general_v41"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_entries = [(root / "pyproject.toml", "pyproject.toml"), (root / "data/lexicon/ENABLE.txt", "data/lexicon/ENABLE.txt")]
    source_entries += [(path, str(path.relative_to(root))) for folder in ("scrabble_bench", "scripts") for path in (root / folder).rglob("*.py")]
    archive(args.output_dir / "source.tar.gz", source_entries)
    archive(
        args.output_dir / "data.tar.gz",
        [(path, f"general_v41/{path.relative_to(data)}") for path in data.rglob("*") if path.is_file()],
    )
    manifest = {
        "version": "v4.1-job-1",
        "output_repo": args.output_repo,
        "init_adapter_repo": "Cochon123/Qwen3-4B-Scrabble-General-v4-pilot-work/sft/final",
        "cost_per_hour_usd": 1.8,
        "timeout_hours": 2.0,
        "max_cost_usd": 3.6,
        "max_length": 1856,
        "stages": [
            {"name": "stage1", "file": "stage1.jsonl", "steps": 200, "learning_rate": 3e-5, "gate_file": "candidate_gate.json", "gate": [["copy", "success_pct", 60], ["ranking", "success_pct", 40]]},
            {"name": "stage2", "file": "stage2.jsonl", "steps": 300, "learning_rate": 2e-5, "gate_file": "plan_gate.json", "gate": [["plan-copy", "success_pct", 50], ["plan-ranking", "success_pct", 30]]},
            {"name": "stage3", "file": "stage3.jsonl", "steps": 300, "learning_rate": 1e-5, "gate_file": "final_gate.json", "gate": []},
        ],
    }
    manifest["sha256"] = {name: sha256(args.output_dir / name) for name in ("source.tar.gz", "data.tar.gz")}
    path = args.output_dir / "job_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.input_repo, repo_type="dataset", private=True, exist_ok=True)
        api.create_repo(args.output_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(args.output_dir), repo_id=args.input_repo, repo_type="dataset", commit_message="Upload immutable V4.1 curriculum bundle")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
