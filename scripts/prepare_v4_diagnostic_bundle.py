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


def archive_source(root: Path, output: Path) -> None:
    entries: list[tuple[Path, str]] = []
    for relative in (Path("pyproject.toml"), Path("data/lexicon/ENABLE.txt")):
        entries.append((root / relative, str(relative)))
    for path in (root / "scrabble_bench").rglob("*.py"):
        entries.append((path, str(path.relative_to(root))))
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path, name in sorted(entries, key=lambda item: item[1]):
                    info = archive.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package and upload the V4 diagnostic inputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/general_v4_diagnostic"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v4_diagnostic_job_bundle"))
    parser.add_argument("--input-repo", default="Cochon123/scrabble-v4-diagnostic")
    parser.add_argument("--output-repo", default="Cochon123/Qwen3-4B-Scrabble-General-v4-pilot-work")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source, name in ((args.data_dir / "diagnostic.json", "diagnostic.json"), (args.data_dir / "manifest.json", "manifest.json")):
        (args.output_dir / name).write_bytes(source.read_bytes())
    archive_source(root, args.output_dir / "source.tar.gz")
    manifest = {
        "version": "v4-diagnostic-job-1",
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "adapter_repo": args.output_repo,
        "output_repo": args.output_repo,
        "seed": 5519,
        "batch_size": 4,
        "cost_per_hour_usd": 0.40,
        "timeout_hours": 1.5,
        "sha256": {
            name: sha256(args.output_dir / name)
            for name in ("diagnostic.json", "manifest.json", "source.tar.gz")
        },
    }
    manifest_path = args.output_dir / "job_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.input_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(args.output_dir),
            repo_id=args.input_repo,
            repo_type="dataset",
            commit_message="Upload immutable V4 ability diagnostic bundle",
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
