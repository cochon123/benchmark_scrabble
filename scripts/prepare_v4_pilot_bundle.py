#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def deterministic_archive(destination: Path, entries: Iterable[tuple[Path, str]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for source, arcname in sorted(entries, key=lambda item: item[1]):
                    info = archive.gettarinfo(str(source), arcname=arcname)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)


def source_entries(root: Path) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    for relative in (Path("pyproject.toml"), Path("uv.lock"), Path("data/lexicon/ENABLE.txt")):
        path = root / relative
        if path.exists():
            entries.append((path, str(relative)))
    for folder in ("scrabble_bench", "scripts"):
        for path in (root / folder).rglob("*.py"):
            entries.append((path, str(path.relative_to(root))))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Package and optionally upload the v4 SFT pilot.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/general_v4_pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v4_pilot_job_bundle"))
    parser.add_argument("--input-repo", default="Cochon123/scrabble-v4-pilot-training")
    parser.add_argument("--output-repo", default="Cochon123/Qwen3-4B-Scrabble-General-v4-pilot-work")
    parser.add_argument("--colab-audit", type=Path, default=Path("artifacts/v4_pilot/colab/v4_colab_audit.json"))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    required = [
        args.data_dir / "train.jsonl",
        args.data_dir / "validation.jsonl",
        args.data_dir / "gate_positions.json",
        args.data_dir / "preferences.jsonl",
        args.data_dir / "manifest.json",
        args.colab_audit,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    colab_audit = json.loads(args.colab_audit.read_text(encoding="utf-8"))
    if not bool(colab_audit.get("passed")):
        raise RuntimeError("Refusing to package a corpus that failed its Colab audit")
    if int(colab_audit["token_lengths"]["full_max"]) > 1408:
        raise RuntimeError("The audited corpus does not fit the declared 1,408-token cap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_archive = args.output_dir / "scrabble_v4_source.tar.gz"
    data_archive = args.output_dir / "scrabble_v4_data.tar.gz"
    deterministic_archive(source_archive, source_entries(root))
    deterministic_archive(
        data_archive,
        [
            (path, f"general_v4_pilot/{path.relative_to(args.data_dir)}")
            for path in args.data_dir.rglob("*")
            if path.is_file()
        ],
    )
    colab_copy = args.output_dir / "v4_colab_audit.json"
    colab_copy.write_bytes(args.colab_audit.read_bytes())
    manifest = {
        "version": 4,
        "purpose": "cost-capped short-SFT gate before any online RL",
        "source_archive": {"filename": source_archive.name, "sha256": sha256(source_archive)},
        "data_archive": {"filename": data_archive.name, "sha256": sha256(data_archive)},
        "colab_audit": {"filename": colab_copy.name, "sha256": sha256(colab_copy)},
        "dataset": {
            "manifest_sha256": sha256(args.data_dir / "manifest.json"),
            "train_sha256": sha256(args.data_dir / "train.jsonl"),
            "validation_sha256": sha256(args.data_dir / "validation.jsonl"),
            "gate_sha256": sha256(args.data_dir / "gate_positions.json"),
            "preferences_sha256": sha256(args.data_dir / "preferences.jsonl"),
            "train_examples": jsonl_count(args.data_dir / "train.jsonl"),
            "validation_examples": jsonl_count(args.data_dir / "validation.jsonl"),
            "gate_positions": len(json.loads((args.data_dir / "gate_positions.json").read_text(encoding="utf-8"))),
        },
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "input_repo": args.input_repo,
        "output_repo": args.output_repo,
        "cost_per_hour_usd": 1.8,
        "hard_job_timeout_hours": 2.0,
        "pilot_budget_usd": 6.0,
        "training": {
            "max_length": 1408,
            "max_steps": 600,
            "learning_rate": 5e-5,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "gradient_checkpointing": False,
            "lora_r": 16,
            "lora_alpha": 32,
            "json_loss_weight": 1.0,
            "placement_loss_weight": 1.0,
            "checkpoint_every": 100,
            "seed": 4407,
        },
        "gate": {
            "board_encoding": "dense",
            "max_new_tokens": 128,
            "max_attempts": 1,
            "legal_move_pct_min": 10.0,
            "parseable_pct_min": 95.0,
            "median_completion_tokens_max": 96,
        },
    }
    manifest_path = args.output_dir / "v4_pilot_job_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.input_repo, repo_type="dataset", private=True, exist_ok=True)
        api.create_repo(args.output_repo, private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(args.output_dir),
            repo_id=args.input_repo,
            repo_type="dataset",
            commit_message="Upload immutable Scrabble v4 pilot bundle",
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
