#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
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


def deterministic_archive(
    destination: Path,
    entries: Iterable[tuple[Path, str]],
) -> None:
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
    for relative in (
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("data/lexicon/ENABLE.txt"),
    ):
        path = root / relative
        if path.exists():
            entries.append((path, str(relative)))
    for folder in ("scrabble_bench", "scripts"):
        for path in (root / folder).rglob("*.py"):
            entries.append((path, str(path.relative_to(root))))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build immutable v3 source/data archives and an HF job manifest."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3_job_bundle"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--output-repo", required=True)
    parser.add_argument("--input-repo", default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json-loss-weight", type=float, default=2.0)
    parser.add_argument("--placement-loss-weight", type=float, default=4.0)
    parser.add_argument("--segment-steps", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--cost-per-hour", type=float, default=1.8)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    train = args.data_dir / "train.jsonl"
    validation = args.data_dir / "validation.jsonl"
    manifest_path = args.data_dir / "manifest.json"
    for required in (train, validation, manifest_path):
        if not required.exists():
            raise FileNotFoundError(required)
    if min(
        args.epochs,
        args.batch_size,
        args.gradient_accumulation,
        args.segment_steps,
        args.checkpoint_every,
    ) <= 0:
        raise SystemExit("Epoch, batch, accumulation, and step values must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_archive = args.output_dir / "scrabble_v3_source.tar.gz"
    data_archive = args.output_dir / "scrabble_v3_data.tar.gz"
    deterministic_archive(source_archive, source_entries(root))
    data_files = [path for path in args.data_dir.rglob("*") if path.is_file()]
    deterministic_archive(
        data_archive,
        [(path, f"general_v3_sft/{path.relative_to(args.data_dir)}") for path in data_files],
    )

    train_examples = jsonl_count(train)
    validation_examples = jsonl_count(validation)
    steps_per_epoch = math.ceil(
        train_examples / (args.batch_size * args.gradient_accumulation)
    )
    job_manifest = {
        "version": 3,
        "source_archive": {"filename": source_archive.name, "sha256": sha256(source_archive)},
        "data_archive": {"filename": data_archive.name, "sha256": sha256(data_archive)},
        "dataset": {
            "manifest_sha256": sha256(manifest_path),
            "train_sha256": sha256(train),
            "validation_sha256": sha256(validation),
            "train_examples": train_examples,
            "validation_examples": validation_examples,
        },
        "model": args.model,
        "output_repo": args.output_repo,
        "cost_per_hour_usd": args.cost_per_hour,
        "training": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "gradient_checkpointing": args.gradient_checkpointing,
            "lora_r": 16,
            "lora_alpha": 32,
            "json_loss_weight": args.json_loss_weight,
            "placement_loss_weight": args.placement_loss_weight,
            "segment_steps": args.segment_steps,
            "checkpoint_every": args.checkpoint_every,
            "total_optimizer_steps": steps_per_epoch * args.epochs,
            "seed": 3407,
        },
    }
    evolution_manifest_path = args.data_dir / "evolution_manifest.json"
    evolution_positions_path = args.data_dir / "evolution_positions.json"
    if evolution_manifest_path.exists() and evolution_positions_path.exists():
        evolution = json.loads(evolution_manifest_path.read_text(encoding="utf-8"))
        job_manifest["evolution"] = {
            "positions_path": "data/general_v3_sft/evolution_positions.json",
            "positions_sha256": sha256(evolution_positions_path),
            "positions": int(evolution["positions"]),
            "position_ids": evolution["position_ids"],
            "source_game_ids": evolution["source_game_ids"],
            "plies": evolution["plies"],
            "usage": evolution["usage"],
            "protocol": evolution["protocol"],
            "checkpoint_labels": evolution["checkpoint_labels"],
        }
    output_manifest = args.output_dir / "v3_job_manifest.json"
    output_manifest.write_text(json.dumps(job_manifest, indent=2), encoding="utf-8")

    if args.input_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.input_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(args.output_dir),
            repo_id=args.input_repo,
            repo_type="dataset",
            commit_message="Upload immutable Scrabble v3 training bundle",
        )
    print(json.dumps(job_manifest, indent=2))


if __name__ == "__main__":
    main()
