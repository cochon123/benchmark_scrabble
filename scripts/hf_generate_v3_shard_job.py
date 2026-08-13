# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=0.27"]
# ///
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


INPUT = Path("/input")
WORK = Path("/workspace")
SOURCE = WORK / "scrabble_ai"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--output-repo", required=True)
    parser.add_argument("--workers", type=int, default=30)
    args = parser.parse_args()
    started = time.time()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.output_repo, repo_type="dataset", private=True, exist_ok=True)
    status = {
        "state": "starting",
        "shard": args.shard_index,
        "games": args.games,
        "seed_start": args.seed_start,
        "started_at_utc": utc_now(),
    }
    try:
        manifest = json.loads(
            (INPUT / "v3_generation_manifest.json").read_text(encoding="utf-8")
        )
        archive = INPUT / manifest["archive"]["filename"]
        actual = sha256(archive)
        if actual != manifest["archive"]["sha256"]:
            raise RuntimeError(f"Generation source hash mismatch: {actual}")
        SOURCE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(SOURCE, filter="data")
        if sha256(SOURCE / "data/lexicon/ENABLE.txt") != manifest["lexicon_sha256"]:
            raise RuntimeError("Lexicon hash mismatch after extraction")
        if sha256(SOURCE / "data/dataset/benchmark_positions.json") != manifest["benchmark_sha256"]:
            raise RuntimeError("Benchmark exclusion-set hash mismatch after extraction")

        output = WORK / f"shard-{args.shard_index:02d}"
        command = [
            sys.executable,
            str(SOURCE / "scripts/generate_training_data.py"),
            "--games", str(args.games),
            "--seed-start", str(args.seed_start),
            "--workers", str(args.workers),
            "--output-dir", str(output),
            "--min-ply", "0",
            "--max-ply", "14",
            "--candidate-limit", "5",
            "--no-augment",
        ]
        print("[generate] " + " ".join(command), flush=True)
        environment = os.environ.copy()
        prior_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{SOURCE}{os.pathsep}{prior_pythonpath}" if prior_pythonpath else str(SOURCE)
        )
        subprocess.run(command, cwd=SOURCE, env=environment, check=True)
        status.update(
            state="complete",
            completed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            manifest=json.loads((output / "manifest.json").read_text(encoding="utf-8")),
        )
        status_path = output / "job_status.json"
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        api.upload_folder(
            folder_path=str(output),
            path_in_repo=f"shard-{args.shard_index:02d}",
            repo_id=args.output_repo,
            repo_type="dataset",
            commit_message=f"Upload Scrabble v3 position shard {args.shard_index:02d}",
        )
        print(json.dumps(status, indent=2), flush=True)
    except Exception as error:
        status.update(
            state="error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            failed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
        )
        error_path = WORK / f"shard-{args.shard_index:02d}-error.json"
        error_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(error_path),
            path_in_repo=f"shard-{args.shard_index:02d}/job_status.json",
            repo_id=args.output_repo,
            repo_type="dataset",
            commit_message=f"Record failed Scrabble v3 shard {args.shard_index:02d}",
        )
        raise


if __name__ == "__main__":
    main()
