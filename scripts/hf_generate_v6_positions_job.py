# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=1.0"]
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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=450)
    parser.add_argument("--seed-start", type=int, default=900_000)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--output-repo", required=True)
    args = parser.parse_args()
    started = time.time()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.output_repo, repo_type="dataset", private=True, exist_ok=True)
    status = {
        "state": "starting",
        "started_at_utc": now(),
        "games": args.games,
        "seed_start": args.seed_start,
        "candidate_limit": 16,
        "candidate_strategy": "stratified",
    }
    try:
        manifest = json.loads((INPUT / "v3_generation_manifest.json").read_text(encoding="utf-8"))
        archive = INPUT / manifest["archive"]["filename"]
        if sha256(archive) != manifest["archive"]["sha256"]:
            raise RuntimeError("Generation source hash mismatch")
        SOURCE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(SOURCE, filter="data")
        if sha256(SOURCE / "data/lexicon/ENABLE.txt") != manifest["lexicon_sha256"]:
            raise RuntimeError("Lexicon hash mismatch")
        output = WORK / "v6-positions"
        command = [
            sys.executable,
            str(SOURCE / "scripts/generate_training_data.py"),
            "--games", str(args.games),
            "--seed-start", str(args.seed_start),
            "--workers", str(args.workers),
            "--output-dir", str(output),
            "--min-ply", "0",
            "--max-ply", "14",
            "--candidate-limit", "16",
            "--candidate-strategy", "stratified",
            "--no-augment",
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE)
        print("[generate] " + " ".join(command), flush=True)
        subprocess.run(command, cwd=SOURCE, env=environment, check=True)
        status.update(
            state="complete",
            completed_at_utc=now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * 1.9,
            manifest=json.loads((output / "manifest.json").read_text(encoding="utf-8")),
        )
        (output / "job_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        api.upload_folder(
            folder_path=str(output),
            repo_id=args.output_repo,
            repo_type="dataset",
            commit_message="Upload fresh V6 score-stratified positions",
        )
        print(json.dumps(status, indent=2), flush=True)
    except Exception as error:
        status.update(
            state="error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            failed_at_utc=now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * 1.9,
        )
        failure = WORK / "v6_generation_failure.json"
        failure.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(failure),
            path_in_repo="job_status.json",
            repo_id=args.output_repo,
            repo_type="dataset",
            commit_message="Record V6 generation failure",
        )
        raise


if __name__ == "__main__":
    main()
