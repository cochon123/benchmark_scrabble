# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.2",
#   "bitsandbytes>=0.45",
#   "huggingface-hub>=1.0",
#   "jinja2>=3.1",
#   "peft>=0.14",
#   "safetensors>=0.4",
#   "torch>=2.6",
#   "transformers>=4.55",
# ]
# ///
from __future__ import annotations

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
from typing import Any

from huggingface_hub import HfApi


INPUT = Path("/input")
V2_MODEL = Path("/v2-model")
WORK = Path("/workspace")
SOURCE = WORK / "scrabble_ai"
OUTPUT = WORK / "v3-hard-negatives"
COST_PER_HOUR = 0.40


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{SOURCE}{os.pathsep}{environment['PYTHONPATH']}"
        if environment.get("PYTHONPATH")
        else str(SOURCE)
    )
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=SOURCE, env=environment, check=True)


def upload_status(api: HfApi, repo: str, status: dict[str, Any]) -> None:
    path = WORK / "mining_status.json"
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo="mining_status.json",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Update v3 mining status: {status['state']}",
    )


def main() -> None:
    started = time.time()
    manifest = json.loads((INPUT / "v3_mining_manifest.json").read_text(encoding="utf-8"))
    output_repo = str(manifest["output_repo"])
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(output_repo, repo_type="dataset", private=True, exist_ok=True)
    status: dict[str, Any] = {
        "state": "starting",
        "started_at_utc": utc_now(),
        "positions": int(manifest["positions"]["count"]),
    }
    try:
        archive = INPUT / str(manifest["archive"]["filename"])
        if sha256(archive) != str(manifest["archive"]["sha256"]):
            raise RuntimeError("Mining archive hash mismatch")
        SOURCE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(SOURCE, filter="data")
        positions = SOURCE / "data/general_v3_sft/train_mining_positions.json"
        lexicon = SOURCE / "data/lexicon/ENABLE.txt"
        if sha256(positions) != str(manifest["positions"]["sha256"]):
            raise RuntimeError("Mining position hash mismatch")
        if sha256(lexicon) != str(manifest["lexicon_sha256"]):
            raise RuntimeError("Lexicon hash mismatch")
        adapter = V2_MODEL / str(manifest["adapter_path"])
        if not (adapter / "adapter_model.safetensors").exists():
            raise RuntimeError("V2 final adapter is missing")
        upload_status(api, output_repo, status)

        evaluation = WORK / "v2-on-v3-train-mining.json"
        status["state"] = "evaluating"
        upload_status(api, output_repo, status)
        run(
            [
                sys.executable,
                str(SOURCE / "scripts/evaluate_hf.py"),
                "--model", str(manifest["base_model"]),
                "--adapter", str(adapter),
                "--dataset", str(positions),
                "--output", str(evaluation),
                "--board-encoding", "dense",
                "--max-new-tokens", "256",
                "--max-attempts", "3",
                "--batch-size", "2",
                "--do-sample",
                "--temperature", "0.6",
                "--top-p", "0.95",
                "--top-k", "20",
                "--seed", "3407",
            ]
        )
        api.upload_file(
            path_or_fileobj=str(evaluation),
            path_in_repo=evaluation.name,
            repo_id=output_repo,
            repo_type="dataset",
            commit_message="Upload v2-on-v3 train-only mining evaluation",
        )

        status["state"] = "building"
        upload_status(api, output_repo, status)
        run(
            [
                sys.executable,
                str(SOURCE / "scripts/build_v3_hard_negatives.py"),
                "--positions", str(positions),
                "--evaluation", str(evaluation),
                "--output-dir", str(OUTPUT),
            ]
        )
        api.upload_folder(
            folder_path=str(OUTPUT),
            repo_id=output_repo,
            repo_type="dataset",
            commit_message="Upload verifier-derived v3 hard negatives",
        )
        hard_negative_manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
        status.update(
            state="complete",
            completed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * COST_PER_HOUR,
            hard_negatives=hard_negative_manifest,
        )
    except Exception as error:
        status.update(
            state="error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            failed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * COST_PER_HOUR,
        )
        raise
    finally:
        upload_status(api, output_repo, status)
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
