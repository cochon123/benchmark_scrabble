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
MODEL_V2 = Path("/model-v2")
MODEL_V3 = Path("/model-v3")
WORK = Path("/workspace")
SOURCE = WORK / "scrabble_ai"
OUTPUT = WORK / "v3-test20"
COST_PER_HOUR = 0.40
EXPECTED_IDS = (
    "train-500089-0", "train-500090-0", "train-500091-0", "train-500092-0",
    "train-500093-4", "train-500094-4", "train-500095-4", "train-500096-4",
    "train-500097-7", "train-500098-7", "train-500099-7", "train-500189-7",
    "train-500190-10", "train-500191-10", "train-500192-10", "train-500193-10",
    "train-500194-14", "train-500195-14", "train-500196-14", "train-500197-14",
)
EXPECTED_SEMANTIC_SHA256 = "870257faae96ddfcf3d01a84c9643dcfceccd17b5f2260a5876a32055341f650"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def environment() -> dict[str, str]:
    result = os.environ.copy()
    result["PYTHONPATH"] = (
        f"{SOURCE}{os.pathsep}{result['PYTHONPATH']}"
        if result.get("PYTHONPATH")
        else str(SOURCE)
    )
    return result


def main() -> None:
    started = time.time()
    manifest = json.loads((INPUT / "v3_job_manifest.json").read_text(encoding="utf-8"))
    repo = str(manifest["output_repo"])
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    status: dict[str, Any] = {"state": "starting", "started_at_utc": utc_now(), "results": []}
    try:
        for archive_key, destination in (("source_archive", SOURCE), ("data_archive", SOURCE / "data")):
            archive = INPUT / str(manifest[archive_key]["filename"])
            if sha256(archive) != str(manifest[archive_key]["sha256"]):
                raise RuntimeError(f"{archive_key} hash mismatch")
            destination.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:gz") as handle:
                handle.extractall(destination, filter="data")

        all_positions_path = SOURCE / "data/general_v3_sft/test_positions.json"
        all_positions = json.loads(all_positions_path.read_text(encoding="utf-8"))
        by_id = {str(position["id"]): position for position in all_positions}
        missing = [position_id for position_id in EXPECTED_IDS if position_id not in by_id]
        if missing:
            raise RuntimeError(f"Missing test20 positions: {missing}")
        positions = [by_id[position_id] for position_id in EXPECTED_IDS]
        if semantic_sha256(positions) != EXPECTED_SEMANTIC_SHA256:
            raise RuntimeError("Test20 semantic hash mismatch")
        if len({str(position["source_game_id"]) for position in positions}) != 20:
            raise RuntimeError("Test20 positions are not from 20 distinct games")
        if sum(int(position["optimal_score"]) for position in positions) != 630:
            raise RuntimeError("Test20 optimal-point total mismatch")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        dataset = OUTPUT / "positions.json"
        dataset.write_text(json.dumps(positions, indent=2) + "\n", encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(dataset),
            path_in_repo="evaluations/test20/positions.json",
            repo_id=repo,
            commit_message="Add frozen v3 20-board held-out subset",
        )

        evaluations = (
            ("v2-sparse", MODEL_V2 / "final", "sparse"),
            ("v3-sparse", MODEL_V3 / "final", "sparse"),
            ("v3-dense", MODEL_V3 / "final", "dense"),
        )
        summaries: list[dict[str, Any]] = []
        for label, adapter, board_encoding in evaluations:
            if not (adapter / "adapter_model.safetensors").exists():
                raise RuntimeError(f"Missing adapter: {adapter}")
            output = OUTPUT / f"{label}.json"
            command = [
                sys.executable,
                str(SOURCE / "scripts/evaluate_hf.py"),
                "--model", "Qwen/Qwen3-4B-Thinking-2507",
                "--adapter", str(adapter),
                "--dataset", str(dataset),
                "--output", str(output),
                "--board-encoding", board_encoding,
                "--max-new-tokens", "512",
                "--max-attempts", "2",
                "--batch-size", "2",
                "--seed", "3407",
            ]
            print("[evaluate] " + " ".join(command), flush=True)
            subprocess.run(command, cwd=SOURCE, env=environment(), check=True)
            result = json.loads(output.read_text(encoding="utf-8"))
            summary = dict(result["summary"])
            summary.update(label=label, board_encoding=board_encoding, sha256=sha256(output))
            summaries.append(summary)
            status["results"].append(summary)
            api.upload_file(
                path_or_fileobj=str(output),
                path_in_repo=f"evaluations/test20/{output.name}",
                repo_id=repo,
                commit_message=f"Add held-out test20 result {label}",
            )

        comparison = {
            "protocol": {
                "positions": 20,
                "distinct_games": 20,
                "plies": [0, 4, 7, 10, 14],
                "positions_per_ply": 4,
                "optimal_points": 630,
                "semantic_sha256": EXPECTED_SEMANTIC_SHA256,
                "decoding": "greedy",
                "seed": 3407,
                "max_new_tokens": 512,
                "max_attempts": 2,
                "native_encodings": {"v2": "sparse", "v3": "dense"},
                "cross_encoding_control": "v3-sparse",
            },
            "evaluations": summaries,
        }
        comparison_path = OUTPUT / "comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(comparison_path),
            path_in_repo="evaluations/test20/comparison.json",
            repo_id=repo,
            commit_message="Add v2-v3 held-out test20 comparison",
        )
        status.update(
            state="complete",
            completed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * COST_PER_HOUR,
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
        status_path = OUTPUT / "status.json"
        OUTPUT.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(status_path),
            path_in_repo="evaluations/test20/status.json",
            repo_id=repo,
            commit_message=f"Update held-out test20 status: {status['state']}",
        )
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
