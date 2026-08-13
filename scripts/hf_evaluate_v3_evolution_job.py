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
MODEL = Path("/model")
WORK = Path("/workspace")
SOURCE = WORK / "scrabble_ai"
OUTPUT = WORK / "v3-evolution"
COST_PER_HOUR = 0.40


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        positions = SOURCE / str(manifest["evolution"]["positions_path"])
        if sha256(positions) != str(manifest["evolution"]["positions_sha256"]):
            raise RuntimeError("Evolution position hash mismatch")
        protocol = manifest["evolution"]["protocol"]
        labels = [str(label) for label in manifest["evolution"]["checkpoint_labels"]]
        evaluations: list[tuple[str, Path]] = []
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for label in labels:
            adapter: Path | None
            if label == "base":
                adapter = None
            elif label == "final":
                adapter = MODEL / "final"
            else:
                adapter = MODEL / "resume" / f"checkpoint-{label.removeprefix('step-')}"
            if adapter is not None and not (adapter / "adapter_model.safetensors").exists():
                raise RuntimeError(f"Missing evolution adapter: {adapter}")
            output = OUTPUT / f"{label}.json"
            command = [
                sys.executable,
                str(SOURCE / "scripts/evaluate_hf.py"),
                "--model", str(manifest["model"]),
                "--dataset", str(positions),
                "--output", str(output),
                "--board-encoding", str(protocol["board_encoding"]),
                "--max-new-tokens", str(protocol["max_new_tokens"]),
                "--max-attempts", str(protocol["max_attempts"]),
                "--batch-size", "2",
                "--seed", str(protocol["seed"]),
            ]
            if adapter is not None:
                command.extend(["--adapter", str(adapter)])
            print("[evaluate] " + " ".join(command), flush=True)
            subprocess.run(command, cwd=SOURCE, env=environment(), check=True)
            evaluations.append((label, output))
            api.upload_file(
                path_or_fileobj=str(output),
                path_in_repo=f"evaluations/evolution/{output.name}",
                repo_id=repo,
                commit_message=f"Add v3 evolution result {label}",
            )
            status["results"].append({"label": label, "sha256": sha256(output)})
        chart_command = [
            sys.executable,
            str(SOURCE / "scripts/build_v3_evolution_chart.py"),
            "--output-dir", str(OUTPUT),
            "--trainer-state", str(MODEL / "final" / "trainer_state.json"),
        ]
        for label, path in evaluations:
            chart_command.extend(["--evaluation", f"{label}={path}"])
        subprocess.run(chart_command, cwd=SOURCE, env=environment(), check=True)
        api.upload_folder(
            folder_path=str(OUTPUT),
            path_in_repo="evaluations/evolution",
            repo_id=repo,
            commit_message="Add v3 five-board evolution chart",
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
        status_path = WORK / "v3_evolution_status.json"
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(status_path),
            path_in_repo="evaluations/evolution/status.json",
            repo_id=repo,
            commit_message=f"Update v3 evolution status: {status['state']}",
        )
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
