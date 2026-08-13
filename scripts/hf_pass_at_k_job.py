# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.2", "bitsandbytes>=0.45", "huggingface-hub>=1.0",
#   "jinja2>=3.1", "peft>=0.14", "safetensors>=0.4",
#   "torch>=2.6", "transformers>=4.55",
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
WORK = Path("/workspace")
SOURCE = WORK / "scrabble_ai"
V41 = Path("/v41") / "stage2" / "adapter"
V42 = Path("/v42") / "stage1" / "adapter"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upload_json(api: HfApi, repo: str, payload: Any, name: str, message: str) -> None:
    path = WORK / Path(name).name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo, repo_type="dataset", commit_message=message)


def evaluate(adapter: Path, output: Path, temperature: float, *, limit: int | None = None) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SOURCE / "scripts" / "evaluate_pass_at_k.py"),
        "--adapter", str(adapter),
        "--dataset", str(SOURCE / "data" / "general_v42_passk" / "positions.json"),
        "--output", str(output),
        "--temperature", str(temperature),
        "--samples", "32",
        "--sample-chunk", "8",
        "--max-new-tokens", "96",
        "--seed", "8432",
    ]
    if limit:
        command += ["--limit", str(limit)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE)
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=SOURCE, env=environment, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    started = time.time()
    manifest = json.loads((INPUT / "job_manifest.json").read_text(encoding="utf-8"))
    repo = str(manifest["output_repo"])
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    status: dict[str, Any] = {"state": "starting", "started_at_utc": now(), "manifest": manifest}
    try:
        if sha256(INPUT / "source.tar.gz") != manifest["source_sha256"]:
            raise RuntimeError("Source archive hash mismatch")
        SOURCE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(INPUT / "source.tar.gz", "r:gz") as archive:
            archive.extractall(SOURCE, filter="data")
        for adapter in (V41, V42):
            if not (adapter / "adapter_model.safetensors").exists():
                raise FileNotFoundError(adapter)

        status["state"] = "calibrating"
        upload_json(api, repo, status, "status.json", "Start pass@32 calibration")
        calibrations = {}
        for temperature in (0.7, 0.9):
            path = WORK / f"v42-calibration-t{temperature}.json"
            payload = evaluate(V42, path, temperature, limit=5)
            calibrations[str(temperature)] = payload["summary"]
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"calibration/{path.name}", repo_id=repo, repo_type="dataset", commit_message=f"Upload calibration t={temperature}")
        chosen = max(
            (0.7, 0.9),
            key=lambda value: (
                calibrations[str(value)]["board_legal_pass_pct"]["32"],
                calibrations[str(value)]["oracle_best_score_pct"],
                calibrations[str(value)]["unique_parsed_moves_mean"],
            ),
        )
        status.update(state="evaluating", calibrations=calibrations, chosen_temperature=chosen)
        upload_json(api, repo, status, "status.json", "Complete calibration")

        evaluations = {}
        for label, adapter in (("v42-stage1", V42), ("v41-stage2", V41)):
            path = WORK / f"{label}-pass32.json"
            payload = evaluate(adapter, path, chosen)
            evaluations[label] = payload["summary"]
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"evaluations/{path.name}", repo_id=repo, repo_type="dataset", commit_message=f"Upload {label} pass@32")
            status["evaluations"] = evaluations
            upload_json(api, repo, status, "status.json", f"Complete {label}")

        status.update(
            state="complete",
            completed_at_utc=now(),
            evaluations=evaluations,
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(manifest["cost_per_hour_usd"]),
        )
    except Exception as exc:
        status.update(
            state="error",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(manifest["cost_per_hour_usd"]),
        )
        raise
    finally:
        upload_json(api, repo, status, "status.json", "Update pass@32 status")
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
