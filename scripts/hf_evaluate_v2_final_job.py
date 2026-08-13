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
OUTPUT_REPO = "Cochon123/Qwen3-4B-Scrabble-General-v2-work"
COST_PER_HOUR = 0.40


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload(api: HfApi, path: Path, destination: str, message: str) -> None:
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=destination,
        repo_id=OUTPUT_REPO,
        repo_type="model",
        commit_message=message,
    )


def evaluate(label: str, dataset: Path, api: HfApi) -> dict[str, Any]:
    output = WORK / f"v2-final-{label}.json"
    command = [
        sys.executable,
        str(SOURCE / "scripts/evaluate_hf.py"),
        "--model", "Qwen/Qwen3-4B-Thinking-2507",
        "--adapter", str(MODEL / "final"),
        "--dataset", str(dataset),
        "--output", str(output),
        "--max-new-tokens", "256",
        "--max-attempts", "2",
        "--batch-size", "2",
        "--do-sample",
        "--temperature", "0.6",
        "--top-p", "0.95",
        "--top-k", "20",
        "--seed", "3407",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{SOURCE}{os.pathsep}{environment['PYTHONPATH']}"
        if environment.get("PYTHONPATH")
        else str(SOURCE)
    )
    print("[evaluate] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=SOURCE, env=environment, check=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["final_protocol"] = {
        "split": label,
        "checkpoint": "final step 1265",
        "decoding": "sampled",
        "seed": 3407,
        "max_attempts": 2,
        "max_new_tokens": 256,
        "board_encoding": "sparse benchmark-compatible",
        "created_at_utc": utc_now(),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    upload(
        api,
        output,
        f"evaluations/{output.name}",
        f"Add final v2 {label} evaluation",
    )
    return {"path": str(output), "sha256": sha256(output), "summary": payload["summary"]}


def main() -> None:
    started = time.time()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    manifest = json.loads((INPUT / "v2_final_eval_manifest.json").read_text(encoding="utf-8"))
    archive = INPUT / manifest["archive"]["filename"]
    status: dict[str, Any] = {
        "state": "starting",
        "started_at_utc": utc_now(),
        "results": [],
    }
    try:
        if sha256(archive) != manifest["archive"]["sha256"]:
            raise RuntimeError("Evaluation archive hash mismatch")
        SOURCE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(SOURCE, filter="data")
        official = SOURCE / "data/dataset/benchmark_positions.json"
        generated = SOURCE / "data/evaluation/generated_test_positions.json"
        lexicon = SOURCE / "data/lexicon/ENABLE.txt"
        for path, expected in (
            (official, manifest["official"]["sha256"]),
            (generated, manifest["generated_test"]["sha256"]),
            (lexicon, manifest["lexicon_sha256"]),
        ):
            if sha256(path) != expected:
                raise RuntimeError(f"Extracted input hash mismatch: {path}")
        if not (MODEL / "final/adapter_model.safetensors").exists():
            raise RuntimeError("Final adapter is missing from model mount")

        status["state"] = "official"
        status["results"].append(evaluate("official-benchmark", official, api))
        status["state"] = "generated_test"
        status["results"].append(evaluate("generated-test", generated, api))
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
        status_path = WORK / "v2_final_evaluation_status.json"
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        upload(
            api,
            status_path,
            "evaluations/v2_final_evaluation_status.json",
            f"Update final v2 evaluation status: {status['state']}",
        )
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
