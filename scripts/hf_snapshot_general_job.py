#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


INPUT = Path("/input")
MODEL_MOUNT = Path("/model")
WORK = Path("/workspace")
SOURCE = WORK / "scrabble_ai"
OUTPUT_REPO = "Cochon123/Qwen3-4B-Scrabble-General-v2-work"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", type=int, default=1)
    parser.add_argument("--checkpoint-step", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    args = parser.parse_args()
    if args.boards < 1:
        raise ValueError("--boards must be positive")
    if args.checkpoint_step < 1:
        raise ValueError("--checkpoint-step must be positive")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    checkpoint = MODEL_MOUNT / "resume" / f"checkpoint-{args.checkpoint_step}"
    decoding = "sampled" if args.do_sample else "greedy"
    output = WORK / (
        f"snapshot-step{args.checkpoint_step}-{args.boards}boards-"
        f"a{args.max_attempts}-{decoding}.json"
    )

    expected = {
        "scrabble_general_source_v2.tar.gz": "0b07f0bea8695b1b62249164775ffb5558a3243cab07ed430cbd3f0fc0d11b98",
        "scrabble_general_data_v2.tar.gz": "a19d0706bc91dce5e98a1d1092dd45a1d2115dc729b886a548849558bd023325",
    }
    for filename, wanted in expected.items():
        actual = sha256(INPUT / filename)
        if actual != wanted:
            raise RuntimeError(f"Input hash mismatch for {filename}: {actual}")
    SOURCE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(INPUT / "scrabble_general_source_v2.tar.gz", "r:gz") as archive:
        archive.extractall(SOURCE, filter="data")
    (SOURCE / "data").mkdir(parents=True, exist_ok=True)
    with tarfile.open(INPUT / "scrabble_general_data_v2.tar.gz", "r:gz") as archive:
        archive.extractall(SOURCE / "data", filter="data")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state["global_step"]) != args.checkpoint_step:
        raise RuntimeError(
            f"Expected checkpoint {args.checkpoint_step}, found {state['global_step']}"
        )
    audit = json.loads(
        (SOURCE / "data" / "general_v2_sft" / "source_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if not audit["passed"]:
        raise RuntimeError("Source leakage audit is not passing")
    command = [
        sys.executable,
        str(SOURCE / "scripts" / "evaluate_hf.py"),
        "--model",
        "Qwen/Qwen3-4B-Thinking-2507",
        "--adapter",
        str(checkpoint),
        "--dataset",
        str(SOURCE / "data" / "general_v2_sft" / "selection_positions.json"),
        "--boards",
        str(args.boards),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-attempts",
        str(args.max_attempts),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        "3407",
        "--output",
        str(output),
    ]
    if args.do_sample:
        command.append("--do-sample")
    print(f"[snapshot] {' '.join(command)}", flush=True)
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{SOURCE}:{existing_pythonpath}" if existing_pythonpath else str(SOURCE)
    )
    subprocess.run(command, cwd=SOURCE, env=environment, check=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["diagnostic_protocol"] = {
        "purpose": "qualitative behavior comparison only",
        "excluded_from_checkpoint_selection": True,
        "excluded_from_hyperparameter_changes": True,
        "position_set": "predesignated validation-selection subset",
        "position_indices": list(range(args.boards)),
        "max_attempts": args.max_attempts,
        "max_new_tokens": args.max_new_tokens,
        "decoding": decoding,
        "batch_size": args.batch_size,
        "comparison_reference": (
            "local step-100 CPU snapshot exists only for position index 0"
        ),
        "hidden_or_official_positions_used": False,
        "checkpoint": args.checkpoint_step,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret is required")
    HfApi(token=token).upload_file(
        path_or_fileobj=str(output),
        path_in_repo=(
            f"diagnostics/snapshot-step{args.checkpoint_step}-{args.boards}boards-"
            f"a{args.max_attempts}-{decoding}.json"
        ),
        repo_id=OUTPUT_REPO,
        repo_type="model",
        commit_message=(
            f"Add {args.boards}-board validation-only diagnostic at step "
            f"{args.checkpoint_step}"
        ),
    )
    print(
        json.dumps(
            {
                "summary": payload["summary"],
                "results": [
                    {
                        "id": result["id"],
                        "score": result["score"],
                        "optimal_score": result["optimal_score"],
                        "is_optimal": result["is_optimal"],
                        "error": result.get("error"),
                    }
                    for result in payload["results"]
                ],
                "diagnostic_protocol": payload["diagnostic_protocol"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
