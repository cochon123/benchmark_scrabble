#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


CHECKPOINT_FILES = (
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "rng_state.pth",
    "scaler.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
)
REMOTE_ROOT = "/content/qwen3-4b-scrabble-general-lora"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"[orchestrator] $ {' '.join(command)}", flush=True)
    return subprocess.run(command, check=check, text=True)


def capture(command: list[str]) -> str:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout


def wait_for_file(session: str, filename: str, poll_seconds: int) -> None:
    print(f"[orchestrator] waiting for {filename}", flush=True)
    while True:
        try:
            listing = capture(["colab", "ls", "-s", session, REMOTE_ROOT])
        except subprocess.CalledProcessError as error:
            print(f"[orchestrator] list failed: {error}; retrying", flush=True)
            time.sleep(poll_seconds)
            continue
        if filename in listing.splitlines():
            return
        time.sleep(poll_seconds)


def wait_until_ready(session: str, poll_seconds: int) -> None:
    while True:
        status = capture(["colab", "status", "-s", session])
        if "Status: BUSY" not in status:
            return
        time.sleep(poll_seconds)


def download_checkpoint(session: str, workspace: Path, step: int) -> None:
    destination = workspace / "artifacts" / "generalization" / f"checkpoint-{step}"
    destination.mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE_ROOT}/checkpoint-{step}"
    for filename in CHECKPOINT_FILES:
        run(
            [
                "colab",
                "download",
                "-s",
                session,
                f"{remote}/{filename}",
                str(destination / filename),
            ]
        )
    state = json.loads((destination / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state["global_step"]) != step:
        raise RuntimeError(f"checkpoint global_step is {state['global_step']}, expected {step}")
    files = {}
    for path in sorted(destination.iterdir()):
        if not path.is_file():
            continue
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "checkpoint": step,
        "remote": remote,
        "files": files,
    }
    manifest_path = destination / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[orchestrator] checkpoint {step} archived at {destination}", flush=True)


def download_final_files(session: str, workspace: Path) -> None:
    destination = workspace / "artifacts" / "generalization" / "final-step-1265"
    destination.mkdir(parents=True, exist_ok=True)
    listing = set(capture(["colab", "ls", "-s", session, REMOTE_ROOT]).splitlines())
    wanted = (
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "segment_metrics.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_metrics.json",
        "training_metrics_step_1265.json",
        "vocab.json",
    )
    for filename in wanted:
        if filename not in listing:
            continue
        run(
            [
                "colab",
                "download",
                "-s",
                session,
                f"{REMOTE_ROOT}/{filename}",
                str(destination / filename),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue and archive segmented Colab training.")
    parser.add_argument("--session", default="scrabble-general")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--initial-target", type=int, default=200)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    targets = list(range(args.initial_target, 1201, 100)) + [1265]

    for target in targets:
        metrics_name = f"training_metrics_step_{target}.json"
        wait_for_file(args.session, metrics_name, args.poll_seconds)
        wait_until_ready(args.session, args.poll_seconds)
        if target <= 1200:
            download_checkpoint(args.session, workspace, target)
        else:
            download_final_files(args.session, workspace)
            break
        run(
            [
                "colab",
                "exec",
                "-s",
                args.session,
                "-f",
                str(workspace / "scripts" / "colab_train_general_segment.py"),
                "--timeout",
                "5000",
            ]
        )

    print("[orchestrator] training reached step 1265 and final files were archived", flush=True)


if __name__ == "__main__":
    main()
