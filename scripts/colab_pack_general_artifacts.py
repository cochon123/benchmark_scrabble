#!/usr/bin/env python3
from __future__ import annotations

import tarfile
from pathlib import Path


root = Path("/content/qwen3-4b-scrabble-general-lora")
selected = Path("/content/qwen3-4b-scrabble-general-selected")
archive = Path("/content/qwen3-4b-scrabble-general-artifacts.tar.gz")
checkpoint_files = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "README.md",
    "trainer_state.json",
}

with tarfile.open(archive, "w:gz") as bundle:
    for path in sorted(root.iterdir()):
        if path.is_file():
            bundle.add(path, arcname=f"general-lora/{path.name}")
    for checkpoint_name in ("checkpoint-250", "checkpoint-1000", "checkpoint-1250"):
        checkpoint = root / checkpoint_name
        if not checkpoint.exists():
            continue
        for path in sorted(checkpoint.iterdir()):
            if path.is_file() and path.name in checkpoint_files:
                bundle.add(
                    path,
                    arcname=f"general-lora/{checkpoint_name}/{path.name}",
                )
    if selected.exists():
        for path in sorted(selected.iterdir()):
            if path.is_file():
                bundle.add(path, arcname=f"selected-adapter/{path.name}")
    selection = Path("/content/general-checkpoint-selection.json")
    if selection.exists():
        bundle.add(selection, arcname="general-checkpoint-selection.json")
    for name in (
        "general-base-selection-evaluation.json",
        "general-selection-checkpoint250-evaluation.json",
        "general-selection-checkpoint1000-evaluation.json",
        "general-selection-evaluation.json",
        "general-hidden-test-evaluation.json",
        "general-official-smoke-evaluation.json",
    ):
        path = Path("/content") / name
        if path.exists():
            bundle.add(path, arcname=f"evaluations/{name}")

print(f"Packed selected training and evaluation artifacts to {archive}")
