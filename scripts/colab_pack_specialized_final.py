#!/usr/bin/env python3
from __future__ import annotations

import tarfile
from pathlib import Path


root = Path("/content/qwen3-4b-scrabble-specialized-lora")
checkpoint = root / "checkpoint-150"
archive = Path("/content/qwen3-4b-scrabble-specialized-final-lora.tar.gz")
bundle_name = "qwen3-4b-scrabble-specialized-final-lora"
adapter_files = ["README.md", "adapter_config.json", "adapter_model.safetensors"]
root_files = [
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_metrics.json",
]

with tarfile.open(archive, "w:gz") as bundle:
    for name in adapter_files:
        path = checkpoint / name
        if path.exists():
            bundle.add(path, arcname=f"{bundle_name}/{name}")
    for name in root_files:
        path = root / name
        if path.exists():
            bundle.add(path, arcname=f"{bundle_name}/{name}")
    trainer_state = checkpoint / "trainer_state.json"
    if trainer_state.exists():
        bundle.add(trainer_state, arcname=f"{bundle_name}/trainer_state.json")

print(f"Packed final checkpoint {checkpoint} to {archive}")
