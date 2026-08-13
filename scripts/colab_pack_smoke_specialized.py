#!/usr/bin/env python3
from __future__ import annotations

import tarfile
from pathlib import Path


source = Path("/content/qwen3-4b-scrabble-smoke-specialized-lora")
archive = Path("/content/qwen3-4b-scrabble-smoke-specialized-lora.tar.gz")
with tarfile.open(archive, "w:gz") as bundle:
    for path in sorted(source.iterdir()):
        if path.name.startswith("checkpoint-"):
            continue
        bundle.add(path, arcname=f"{source.name}/{path.name}")

    trainer_state = source / "checkpoint-100" / "trainer_state.json"
    if trainer_state.exists():
        bundle.add(trainer_state, arcname=f"{source.name}/trainer_state.json")

print(f"Packed {source} to {archive}")
