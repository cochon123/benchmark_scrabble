#!/usr/bin/env python3
from __future__ import annotations

import tarfile
from pathlib import Path


root = Path("/content/qwen3-4b-scrabble-general-lora")
checkpoints = sorted(
    root.glob("checkpoint-*"),
    key=lambda path: int(path.name.rsplit("-", 1)[1]),
)
if not checkpoints:
    raise SystemExit("No checkpoint exists")
latest = checkpoints[-1]
archive = Path(f"/content/general-resume-{latest.name}.tar.gz")
with tarfile.open(archive, "w:gz") as bundle:
    bundle.add(latest, arcname=f"qwen3-4b-scrabble-general-lora/{latest.name}")
    for path in sorted(root.iterdir()):
        if path.is_file():
            bundle.add(path, arcname=f"qwen3-4b-scrabble-general-lora/{path.name}")
print(f"Packed {latest} to {archive}")
