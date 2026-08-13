#!/usr/bin/env python3
from __future__ import annotations

import tarfile
from pathlib import Path


archive = Path("/content/general_v2_sft_full.tar.gz")
destination = Path("/content/scrabble_ai/data")
destination.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive, "r:gz") as bundle:
    bundle.extractall(destination, filter="data")
print("Training data extracted to /content/scrabble_ai/data/general_v2_sft")
