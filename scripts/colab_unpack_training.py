#!/usr/bin/env python3
from __future__ import annotations

import tarfile
from pathlib import Path


archive = Path("/content/scrabble_training.tar.gz")
destination = Path("/content/scrabble_ai/data")
destination.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive, "r:gz") as bundle:
    bundle.extractall(destination, filter="data")
print("Training data extracted to /content/scrabble_ai/data/training")
