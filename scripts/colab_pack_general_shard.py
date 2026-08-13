#!/usr/bin/env python3
from __future__ import annotations

import tarfile
from pathlib import Path


source = Path("/content/scrabble_ai/data/general_v2_remote")
archive = Path("/content/general_v2_remote.tar.gz")
with tarfile.open(archive, "w:gz") as bundle:
    bundle.add(source, arcname=source.name)
print(f"Packed {source} to {archive}")
