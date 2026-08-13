#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_SHA256 = "321d6976d0f66652603d9809d72770a3ac18140684a020457dc8cbb79516b367"
destination = Path("/content/scrabble_checkpoint_100.tar.gz")
parts = sorted(Path("/content").glob("scrabble_checkpoint_100.part.*"))
if not parts:
    raise SystemExit("No checkpoint archive parts found")
with destination.open("wb") as output:
    for part in parts:
        output.write(part.read_bytes())
actual = hashlib.sha256(destination.read_bytes()).hexdigest()
print(f"Reassembled {len(parts)} parts: sha256={actual}")
if actual != EXPECTED_SHA256:
    raise SystemExit(f"Checkpoint archive checksum mismatch: {actual}")
