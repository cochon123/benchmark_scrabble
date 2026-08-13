from __future__ import annotations

import hashlib
from pathlib import Path


output = Path("/content/v41_stage2_adapter.tar.gz")
parts = sorted(Path("/content").glob("v41_stage2_adapter.part-*"))
if not parts:
    raise RuntimeError("No adapter chunks were uploaded")
with output.open("wb") as target:
    for part in parts:
        target.write(part.read_bytes())
digest = hashlib.sha256(output.read_bytes()).hexdigest()
expected = "d996495477c51ca3d60ece9580ad6adb25554d9913fd8ed6e371354454dd8782"
if digest != expected:
    raise RuntimeError(f"Reassembled adapter hash mismatch: {digest} != {expected}")
print({"parts": len(parts), "bytes": output.stat().st_size, "sha256": digest})
