#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


root = Path("/content/scrabble_ai/data/general_v2_sft")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
checks: dict[str, object] = {}
for split in ("train", "validation", "test"):
    details = manifest["splits"][split]
    jsonl = root / f"{split}.jsonl"
    positions = root / f"{split}_positions.json"
    jsonl_hash = hashlib.sha256(jsonl.read_bytes()).hexdigest()
    positions_hash = hashlib.sha256(positions.read_bytes()).hexdigest()
    records = sum(1 for _ in jsonl.open(encoding="utf-8"))
    checks[split] = {
        "records": records,
        "jsonl_sha256": jsonl_hash,
        "positions_sha256": positions_hash,
        "passed": (
            records == int(details["records"])
            and jsonl_hash == details["jsonl_sha256"]
            and positions_hash == details["positions_sha256"]
        ),
    }

audit = json.loads((root / "source_audit.json").read_text(encoding="utf-8"))
checks["source_audit_passed"] = audit["passed"]
passed = bool(audit["passed"]) and all(
    bool(checks[split]["passed"]) for split in ("train", "validation", "test")
)
checks["passed"] = passed
print(json.dumps(checks, indent=2))
if not passed:
    raise SystemExit("Remote training-data verification failed")
