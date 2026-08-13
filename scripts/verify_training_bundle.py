#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


def verify_archive(details: dict[str, Any]) -> dict[str, Any]:
    path = Path(details["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with tarfile.open(path, "r:gz") as bundle:
        members = bundle.getnames()
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "members": len(members),
        "passed": (
            path.stat().st_size == int(details["bytes"])
            and digest == details["sha256"]
            and bool(members)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify portable Scrabble training archives.")
    parser.add_argument(
        "manifest",
        type=Path,
        default=Path("artifacts/generalization/bundle_manifest.json"),
        nargs="?",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = {
        key: verify_archive(manifest[key])
        for key in ("source_archive", "data_archive")
    }
    result["passed"] = all(item["passed"] for item in result.values())
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Training bundle verification failed")


if __name__ == "__main__":
    main()
