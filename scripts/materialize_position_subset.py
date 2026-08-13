#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scrabble_bench.subsets import materialize_position_subset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a frozen position subset by ID.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected, source = materialize_position_subset(config, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    payload = {
        "version": int(config["version"]),
        "name": str(config["name"]),
        "usage": str(config["usage"]),
        "source": str(source.relative_to(root)),
        "source_sha256": sha256(source),
        "positions": len(selected),
        "position_ids": [str(position["id"]) for position in selected],
        "source_game_ids": [str(position["source_game_id"]) for position in selected],
        "plies": [int(position["band_ply"]) for position in selected],
        "protocol": config["protocol"],
        "checkpoint_labels": config["checkpoint_labels"],
        "output": str(output.relative_to(root)),
        "output_sha256": sha256(output),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
