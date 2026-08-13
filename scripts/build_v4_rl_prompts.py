#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scrabble_bench.runner import prompt_for_position


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze train-only prompts for the v4 RL pilot.")
    parser.add_argument("--positions", type=Path, default=Path("data/general_v4_pilot/train_positions.json"))
    parser.add_argument("--output", type=Path, default=Path("data/general_v4_pilot/rl_prompts.jsonl"))
    parser.add_argument("--prompts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=5407)
    args = parser.parse_args()

    positions = json.loads(args.positions.read_text(encoding="utf-8"))
    non_opening = [item for item in positions if int(item["band_ply"]) > 0]
    non_opening.sort(key=lambda item: stable_key(args.seed, str(item["id"])))
    selected = non_opening[: args.prompts]
    if len(selected) != args.prompts:
        raise RuntimeError(f"Requested {args.prompts} prompts, found {len(selected)}")
    rows = [
        {
            "id": item["id"],
            "source_game_id": item["source_game_id"],
            "band_ply": int(item["band_ply"]),
            "prompt": prompt_for_position(item, board_encoding="dense"),
            "position": item,
        }
        for item in selected
    ]
    digest = write_jsonl(args.output, rows)
    manifest = {
        "source": str(args.positions),
        "usage": "train-only online verifier reward; never evaluation",
        "prompts": len(rows),
        "games": len({item["source_game_id"] for item in rows}),
        "openings": sum(item["band_ply"] == 0 for item in rows),
        "sha256": digest,
        "seed": args.seed,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
