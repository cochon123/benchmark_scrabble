#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.evaluation.read_text())
    games = {row["id"]: row["source_game_id"] for row in json.loads(args.positions.read_text())}
    for row in payload["details"]:
        row["source_game_id"] = games[row["id"]]
    payload["summary"]["unique_games"] = len({row["source_game_id"] for row in payload["details"]})
    args.evaluation.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__": main()
