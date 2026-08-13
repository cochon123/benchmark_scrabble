from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def materialize_position_subset(
    config: dict[str, Any], root: Path
) -> tuple[list[dict[str, Any]], Path]:
    source = root / str(config["source"])
    positions = json.loads(source.read_text(encoding="utf-8"))
    by_id = {str(position["id"]): position for position in positions}
    requested = [str(position_id) for position_id in config["position_ids"]]
    if len(requested) != len(set(requested)):
        raise ValueError("Evolution position IDs must be unique")
    missing = [position_id for position_id in requested if position_id not in by_id]
    if missing:
        raise ValueError(f"Evolution positions missing from source: {missing}")
    selected = [by_id[position_id] for position_id in requested]
    games = [str(position["source_game_id"]) for position in selected]
    if len(games) != len(set(games)):
        raise ValueError("Evolution positions must come from distinct games")
    return selected, source
