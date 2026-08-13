from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Callable

from .lexicon import Lexicon
from .runner import prompt_for_position
from .solver import grid_from_position, validate_and_score_move


Transform = Callable[[int, int], tuple[int, int]]


def _identity(row: int, col: int) -> tuple[int, int]:
    return row, col


def _transpose(row: int, col: int) -> tuple[int, int]:
    return col, row


TRANSFORMS: dict[str, Transform] = {
    "identity": _identity,
    # Scrabble words must read left-to-right or top-to-bottom. Most dihedral
    # transforms reverse tile order and therefore turn valid words into their
    # usually-invalid reversals. Transpose is the only non-identity board
    # symmetry that preserves both the premium layout and word order.
    "transpose": _transpose,
}


def position_key(position: dict[str, Any]) -> str:
    cells = sorted(
        (
            int(cell["row"]),
            int(cell["col"]),
            str(cell["letter"]).upper(),
            bool(cell.get("is_blank", False)),
        )
        for cell in position["board"]
    )
    payload = {"board": cells, "rack": "".join(sorted(position["rack"]))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def transform_position(position: dict[str, Any], transform_name: str) -> dict[str, Any]:
    transform = TRANSFORMS[transform_name]
    transformed = deepcopy(position)
    transformed["id"] = f"{position['id']}--{transform_name}"
    transformed["board"] = [
        {
            **cell,
            "row": transform(int(cell["row"]), int(cell["col"]))[0],
            "col": transform(int(cell["row"]), int(cell["col"]))[1],
        }
        for cell in position["board"]
    ]

    if "optimal_moves" in transformed:
        for move in transformed["optimal_moves"]:
            move["placements"] = _transform_placements(move["placements"], transform)
    if "candidate_moves" in transformed:
        for move in transformed["candidate_moves"]:
            move["placements"] = _transform_placements(move["placements"], transform)
    if "canonical_optimal_move" in transformed:
        transformed["canonical_optimal_move"]["placements"] = _transform_placements(
            transformed["canonical_optimal_move"]["placements"],
            transform,
        )
    return transformed


def _transform_placements(placements: list[dict[str, Any]], transform: Transform) -> list[dict[str, Any]]:
    result = []
    for placement in placements:
        row, col = transform(int(placement["row"]), int(placement["col"]))
        result.append({**placement, "row": row, "col": col})
    return sorted(result, key=lambda item: (item["row"], item["col"], item["letter"]))


def assistant_payload(position: dict[str, Any]) -> str:
    placements = [
        {
            "row": int(item["row"]),
            "col": int(item["col"]),
            "letter": str(item["letter"]).upper(),
        }
        for item in position["canonical_optimal_move"]["placements"]
    ]
    return json.dumps(
        {"tool": "play_move", "arguments": {"placements": placements}},
        separators=(",", ":"),
    )


def assistant_completion(position: dict[str, Any], *, include_reasoning: bool = False) -> str:
    payload = assistant_payload(position)
    if not include_reasoning:
        return payload
    move = position["canonical_optimal_move"]
    placements = move["placements"]
    if len({int(item["row"]) for item in placements}) == 1:
        orientation = "across"
    elif len({int(item["col"]) for item in placements}) == 1:
        orientation = "down"
    else:
        orientation = "single tile"
    placement_text = ", ".join(
        f"{str(item['letter']).upper()} at row {int(item['row'])}, col {int(item['col'])}"
        for item in placements
    )
    words = ", ".join(str(word).upper() for word in move.get("words", []))
    return "\n".join(
        [
            f"The best immediate move scores {int(position['optimal_score'])} points.",
            f"It forms: {words}.",
            f"The move is {orientation}; place only these new tiles: {placement_text}.",
            "</think>",
            payload,
        ]
    )


def training_record(
    position: dict[str, Any],
    transform_name: str = "identity",
    *,
    include_reasoning: bool = False,
) -> dict[str, Any]:
    item = transform_position(position, transform_name) if transform_name != "identity" else deepcopy(position)
    messages = prompt_for_position(item)
    messages.append(
        {
            "role": "assistant",
            "content": assistant_completion(item, include_reasoning=include_reasoning),
        }
    )
    return {
        "id": item["id"],
        "source_id": position["id"],
        "transform": transform_name,
        "position_key": position_key(item),
        "optimal_score": int(item["optimal_score"]),
        "messages": messages,
    }


def recovery_training_record(
    position: dict[str, Any],
    lexicon: Lexicon,
    transform_name: str = "identity",
    *,
    include_reasoning: bool = False,
) -> dict[str, Any]:
    item = transform_position(position, transform_name) if transform_name != "identity" else deepcopy(position)
    raw_response, error = _invalid_attempt(item, lexicon)
    messages = prompt_for_position(
        item,
        {"raw_response": raw_response, "error": error},
    )
    messages.append(
        {
            "role": "assistant",
            "content": assistant_completion(item, include_reasoning=include_reasoning),
        }
    )
    return {
        "id": f"{item['id']}--recovery",
        "source_id": position["id"],
        "transform": transform_name,
        "position_key": position_key(item),
        "optimal_score": int(item["optimal_score"]),
        "record_type": "invalid-move-recovery",
        "rejected_response": raw_response,
        "rejection_error": error,
        "messages": messages,
    }


def _invalid_attempt(position: dict[str, Any], lexicon: Lexicon) -> tuple[str, str]:
    target = [
        {
            "row": int(item["row"]),
            "col": int(item["col"]),
            "letter": str(item["letter"]).upper(),
        }
        for item in position["canonical_optimal_move"]["placements"]
    ]
    rows = {item["row"] for item in target}
    cols = {item["col"] for item in target}
    offsets = [(1, 0), (-1, 0)] if len(rows) == 1 else [(0, 1), (0, -1)]
    candidates: list[list[dict[str, Any]]] = []
    for row_offset, col_offset in offsets:
        shifted = [
            {
                **item,
                "row": item["row"] + row_offset,
                "col": item["col"] + col_offset,
            }
            for item in target
        ]
        if all(0 <= item["row"] < 15 and 0 <= item["col"] < 15 for item in shifted):
            candidates.append(shifted)

    replacement = next(
        letter
        for letter in "ZXQJKVFWYBPGUMC"
        if letter not in position["rack"]
    )
    wrong_rack = deepcopy(target)
    wrong_rack[0]["letter"] = replacement
    candidates.append(wrong_rack)
    candidates.append([])

    rotation = int(position_key(position)[:8], 16) % len(candidates)
    candidates = candidates[rotation:] + candidates[:rotation]
    grid = grid_from_position(position["board"])
    for placements in candidates:
        raw_response = json.dumps(
            {"tool": "play_move", "arguments": {"placements": placements}},
            separators=(",", ":"),
        )
        try:
            validate_and_score_move(lexicon, grid, position["rack"], placements)
        except Exception as exc:
            return raw_response, str(exc)
    raise RuntimeError(f"Could not construct an invalid attempt for {position['id']}")
