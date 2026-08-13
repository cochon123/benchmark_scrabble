from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Iterable

from .evaluation import error_category
from .lexicon import Lexicon
from .runner import prompt_for_position
from .solver import grid_from_position, validate_and_score_move


def stable_key(seed: int, value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{seed}:{encoded}".encode()).hexdigest()


def placement_key(placements: Iterable[dict[str, Any]]) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        sorted(
            (int(item["row"]), int(item["col"]), str(item["letter"]).upper())
            for item in placements
        )
    )


def clean_placements(placements: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"row": row, "col": col, "letter": letter}
        for row, col, letter in placement_key(placements)
    ]


def legal_candidates(position: dict[str, Any], lexicon: Lexicon) -> list[dict[str, Any]]:
    grid = grid_from_position(position["board"])
    seen: set[tuple[tuple[int, int, str], ...]] = set()
    candidates: list[dict[str, Any]] = []
    raw_moves = [position["canonical_optimal_move"], *position.get("candidate_moves", [])]
    for raw in raw_moves:
        placements = clean_placements(raw["placements"])
        key = placement_key(placements)
        if key in seen:
            continue
        move = validate_and_score_move(lexicon, grid, str(position["rack"]), placements)
        seen.add(key)
        candidates.append(
            {
                "placements": placements,
                "legal": True,
                "score": move.score,
                "kind": "optimal" if move.score == int(position["optimal_score"]) else "legal_suboptimal",
                "error_category": None,
            }
        )
    candidates.sort(key=lambda item: (-int(item["score"]), placement_key(item["placements"])))
    return candidates


def _mutation_stream(position: dict[str, Any]) -> Iterable[list[dict[str, Any]]]:
    optimal = clean_placements(position["canonical_optimal_move"]["placements"])
    board = list(position["board"])
    rack = str(position["rack"]).upper()

    # Hard collision variants are checked before rack or geometry by the verifier.
    for cell in board:
        wrong = next(letter for letter in "ETAOINSHRDLUCMFPGWYBVKXJQZ" if letter != str(cell["letter"]).upper())
        mutant = deepcopy(optimal)
        mutant[0] = {"row": int(cell["row"]), "col": int(cell["col"]), "letter": wrong}
        yield mutant

    # Preserve the rack letters while moving the entire placement pattern. These
    # target anchor, direction, connection, collision, and cross-check behavior.
    for dr in range(-14, 15):
        for dc in range(-14, 15):
            if dr == 0 and dc == 0:
                continue
            shifted = [
                {"row": item["row"] + dr, "col": item["col"] + dc, "letter": item["letter"]}
                for item in optimal
            ]
            if all(0 <= item["row"] < 15 and 0 <= item["col"] < 15 for item in shifted):
                yield shifted

    # Turn an across placement into down, or vice versa, around its first tile.
    first = optimal[0]
    transposed = [
        {
            "row": first["row"] + (item["col"] - first["col"]),
            "col": first["col"] + (item["row"] - first["row"]),
            "letter": item["letter"],
        }
        for item in optimal
    ]
    if all(0 <= item["row"] < 15 and 0 <= item["col"] < 15 for item in transposed):
        yield transposed

    # Letter permutations preserve rack supply but usually break a main/cross word.
    for offset in range(1, len(optimal)):
        letters = [item["letter"] for item in optimal]
        letters = letters[offset:] + letters[:offset]
        yield [{**item, "letter": letter} for item, letter in zip(optimal, letters, strict=True)]

    # Rack and bounds controls guarantee enough exact negatives on every board.
    unavailable = next(letter for letter in "ZQXJKVFWYBPGUMC" if letter not in rack)
    rack_mutant = deepcopy(optimal)
    rack_mutant[0]["letter"] = unavailable
    yield rack_mutant
    for row in range(15, 30):
        bounds = deepcopy(optimal)
        bounds[0]["row"] = row
        yield bounds


def invalid_candidates(position: dict[str, Any], lexicon: Lexicon) -> list[dict[str, Any]]:
    grid = grid_from_position(position["board"])
    seen: set[tuple[tuple[int, int, str], ...]] = set()
    invalid: list[dict[str, Any]] = []
    for placements in _mutation_stream(position):
        key = placement_key(placements)
        if key in seen:
            continue
        seen.add(key)
        try:
            validate_and_score_move(lexicon, grid, str(position["rack"]), placements)
        except Exception as error:
            invalid.append(
                {
                    "placements": clean_placements(placements),
                    "legal": False,
                    "score": 0,
                    "kind": "invalid",
                    "error_category": error_category(str(error)),
                }
            )
    # Prefer category diversity, then stable deterministic fill.
    invalid.sort(key=lambda item: stable_key(5519, item["placements"]))
    diverse: list[dict[str, Any]] = []
    remainder: list[dict[str, Any]] = []
    used_categories: set[str] = set()
    for item in invalid:
        category = str(item["error_category"])
        if category not in used_categories:
            used_categories.add(category)
            diverse.append(item)
        else:
            remainder.append(item)
    return diverse + remainder


def _render_task(
    position: dict[str, Any],
    *,
    task: str,
    candidates: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    candidates = deepcopy(candidates)
    candidates.sort(key=lambda item: stable_key(seed, item["placements"]))
    messages = prompt_for_position(position, board_encoding="dense")
    user = json.loads(messages[-1]["content"])
    instructions = {
        "copy": "The exact verified answer is marked selected=true. Copy that candidate as raw play_move JSON only.",
        "legality": "Exactly one candidate move is legal. Return that candidate as raw play_move JSON only.",
        "ranking": "Choose the highest-scoring legal move from candidate_moves. Return it as raw play_move JSON only.",
        "anchor_direction": "All candidates reuse the intended rack letters, but only one has the correct board anchor/direction. Return the legal candidate as raw play_move JSON only.",
    }
    user["instruction"] = instructions[task]
    user["candidate_moves"] = [
        {
            "candidate": index,
            "placements": item["placements"],
            **({"selected": bool(item.get("selected"))} if task == "copy" else {}),
        }
        for index, item in enumerate(candidates)
    ]
    messages[-1]["content"] = json.dumps(user, separators=(",", ":"))
    selected = next(item for item in candidates if item.get("target"))
    return {
        "id": f"{position['id']}--diagnostic-{task}",
        "source_id": position["id"],
        "source_game_id": position["source_game_id"],
        "band_ply": int(position["band_ply"]),
        "task": task,
        "position": position,
        "messages": messages,
        "target_placements": selected["placements"],
        "target_score": int(selected["score"]),
        "candidates": candidates,
    }


def diagnostic_record(
    position: dict[str, Any],
    lexicon: Lexicon,
    *,
    task: str,
    seed: int,
) -> dict[str, Any]:
    legal = legal_candidates(position, lexicon)
    invalid = invalid_candidates(position, lexicon)
    if len(legal) < 2 or len(invalid) < 7:
        raise RuntimeError(f"Insufficient candidates for {position['id']}: {len(legal)} legal/{len(invalid)} invalid")

    if task == "copy":
        target = deepcopy(legal[seed % len(legal)])
        target.update(target=True, selected=True)
        decoys = [deepcopy(item) for item in invalid[:7]]
        candidates = [target, *decoys]
    elif task == "legality":
        target = deepcopy(legal[seed % len(legal)])
        target["target"] = True
        candidates = [target, *deepcopy(invalid[:7])]
    elif task == "ranking":
        target = deepcopy(legal[0])
        target["target"] = True
        legal_rest = [deepcopy(item) for item in legal[1:5]]
        candidates = [target, *legal_rest, *deepcopy(invalid[: 7 - len(legal_rest)])]
    elif task == "anchor_direction":
        geometry_categories = {"collision", "geometry", "connection", "cross_word", "main_word"}
        geometry = [item for item in invalid if item["error_category"] in geometry_categories]
        canonical_key = placement_key(position["canonical_optimal_move"]["placements"])
        target = deepcopy(next(item for item in legal if placement_key(item["placements"]) == canonical_key))
        optimal_letters = sorted(item["letter"] for item in target["placements"])
        geometry = [
            item for item in geometry
            if sorted(part["letter"] for part in item["placements"]) == optimal_letters
        ]
        if len(geometry) < 7:
            raise RuntimeError(f"Insufficient geometry decoys for {position['id']}: {len(geometry)}")
        target["target"] = True
        candidates = [target, *deepcopy(geometry[:7])]
    else:
        raise ValueError(f"Unknown diagnostic task: {task}")

    # Mark the target before shuffling. `_render_task` locates it afterward.
    return _render_task(position, task=task, candidates=candidates, seed=seed)
