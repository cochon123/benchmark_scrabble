from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from typing import Any

from .constants import ALPHABET, BOARD_SIZE, CENTER
from .evaluation import error_category
from .lexicon import Lexicon
from .runner import dense_board_text
from .solver import (
    _cross_check_letters,
    grid_from_position,
    in_bounds,
    neighbors,
    read_word,
    validate_and_score_move,
    word_text,
)
from .training import assistant_payload, position_key
from .v4_diagnostic import clean_placements, stable_key
from .v41_training import move_plan


def localization_messages(
    position: dict[str, Any],
    *,
    mode: str,
    instruction: str,
    extra: dict[str, Any],
    response_format: str,
) -> list[dict[str, str]]:
    system = "\n".join(
        [
            "You are a Scrabble board-localization classifier.",
            "All row and column coordinates are zero-indexed on a 15 by 15 board.",
            "In the dense grid, . is empty and a lowercase letter is an existing zero-point blank.",
            instruction,
            f"Return exactly one compact JSON object in this format: {response_format}",
            "Do not include prose or markdown.",
        ]
    )
    payload = {
        "task_mode": mode,
        "board_size": BOARD_SIZE,
        "rack": list(str(position["rack"])),
        "board_encoding": "dense-grid",
        "board_grid": dense_board_text(position).splitlines(),
        **extra,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def _record(
    position: dict[str, Any],
    record_type: str,
    messages: list[dict[str, str]],
    assistant: str,
) -> dict[str, Any]:
    return {
        "id": f"{position['id']}--{record_type}",
        "source_id": position["id"],
        "transform": "identity",
        "position_key": position_key(position),
        "optimal_score": int(position["optimal_score"]),
        "record_type": record_type,
        "messages": messages + [{"role": "assistant", "content": assistant}],
    }


def placements_for_start(
    position: dict[str, Any],
    lexicon: Lexicon,
    *,
    word: str,
    direction: str,
    start_row: int,
    start_col: int,
) -> tuple[list[dict[str, Any]], int]:
    word = word.upper()
    dr, dc = ((0, 1) if direction == "across" else (1, 0))
    if direction not in {"across", "down"}:
        raise ValueError(f"Unsupported direction: {direction}")
    end_row = start_row + dr * (len(word) - 1)
    end_col = start_col + dc * (len(word) - 1)
    if not in_bounds(start_row, start_col) or not in_bounds(end_row, end_col):
        raise ValueError("Word does not fit from this start square.")
    grid = grid_from_position(position["board"])
    placements = []
    for offset, letter in enumerate(word):
        row, col = start_row + dr * offset, start_col + dc * offset
        tile = grid[row][col]
        if tile is not None:
            if tile.letter != letter:
                raise ValueError(f"Existing tile mismatch at ({row}, {col}).")
            continue
        placements.append({"row": row, "col": col, "letter": letter})
    move = validate_and_score_move(lexicon, grid, str(position["rack"]), placements)
    new_grid = [[cell for cell in row] for row in grid]
    for item in move.placements:
        from .solver import BoardTile

        new_grid[item.row][item.col] = BoardTile(item.letter, item.is_blank)
    tiles = read_word(new_grid, start_row, start_col, dr, dc)
    if not tiles or (tiles[0][0], tiles[0][1]) != (start_row, start_col) or word_text(tiles) != word:
        raise ValueError("The start produces a different extended main word.")
    return clean_placements(placements), int(move.score)


def all_start_outcomes(position: dict[str, Any], lexicon: Lexicon) -> list[dict[str, Any]]:
    plan = move_plan(position, clean_placements(position["canonical_optimal_move"]["placements"]), lexicon)
    word, direction = str(plan["word"]), str(plan["direction"])
    outcomes = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            try:
                placements, score = placements_for_start(
                    position,
                    lexicon,
                    word=word,
                    direction=direction,
                    start_row=row,
                    start_col=col,
                )
                outcomes.append(
                    {
                        "row": row,
                        "col": col,
                        "legal": True,
                        "score": score,
                        "placements": placements,
                        "error_category": None,
                    }
                )
            except Exception as error:
                outcomes.append(
                    {
                        "row": row,
                        "col": col,
                        "legal": False,
                        "score": 0,
                        "placements": [],
                        "error_category": error_category(str(error)),
                    }
                )
    return outcomes


def start_candidates(
    position: dict[str, Any], lexicon: Lexicon, *, seed: int, limit: int = 16
) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    plan = move_plan(position, clean_placements(position["canonical_optimal_move"]["placements"]), lexicon)
    canonical = (int(plan["start_row"]), int(plan["start_col"]))
    outcomes = all_start_outcomes(position, lexicon)
    targets = {
        (int(item["row"]), int(item["col"]))
        for item in outcomes
        if item["legal"] and int(item["score"]) == int(position["optimal_score"])
    }
    targets.add(canonical)
    by_coord = {(int(item["row"]), int(item["col"])): item for item in outcomes}
    selected = [by_coord[canonical]]
    other_legal = [item for item in outcomes if item["legal"] and (item["row"], item["col"]) != canonical]
    other_legal.sort(key=lambda item: (-int(item["score"]), stable_key(seed, [item["row"], item["col"]])))
    selected.extend(other_legal[:3])
    invalid = [item for item in outcomes if not item["legal"]]
    invalid.sort(
        key=lambda item: (
            abs(int(item["row"]) - canonical[0]) + abs(int(item["col"]) - canonical[1]),
            stable_key(seed, [item["row"], item["col"]]),
        )
    )
    seen_categories: set[str] = set()
    for item in invalid:
        category = str(item["error_category"])
        if category in seen_categories:
            continue
        seen_categories.add(category)
        selected.append(item)
        if len(selected) >= limit:
            break
    for item in invalid:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    selected = selected[:limit]
    selected.sort(key=lambda item: stable_key(seed + 1, [item["row"], item["col"]]))
    return selected, targets


def start_choice_record(position: dict[str, Any], lexicon: Lexicon, *, seed: int) -> dict[str, Any]:
    plan = move_plan(position, clean_placements(position["canonical_optimal_move"]["placements"]), lexicon)
    candidates, targets = start_candidates(position, lexicon, seed=seed)
    target_index = next(
        index
        for index, item in enumerate(candidates)
        if (int(item["row"]), int(item["col"])) == (int(plan["start_row"]), int(plan["start_col"]))
    )
    messages = localization_messages(
        position,
        mode="V42_START_CHOICE",
        instruction="The optimal main word and direction are supplied. Choose the candidate start that makes its highest-scoring legal placement.",
        extra={
            "word": plan["word"],
            "direction": plan["direction"],
            "candidate_starts": [
                {"candidate": index, "row": int(item["row"]), "col": int(item["col"])}
                for index, item in enumerate(candidates)
            ],
        },
        response_format='{"candidate":number}',
    )
    return _record(
        position,
        "v42-start-choice",
        messages,
        json.dumps({"candidate": target_index}, separators=(",", ":")),
    )


def start_location_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = move_plan(position, clean_placements(position["canonical_optimal_move"]["placements"]), lexicon)
    messages = localization_messages(
        position,
        mode="V42_LOCATE_WORD",
        instruction="The optimal main word and direction are supplied. Locate the exact start square of its highest-scoring legal placement.",
        extra={"word": plan["word"], "direction": plan["direction"]},
        response_format='{"start_row":number,"start_col":number}',
    )
    target = {"start_row": int(plan["start_row"]), "start_col": int(plan["start_col"])}
    return _record(position, "v42-start-location", messages, json.dumps(target, separators=(",", ":")))


def anchor_mask_record(position: dict[str, Any], *, seed: int, limit: int = 16) -> dict[str, Any]:
    grid = grid_from_position(position["board"])
    anchors = {CENTER} if not position["board"] else {
        (row, col)
        for row in range(BOARD_SIZE)
        for col in range(BOARD_SIZE)
        if grid[row][col] is None and any(grid[nr][nc] is not None for nr, nc in neighbors(row, col))
    }
    anchor_items = sorted(anchors, key=lambda item: stable_key(seed, item))
    nonanchors = [
        (row, col)
        for row in range(BOARD_SIZE)
        for col in range(BOARD_SIZE)
        if grid[row][col] is None and (row, col) not in anchors
    ]
    nonanchors.sort(key=lambda item: stable_key(seed + 1, item))
    cells = (anchor_items[: max(1, limit // 2)] + nonanchors[: max(0, limit - len(anchor_items[: limit // 2]))])[:limit]
    cells.sort(key=lambda item: stable_key(seed + 2, item))
    target_indices = [index for index, item in enumerate(cells) if item in anchors]
    messages = localization_messages(
        position,
        mode="V42_ANCHOR_MASK",
        instruction="Select every candidate empty square adjacent to an existing tile; on an empty board select only the center.",
        extra={"candidate_squares": [{"candidate": i, "row": r, "col": c} for i, (r, c) in enumerate(cells)]},
        response_format='{"anchor_candidates":[number]}',
    )
    return _record(
        position,
        "v42-anchor-mask",
        messages,
        json.dumps({"anchor_candidates": target_indices}, separators=(",", ":")),
    )


def cross_check_record(position: dict[str, Any], lexicon: Lexicon, *, seed: int) -> dict[str, Any]:
    grid = grid_from_position(position["board"])
    options = []
    for direction in ("across", "down"):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if grid[row][col] is not None:
                    continue
                allowed = _cross_check_letters(grid, direction, row, col, lexicon.words)
                if allowed != frozenset(ALPHABET):
                    options.append((direction, row, col, allowed))
    if not options:
        raise ValueError(f"No constrained cross-check square for {position['id']}")
    direction, row, col, allowed = sorted(options, key=lambda item: stable_key(seed, item[:3]))[0]
    letters = "".join(sorted(allowed))
    messages = localization_messages(
        position,
        mode="V42_CROSS_CHECK",
        instruction="Return every letter that can be placed at the square without creating an invalid perpendicular cross-word.",
        extra={"main_direction": direction, "square": {"row": row, "col": col}},
        response_format='{"allowed_letters":"ABCDEFGHIJKLMNOPQRSTUVWXYZ subset in alphabetical order"}',
    )
    return _record(
        position,
        "v42-cross-check",
        messages,
        json.dumps({"allowed_letters": letters}, separators=(",", ":")),
    )


def start_to_move_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = move_plan(position, clean_placements(position["canonical_optimal_move"]["placements"]), lexicon)
    from .v41_training import _tagged_prompt

    messages, user = _tagged_prompt(
        position,
        "V42_START_TO_MOVE",
        "The verifier supplies the optimal word, direction, and exact start square. Convert it into only the new tile placements and reply with raw play_move JSON.",
    )
    user["word_plan"] = {
        "word": plan["word"],
        "direction": plan["direction"],
        "start_row": plan["start_row"],
        "start_col": plan["start_col"],
    }
    messages[-1]["content"] = json.dumps(user, separators=(",", ":"))
    return _record(position, "v42-start-to-move", messages, assistant_payload(position))


def word_direction_move_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    from .v41_training import word_direction_record

    record = deepcopy(word_direction_record(position, lexicon))
    record["id"] = f"{position['id']}--v42-word-direction-move"
    record["record_type"] = "v42-word-direction-move"
    user = json.loads(record["messages"][-2]["content"])
    user["task_mode"] = "V42_LOCATE_AND_MOVE"
    record["messages"][-2]["content"] = json.dumps(user, separators=(",", ":"))
    return record
