from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .lexicon import Lexicon
from .runner import prompt_for_position
from .solver import BoardTile, grid_from_position, infer_orientation, read_word, validate_and_score_move, word_text
from .training import assistant_payload, position_key
from .v4_diagnostic import clean_placements, diagnostic_record, legal_candidates, placement_key, stable_key


def move_plan(position: dict[str, Any], placements: list[dict[str, Any]], lexicon: Lexicon) -> dict[str, Any]:
    grid = grid_from_position(position["board"])
    move = validate_and_score_move(lexicon, grid, str(position["rack"]), placements)
    new_grid = [[cell for cell in row] for row in grid]
    for item in move.placements:
        new_grid[item.row][item.col] = BoardTile(item.letter, item.is_blank)
    orientation = infer_orientation(move.placements)
    if orientation is None:
        placement = move.placements[0]
        options = []
        for direction, dr, dc in (("across", 0, 1), ("down", 1, 0)):
            tiles = read_word(new_grid, placement.row, placement.col, dr, dc)
            text = word_text(tiles)
            if len(tiles) > 1 and text in lexicon.words:
                options.append((len(tiles), direction, tiles, text))
        if not options:
            raise RuntimeError(f"Cannot infer a main line for {position['id']}")
        _, orientation, tiles, text = sorted(options, key=lambda item: (-item[0], item[1]))[0]
    else:
        first = move.placements[0]
        dr, dc = ((0, 1) if orientation == "across" else (1, 0))
        tiles = read_word(new_grid, first.row, first.col, dr, dc)
        text = word_text(tiles)
    anchor = sorted(
        move.placements,
        key=lambda item: (item.col, item.row) if orientation == "across" else (item.row, item.col),
    )[0]
    return {
        "word": text,
        "start_row": int(tiles[0][0]),
        "start_col": int(tiles[0][1]),
        "direction": orientation,
        "anchor_row": anchor.row,
        "anchor_col": anchor.col,
    }


def _base_record(
    position: dict[str, Any],
    record_type: str,
    messages: list[dict[str, str]],
    *,
    assistant_content: str | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{position['id']}--{record_type}",
        "source_id": position["id"],
        "transform": "identity",
        "position_key": position_key(position),
        "optimal_score": int(position["optimal_score"]),
        "record_type": record_type,
        "messages": messages + [{"role": "assistant", "content": assistant_content or assistant_payload(position)}],
    }


def _tagged_prompt(position: dict[str, Any], mode: str, instruction: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    messages = prompt_for_position(position, board_encoding="dense")
    user = json.loads(messages[-1]["content"])
    user["task_mode"] = mode
    user["instruction"] = instruction
    return messages, user


def full_candidate_record(position: dict[str, Any], lexicon: Lexicon, *, task: str, seed: int) -> dict[str, Any]:
    diagnostic = diagnostic_record(position, lexicon, task=task, seed=seed)
    messages = deepcopy(diagnostic["messages"])
    user = json.loads(messages[-1]["content"])
    user["task_mode"] = f"V41_{task.upper()}"
    messages[-1]["content"] = json.dumps(user, separators=(",", ":"))
    from .v4_training import move_payload

    return _base_record(
        position,
        f"v41-{task}",
        messages,
        assistant_content=move_payload(diagnostic["target_placements"]),
    )


def plan_copy_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = move_plan(position, clean_placements(position["canonical_optimal_move"]["placements"]), lexicon)
    messages, user = _tagged_prompt(
        position,
        "V41_PLAN_TO_MOVE",
        "The verifier supplies the optimal word plan. Convert it into only the new tile placements and reply with raw play_move JSON.",
    )
    user["selected_plan"] = plan
    messages[-1]["content"] = json.dumps(user, separators=(",", ":"))
    return _base_record(position, "v41-plan-copy", messages)


def plan_ranking_record(position: dict[str, Any], lexicon: Lexicon, *, seed: int) -> dict[str, Any]:
    candidates = legal_candidates(position, lexicon)
    plans = []
    seen = set()
    for candidate in candidates:
        plan = move_plan(position, candidate["placements"], lexicon)
        key = json.dumps(plan, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        plans.append({"plan": plan, "score": candidate["score"]})
    plans.sort(key=lambda item: stable_key(seed, item["plan"]))
    if len(plans) < 2:
        raise RuntimeError(f"Not enough unique plans for {position['id']}")
    messages, user = _tagged_prompt(
        position,
        "V41_PLAN_RANK",
        "Choose the highest-scoring legal move plan, convert it into only the new tile placements, and reply with raw play_move JSON.",
    )
    user["candidate_plans"] = [dict(candidate=index, **item["plan"]) for index, item in enumerate(plans)]
    messages[-1]["content"] = json.dumps(user, separators=(",", ":"))
    return _base_record(position, "v41-plan-ranking", messages)


def word_direction_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = move_plan(position, clean_placements(position["canonical_optimal_move"]["placements"]), lexicon)
    messages, user = _tagged_prompt(
        position,
        "V41_WORD_DIRECTION",
        "The verifier supplies the optimal main word and direction but not its location. Find its legal placement and reply with raw play_move JSON.",
    )
    user["word_hint"] = {"word": plan["word"], "direction": plan["direction"]}
    messages[-1]["content"] = json.dumps(user, separators=(",", ":"))
    return _base_record(position, "v41-word-direction", messages)


def anchor_direction_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = move_plan(position, clean_placements(position["canonical_optimal_move"]["placements"]), lexicon)
    messages, user = _tagged_prompt(
        position,
        "V41_ANCHOR_DIRECTION",
        "The verifier supplies one new-tile anchor and direction but not the word. Find the optimal legal move and reply with raw play_move JSON.",
    )
    user["anchor_hint"] = {
        "row": plan["anchor_row"],
        "col": plan["anchor_col"],
        "direction": plan["direction"],
    }
    messages[-1]["content"] = json.dumps(user, separators=(",", ":"))
    return _base_record(position, "v41-anchor-direction", messages)


def free_record(position: dict[str, Any]) -> dict[str, Any]:
    messages, user = _tagged_prompt(
        position,
        "V41_FREE_MOVE",
        "Find the highest-scoring legal move without candidate or plan hints. Reply with raw play_move JSON only.",
    )
    messages[-1]["content"] = json.dumps(user, separators=(",", ":"))
    return _base_record(position, "v41-free", messages)


def diagnostic_from_training_record(record: dict[str, Any], position: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "source_id": position["id"],
        "source_game_id": position["source_game_id"],
        "band_ply": int(position["band_ply"]),
        "task": record["record_type"].removeprefix("v41-"),
        "position": position,
        "messages": record["messages"][:-1],
        "target_placements": clean_placements(position["canonical_optimal_move"]["placements"]),
    }
