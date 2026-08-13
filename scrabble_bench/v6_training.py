from __future__ import annotations

import hashlib
import json
from typing import Any

from .lexicon import Lexicon
from .runner import prompt_for_position
from .solver import grid_from_position, validate_and_score_move
from .training import assistant_payload, position_key, transform_position
from .v41_training import move_plan
from .v4_diagnostic import clean_placements, invalid_candidates, placement_key


def stable_int(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def board_encoding(identifier: str, seed: int, *, dense_rate: float = 0.3) -> str:
    threshold = int(dense_rate * 10_000)
    return "dense" if stable_int(identifier, seed, "encoding") % 10_000 < threshold else "sparse"


def compact_action(
    position: dict[str, Any], placements: list[dict[str, Any]], lexicon: Lexicon
) -> tuple[str, int]:
    clean = clean_placements(placements)
    plan = move_plan(position, clean, lexicon)
    move = validate_and_score_move(
        lexicon, grid_from_position(position["board"]), str(position["rack"]), clean
    )
    direction = "A" if plan["direction"] == "across" else "D"
    code = (
        f"R{int(plan['start_row']):02d}C{int(plan['start_col']):02d}{direction}:"
        f"{plan['word']}"
    )
    return code, int(move.score)


def _base_row(
    position: dict[str, Any],
    source: dict[str, Any],
    transform: str,
    record_type: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "id": f"{source['id']}--{transform}--{record_type}",
        "source_id": source["id"],
        "source_game_id": source["source_game_id"],
        "transform": transform,
        "position_key": position_key(position),
        "optimal_score": int(position["optimal_score"]),
        "record_type": record_type,
        "messages": messages,
    }


def direct_record(
    source: dict[str, Any], lexicon: Lexicon, *, transform: str, seed: int
) -> dict[str, Any]:
    del lexicon
    position = transform_position(source, transform)
    messages = prompt_for_position(
        position, board_encoding=board_encoding(str(source["id"]), seed)
    )
    messages.append({"role": "assistant", "content": assistant_payload(position)})
    return _base_row(position, source, transform, "v6-direct", messages)


def _search_candidates(position: dict[str, Any], lexicon: Lexicon) -> list[tuple[str, int, list[dict[str, Any]]]]:
    raw = position.get("candidate_moves") or position.get("optimal_moves") or []
    candidates: list[tuple[str, int, list[dict[str, Any]]]] = []
    seen: set[str] = set()
    for item in raw:
        placements = clean_placements(item["placements"])
        key = placement_key(placements)
        if key in seen:
            continue
        seen.add(key)
        code, score = compact_action(position, placements, lexicon)
        candidates.append((code, score, placements))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    if not candidates:
        raise RuntimeError(f"No candidates for {position['id']}")
    # Three hard alternatives plus one representative from the lower score range.
    chosen = candidates[:3]
    if len(candidates) > 3:
        lower = candidates[max(3, len(candidates) // 2)]
        if lower[0] not in {item[0] for item in chosen}:
            chosen.append(lower)
    return chosen


def search_record(
    source: dict[str, Any], lexicon: Lexicon, *, transform: str, seed: int
) -> dict[str, Any]:
    position = transform_position(source, transform)
    candidates = _search_candidates(position, lexicon)
    best_code, best_score, _ = candidates[0]
    trace = "|".join(f"{code}={score}" for code, score, _ in candidates)
    completion = (
        f"<think>MOVES {trace};BEST {best_code}={best_score}</think>\n"
        f"{assistant_payload(position)}"
    )
    messages = prompt_for_position(
        position, board_encoding=board_encoding(str(source["id"]), seed)
    )
    messages.append({"role": "assistant", "content": completion})
    return _base_row(position, source, transform, "v6-compact-search", messages)


def value_record(
    source: dict[str, Any], lexicon: Lexicon, *, transform: str, seed: int
) -> dict[str, Any]:
    position = transform_position(source, transform)
    legal = _search_candidates(position, lexicon)
    use_invalid = stable_int(source["id"], transform, seed, "invalid") % 2 == 0
    candidate_placements: list[dict[str, Any]]
    if use_invalid:
        invalid = invalid_candidates(position, lexicon)
        if invalid:
            chosen = invalid[stable_int(source["id"], seed, "bad") % len(invalid)]
            candidate_placements = clean_placements(chosen["placements"])
            target = {"legal": False, "score_bin": 0, "score": 0}
        else:
            use_invalid = False
    if not use_invalid:
        index = stable_int(source["id"], transform, seed, "legal") % len(legal)
        _, score, candidate_placements = legal[index]
        optimum = max(1, int(position["optimal_score"]))
        score_bin = min(10, max(1, round(10 * score / optimum)))
        target = {"legal": True, "score_bin": score_bin, "score": score}

    messages = prompt_for_position(
        position, board_encoding=board_encoding(str(source["id"]), seed)
    )
    system = (
        "You are a strict Scrabble move evaluator. Check the supplied placements against "
        "the board, rack, connection, dictionary, and scoring rules. Return only compact JSON "
        'as {"legal":boolean,"score_bin":number,"score":number}. score_bin is the move score '
        "divided by the board optimum in ten bins from 1 to 10; illegal moves use zeros."
    )
    user = json.loads(messages[1]["content"])
    user["instruction"] = "Evaluate this candidate move rather than proposing a new move."
    user["candidate_placements"] = candidate_placements
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, separators=(",", ":"))},
        {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))},
    ]
    return _base_row(position, source, transform, "v6-action-value", messages)
