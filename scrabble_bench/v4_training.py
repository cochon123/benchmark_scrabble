from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from .evaluation import error_category
from .lexicon import Lexicon
from .runner import parse_tool_payload, prompt_for_position
from .solver import grid_from_position, validate_and_score_move
from .training import _invalid_attempt, assistant_payload, position_key


def move_payload(placements: list[dict[str, Any]]) -> str:
    """Render placements in the benchmark's deployment format."""
    normalized = [
        {
            "row": int(item["row"]),
            "col": int(item["col"]),
            "letter": str(item["letter"]).upper(),
        }
        for item in placements
    ]
    return json.dumps(
        {"tool": "play_move", "arguments": {"placements": normalized}},
        separators=(",", ":"),
    )


def _stable_order(value: Any, seed: int) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{seed}:{payload}".encode()).hexdigest()


def _ranking_candidates(position: dict[str, Any], *, seed: int, limit: int = 4) -> list[dict[str, Any]]:
    optimal = position["canonical_optimal_move"]
    optimal_key = json.dumps(optimal["placements"], sort_keys=True)
    lower_legal = [
        move
        for move in position.get("candidate_moves", [])
        if int(move["score"]) < int(position["optimal_score"])
        and json.dumps(move["placements"], sort_keys=True) != optimal_key
    ]
    lower_legal.sort(key=lambda item: _stable_order(item["placements"], seed))

    placements = deepcopy(optimal["placements"])
    invalid: list[dict[str, Any]] = []
    if placements:
        wrong_letter = deepcopy(placements)
        rack = str(position["rack"])
        replacement = next(letter for letter in "ZXQJKVFWYBPGUMC" if letter not in rack)
        wrong_letter[0]["letter"] = replacement
        invalid.append({"placements": wrong_letter, "score": None, "kind": "invalid"})

        out_of_bounds = deepcopy(placements)
        out_of_bounds[0]["row"] = 15
        invalid.append({"placements": out_of_bounds, "score": None, "kind": "invalid"})

    candidates = [
        {"placements": deepcopy(optimal["placements"]), "score": int(optimal["score"]), "kind": "optimal"},
        *[
            {
                "placements": deepcopy(item["placements"]),
                "score": int(item["score"]),
                "kind": "legal_suboptimal",
            }
            for item in lower_legal[:2]
        ],
        *invalid,
    ][:limit]
    if len(candidates) < 2:
        raise RuntimeError(f"Not enough ranking candidates for {position['id']}")
    candidates.sort(key=lambda item: _stable_order(item["placements"], seed + 1))
    return candidates


def full_move_record(position: dict[str, Any], *, board_encoding: str = "dense") -> dict[str, Any]:
    return {
        "id": f"{position['id']}--v4-full",
        "source_id": position["id"],
        "transform": "identity",
        "position_key": position_key(position),
        "optimal_score": int(position["optimal_score"]),
        "record_type": "v4-full-move",
        "messages": prompt_for_position(position, board_encoding=board_encoding)
        + [{"role": "assistant", "content": assistant_payload(position)}],
    }


def recovery_record(
    position: dict[str, Any],
    lexicon: Lexicon,
    *,
    board_encoding: str = "dense",
) -> dict[str, Any]:
    rejected, error = _invalid_attempt(position, lexicon)
    return {
        "id": f"{position['id']}--v4-recovery",
        "source_id": position["id"],
        "transform": "identity",
        "position_key": position_key(position),
        "optimal_score": int(position["optimal_score"]),
        "record_type": "v4-verifier-recovery",
        "messages": prompt_for_position(
            position,
            {"raw_response": rejected, "error": error},
            board_encoding=board_encoding,
        )
        + [{"role": "assistant", "content": assistant_payload(position)}],
    }


def ranking_record(
    position: dict[str, Any],
    *,
    seed: int = 3407,
    board_encoding: str = "dense",
) -> dict[str, Any]:
    messages = prompt_for_position(position, board_encoding=board_encoding)
    user = json.loads(messages[-1]["content"])
    candidates = _ranking_candidates(position, seed=seed)
    user["instruction"] = (
        "Training drill: choose the highest-scoring legal move from candidate_moves. "
        "Some candidates may be illegal. Reply with that move as the normal raw play_move JSON only."
    )
    user["candidate_moves"] = [
        {
            "candidate": index,
            "placements": [
                {
                    "row": int(item["row"]),
                    "col": int(item["col"]),
                    "letter": str(item["letter"]).upper(),
                }
                for item in candidate["placements"]
            ],
        }
        for index, candidate in enumerate(candidates)
    ]
    messages[-1]["content"] = json.dumps(user, separators=(",", ":"))
    messages.append({"role": "assistant", "content": assistant_payload(position)})
    return {
        "id": f"{position['id']}--v4-ranking",
        "source_id": position["id"],
        "transform": "identity",
        "position_key": position_key(position),
        "optimal_score": int(position["optimal_score"]),
        "record_type": "v4-candidate-ranking",
        "messages": messages,
    }


def preference_record(
    position: dict[str, Any],
    lexicon: Lexicon,
    *,
    seed: int = 3407,
    board_encoding: str = "dense",
) -> dict[str, Any]:
    candidates = _ranking_candidates(position, seed=seed)
    rejected_candidate = next(
        (item for item in candidates if item["kind"] == "legal_suboptimal"),
        next(item for item in candidates if item["kind"] == "invalid"),
    )
    rejected = move_payload(rejected_candidate["placements"])
    rejected_score: int | None = None
    rejected_error: str | None = None
    try:
        move = validate_and_score_move(
            lexicon,
            grid_from_position(position["board"]),
            str(position["rack"]),
            rejected_candidate["placements"],
        )
        rejected_score = move.score
    except Exception as error:
        rejected_error = str(error)
    return {
        "id": f"{position['id']}--v4-preference",
        "source_id": position["id"],
        "position_key": position_key(position),
        "record_type": "v4-verifier-preference",
        "prompt": prompt_for_position(position, board_encoding=board_encoding),
        "chosen": assistant_payload(position),
        "rejected": rejected,
        "rejected_score": rejected_score,
        "rejected_error": rejected_error,
        "error_category": "legal_suboptimal" if rejected_error is None else error_category(rejected_error),
        "optimal_score": int(position["optimal_score"]),
    }


def verifier_reward(
    position: dict[str, Any],
    completion: str,
    lexicon: Lexicon,
    *,
    completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Score a completion using only parser and exact Scrabble-verifier facts."""
    result: dict[str, Any] = {
        "reward": -1.0,
        "parseable": False,
        "legal": False,
        "score": 0,
        "optimal_score": int(position["optimal_score"]),
        "optimal": False,
        "error": None,
        "error_category": None,
    }
    try:
        payload = parse_tool_payload(completion)
        result["parseable"] = True
        placements = payload["arguments"]["placements"]
        move = validate_and_score_move(
            lexicon,
            grid_from_position(position["board"]),
            str(position["rack"]),
            placements,
        )
    except Exception as error:
        message = str(error)
        category = error_category(message)
        # Partial shaping distinguishes all-invalid groups without letting an
        # invalid move collect any score reward.
        shaped = {
            "parse": -1.0,
            "bounds": -0.8,
            "rack": -0.65,
            "collision": -0.55,
            "geometry": -0.45,
            "connection": -0.35,
            "main_word": -0.25,
            "cross_word": -0.20,
            "unknown": -0.70,
        }.get(category, -0.70)
        result.update(reward=shaped, error=message, error_category=category)
        return result

    ratio = min(move.score / max(int(position["optimal_score"]), 1), 1.0)
    optimal = move.score == int(position["optimal_score"])
    reward = 1.0 + ratio + (0.25 if optimal else 0.0)
    if completion_tokens is not None:
        reward -= min(0.20, 0.002 * max(completion_tokens - 96, 0))
    result.update(
        reward=reward,
        legal=True,
        score=move.score,
        optimal=optimal,
        error=None,
        error_category=None,
    )
    return result
