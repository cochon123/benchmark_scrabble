from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from .constants import BOARD_SIZE
from .lexicon import Lexicon
from .runner import dense_board_text
from .training import assistant_payload, position_key
from .v4_diagnostic import clean_placements
from .v41_training import move_plan
from .v42_training import placements_for_start


AXIS_LABELS = tuple("ABCDEFGHIJKLMNO")
_PLAN_RE = re.compile(
    r"^ROW_([A-O])\|COL_([A-O])\|DIR_([AD])\|WORD_([A-Z]+)$"
)
_LOCATION_RE = re.compile(r"^ROW_([A-O])\|COL_([A-O])$")
_ROW_RE = re.compile(r"^ROW_([A-O])$")
_COL_RE = re.compile(r"^COL_([A-O])$")
_WORD_RE = re.compile(r"^WORD_([A-Z]+)$")


def axis_token(prefix: str, index: int) -> str:
    if prefix not in {"ROW", "COL"}:
        raise ValueError(f"Unsupported coordinate prefix: {prefix}")
    if not 0 <= index < BOARD_SIZE:
        raise ValueError(f"Coordinate is outside the board: {index}")
    return f"{prefix}_{AXIS_LABELS[index]}"


def decode_axis(label: str) -> int:
    try:
        return AXIS_LABELS.index(label)
    except ValueError as error:
        raise ValueError(f"Unknown axis label: {label}") from error


def factorized_plan(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = move_plan(
        position,
        clean_placements(position["canonical_optimal_move"]["placements"]),
        lexicon,
    )
    return {
        "row": int(plan["start_row"]),
        "col": int(plan["start_col"]),
        "direction": str(plan["direction"]),
        "word": str(plan["word"]).upper(),
    }


def factorized_plan_text(plan: dict[str, Any]) -> str:
    direction = "A" if str(plan["direction"]) == "across" else "D"
    return "|".join(
        [
            axis_token("ROW", int(plan["row"])),
            axis_token("COL", int(plan["col"])),
            f"DIR_{direction}",
            f"WORD_{str(plan['word']).upper()}",
        ]
    )


def parse_factorized_plan(text: str) -> dict[str, Any]:
    match = _PLAN_RE.fullmatch(text.strip())
    if not match:
        raise ValueError("Response did not match the factorized action grammar.")
    row, col, direction, word = match.groups()
    return {
        "row": decode_axis(row),
        "col": decode_axis(col),
        "direction": "across" if direction == "A" else "down",
        "word": word,
    }


def parse_factorized_location(text: str) -> tuple[int, int]:
    match = _LOCATION_RE.fullmatch(text.strip())
    if not match:
        raise ValueError("Response did not match the factorized location grammar.")
    return decode_axis(match.group(1)), decode_axis(match.group(2))


def parse_row_token(text: str) -> int:
    match = _ROW_RE.fullmatch(text.strip())
    if not match:
        raise ValueError("Response did not contain one row token.")
    return decode_axis(match.group(1))


def parse_col_token(text: str) -> int:
    match = _COL_RE.fullmatch(text.strip())
    if not match:
        raise ValueError("Response did not contain one column token.")
    return decode_axis(match.group(1))


def parse_word_token(text: str) -> str:
    match = _WORD_RE.fullmatch(text.strip())
    if not match:
        raise ValueError("Response did not contain one word token.")
    return match.group(1)


def plan_to_placements(
    position: dict[str, Any], plan: dict[str, Any], lexicon: Lexicon
) -> tuple[list[dict[str, Any]], int]:
    return placements_for_start(
        position,
        lexicon,
        word=str(plan["word"]),
        direction=str(plan["direction"]),
        start_row=int(plan["row"]),
        start_col=int(plan["col"]),
    )


def factorized_messages(
    position: dict[str, Any],
    *,
    mode: str,
    instruction: str,
    response_format: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    mapping = ",".join(f"{label}={index:02d}" for index, label in enumerate(AXIS_LABELS))
    system = "\n".join(
        [
            "You are a Scrabble factorized-action policy.",
            "The board is 15 by 15 and its displayed numeric coordinates are zero-indexed.",
            f"Categorical coordinate mapping for both axes: {mapping}.",
            "DIR_A means across and DIR_D means down.",
            "A word plan names the complete main word, including existing crossing letters.",
            instruction,
            f"Return exactly: {response_format}",
            "Do not include JSON, prose, markdown, or extra whitespace.",
        ]
    )
    payload: dict[str, Any] = {
        "task_mode": mode,
        "rack": list(str(position["rack"])),
        "board_encoding": "dense-grid",
        "board_grid": dense_board_text(position).splitlines(),
    }
    if extra:
        payload.update(extra)
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
        "source_game_id": position["source_game_id"],
        "transform": "identity",
        "position_key": position_key(position),
        "optimal_score": int(position["optimal_score"]),
        "record_type": record_type,
        "messages": messages + [{"role": "assistant", "content": assistant}],
    }


def row_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = factorized_plan(position, lexicon)
    messages = factorized_messages(
        position,
        mode="V7_ROW",
        instruction="The optimal main word and direction are supplied. Classify its optimal start row.",
        response_format="ROW_X",
        extra={"word": plan["word"], "direction": plan["direction"]},
    )
    return _record(position, "v7-row", messages, axis_token("ROW", plan["row"]))


def col_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = factorized_plan(position, lexicon)
    row = axis_token("ROW", plan["row"])
    messages = factorized_messages(
        position,
        mode="V7_COLUMN",
        instruction="The optimal word, direction, and start row are supplied. Classify its optimal start column.",
        response_format="COL_X",
        extra={"word": plan["word"], "direction": plan["direction"], "start_row_token": row},
    )
    return _record(position, "v7-column", messages, axis_token("COL", plan["col"]))


def location_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = factorized_plan(position, lexicon)
    messages = factorized_messages(
        position,
        mode="V7_LOCATION",
        instruction="The optimal main word and direction are supplied. Classify both coordinates of its optimal start.",
        response_format="ROW_X|COL_X",
        extra={"word": plan["word"], "direction": plan["direction"]},
    )
    target = f"{axis_token('ROW', plan['row'])}|{axis_token('COL', plan['col'])}"
    return _record(position, "v7-location", messages, target)


def direction_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = factorized_plan(position, lexicon)
    messages = factorized_messages(
        position,
        mode="V7_DIRECTION",
        instruction="The optimal main word is supplied. Classify the direction of its highest-scoring placement.",
        response_format="DIR_A_OR_D",
        extra={"word": plan["word"]},
    )
    target = "DIR_A" if plan["direction"] == "across" else "DIR_D"
    return _record(position, "v7-direction", messages, target)


def word_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = factorized_plan(position, lexicon)
    messages = factorized_messages(
        position,
        mode="V7_WORD",
        instruction="The optimal start and direction are supplied. Return the complete optimal main word.",
        response_format="WORD_TEXT",
        extra={
            "start_row_token": axis_token("ROW", plan["row"]),
            "start_col_token": axis_token("COL", plan["col"]),
            "direction_token": "DIR_A" if plan["direction"] == "across" else "DIR_D",
        },
    )
    return _record(position, "v7-word", messages, f"WORD_{plan['word']}")


def plan_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = factorized_plan(position, lexicon)
    messages = factorized_messages(
        position,
        mode="V7_FREE_PLAN",
        instruction="Choose the highest-scoring legal move and express it in the fixed factorized action grammar.",
        response_format="ROW_X|COL_X|DIR_A_OR_D|WORD_TEXT",
    )
    return _record(position, "v7-free-plan", messages, factorized_plan_text(plan))


def plan_to_move_record(position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = factorized_plan(position, lexicon)
    from .v41_training import _tagged_prompt

    messages, user = _tagged_prompt(
        position,
        "V7_PLAN_TO_MOVE",
        "The selected factorized plan is supplied. Convert it into only the new tile placements and reply with raw play_move JSON.",
    )
    user["selected_factorized_plan"] = factorized_plan_text(plan)
    messages[-1]["content"] = json.dumps(user, separators=(",", ":"))
    return _record(position, "v7-plan-to-move", messages, assistant_payload(position))


def correction_record(
    position: dict[str, Any],
    lexicon: Lexicon,
    *,
    previous_attempt: str,
    first_error: str,
) -> dict[str, Any]:
    plan = factorized_plan(position, lexicon)
    messages = factorized_messages(
        position,
        mode="V7_ONPOLICY_CORRECTION",
        instruction="Correct the policy's previous factorized move after deterministic verifier feedback.",
        response_format="ROW_X|COL_X|DIR_A_OR_D|WORD_TEXT",
        extra={
            "previous_attempt": previous_attempt,
            "verifier_first_error": first_error,
        },
    )
    return _record(position, "v7-onpolicy-correction", messages, factorized_plan_text(plan))


def diagnostic_record(
    position: dict[str, Any], lexicon: Lexicon, *, task: str
) -> dict[str, Any]:
    builders = {
        "row": row_record,
        "column": col_record,
        "location": location_record,
        "word": word_record,
        "free-plan": plan_record,
    }
    try:
        record = builders[task](position, lexicon)
    except KeyError as error:
        raise ValueError(f"Unsupported V7 diagnostic task: {task}") from error
    plan = factorized_plan(position, lexicon)
    return {
        "id": record["id"],
        "source_id": position["id"],
        "source_game_id": position["source_game_id"],
        "band_ply": int(position["band_ply"]),
        "task": task,
        "position": deepcopy(position),
        "messages": record["messages"][:-1],
        "target_content": record["messages"][-1]["content"],
        "target_plan": plan,
    }
