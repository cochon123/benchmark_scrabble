from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from .lexicon import Lexicon
from .runner import prompt_for_position
from .training import assistant_payload, position_key, transform_position
from .v3_training import analyze_legal_move
from .v4_diagnostic import clean_placements, placement_key


def _candidate_moves(position: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    moves = position.get("candidate_moves") or position.get("optimal_moves") or []
    unique: dict[str, dict[str, Any]] = {}
    for move in moves:
        placements = clean_placements(move["placements"])
        unique.setdefault(placement_key(placements), {**move, "placements": placements})
    return sorted(
        unique.values(),
        key=lambda item: (-int(item["score"]), placement_key(item["placements"])),
    )[:limit]


def _compact_candidate(analysis: dict[str, Any]) -> str:
    main_tiles = analysis["word_breakdown"][0]["tiles"]
    first = min(main_tiles, key=lambda item: (int(item["row"]), int(item["col"])))
    direction = "A" if len({int(item["row"]) for item in main_tiles}) == 1 else "D"
    crosses = ",".join(analysis["cross_words"]) or "-"
    return (
        f"{analysis['main_word']}@{int(first['row'])},{int(first['col'])}{direction}"
        f"/{crosses}={int(analysis['total_score'])}"
    )


def compact_reasoning_completion(position: dict[str, Any], lexicon: Lexicon) -> str:
    """A short verified search trace for a thinking-model completion.

    The opening ``<think>`` token is supplied by Qwen's thinking chat template,
    so the completion starts with trace text and explicitly closes reasoning.
    """
    analyses = [analyze_legal_move(position, move, lexicon) for move in _candidate_moves(position)]
    winner = analyze_legal_move(position, position["canonical_optimal_move"], lexicon)
    contact_text = ",".join(
        f"{item['letter']}@{item['row']},{item['col']}" for item in winner["existing_contacts"]
    ) or "adjacent"
    new_text = ",".join(
        f"{item['letter']}@{item['row']},{item['col']}" for item in winner["placements"]
    )
    used = "".join(winner["rack_used"])
    candidates = ";".join(_compact_candidate(item) for item in analyses)
    words = ",".join(item["word"] for item in winner["word_breakdown"])
    return "\n".join(
        [
            f"SCAN rack={position['rack']}; inspect both directions at occupied-edge anchors.",
            f"TRY {candidates}",
            f"VERIFY {winner['main_word']}: new={new_text}; use={used}; "
            f"existing={contact_text}; "
            f"words={words}; score={winner['word_score']}+bingo{winner['bingo_bonus']}="
            f"{winner['total_score']}.",
            f"CHOOSE {winner['main_word']}={winner['total_score']}",
            "</think>",
            assistant_payload(position),
        ]
    )


def compact_reasoning_record(
    source: dict[str, Any],
    lexicon: Lexicon,
    *,
    transform: str = "identity",
) -> dict[str, Any]:
    position = transform_position(source, transform) if transform != "identity" else deepcopy(source)
    messages = prompt_for_position(position, board_encoding="dense")
    messages.append(
        {"role": "assistant", "content": compact_reasoning_completion(position, lexicon)}
    )
    return {
        "id": f"{source['id']}--{transform}--v10-compact-reasoning",
        "source_id": source["id"],
        "source_game_id": source["source_game_id"],
        "transform": transform,
        "position_key": position_key(position),
        "optimal_score": int(position["optimal_score"]),
        "record_type": "v10-compact-reasoning",
        "messages": messages,
    }


def normalize_teacher_completion(response: str) -> str:
    """Keep a bounded visible trace and the final play_move object."""
    text = response.strip()
    json_match = re.search(
        r'(\{"tool"\s*:\s*"play_move".*\})\s*$',
        text,
        flags=re.DOTALL,
    )
    if not json_match:
        raise ValueError("Teacher response has no final play_move JSON")
    payload = json.loads(json_match.group(1))
    trace = text[: json_match.start()].strip()
    if "<think>" in trace:
        trace = trace.split("<think>", 1)[1]
    if "</think>" in trace:
        trace = trace.split("</think>", 1)[0]
    words = trace.split()
    if len(words) > 350:
        words = words[:350]
    trace = " ".join(words).strip()
    compact_json = json.dumps(payload, separators=(",", ":"))
    return f"{trace}\n</think>\n{compact_json}"


def teacher_reasoning_record(
    position: dict[str, Any], response: str, *, replica: int
) -> dict[str, Any]:
    messages = prompt_for_position(position, board_encoding="dense")
    messages.append(
        {"role": "assistant", "content": normalize_teacher_completion(response)}
    )
    return {
        "id": f"{position['id']}--teacher-{replica}",
        "source_id": position["id"],
        "source_game_id": position["source_game_id"],
        "transform": "identity",
        "position_key": position_key(position),
        "optimal_score": int(position["optimal_score"]),
        "record_type": "v10-kimi-reasoning",
        "messages": messages,
    }
