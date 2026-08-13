from __future__ import annotations

import json
from typing import Any

from .constants import BINGO_BONUS, LETTER_MULTIPLIERS, LETTER_VALUES, WORD_MULTIPLIERS
from .lexicon import Lexicon
from .runner import dense_board_text, prompt_for_position
from .solver import (
    BoardTile,
    grid_from_position,
    infer_orientation,
    read_word,
    score_word,
    validate_and_score_move,
    word_text,
)
from .training import assistant_payload, position_key, transform_position


def _word_audit(
    word_tiles: list[tuple[int, int, BoardTile]],
    new_positions: set[tuple[int, int]],
) -> dict[str, Any]:
    tiles = []
    letter_subtotal = 0
    word_multiplier = 1
    for row, col, tile in word_tiles:
        is_new = (row, col) in new_positions
        letter_multiplier = LETTER_MULTIPLIERS[row][col] if is_new else 1
        square_word_multiplier = WORD_MULTIPLIERS[row][col] if is_new else 1
        value = 0 if tile.is_blank else LETTER_VALUES[tile.letter]
        contribution = value * letter_multiplier
        letter_subtotal += contribution
        word_multiplier *= square_word_multiplier
        tiles.append(
            {
                "row": row,
                "col": col,
                "letter": tile.letter,
                "is_new": is_new,
                "is_blank": tile.is_blank,
                "letter_value": value,
                "letter_multiplier": letter_multiplier,
                "word_multiplier": square_word_multiplier,
                "contribution": contribution,
            }
        )
    return {
        "word": word_text(word_tiles),
        "tiles": tiles,
        "letter_subtotal": letter_subtotal,
        "word_multiplier": word_multiplier,
        "score": letter_subtotal * word_multiplier,
    }


def analyze_legal_move(
    position: dict[str, Any],
    move_data: dict[str, Any],
    lexicon: Lexicon,
) -> dict[str, Any]:
    """Validate a move and expose the intermediate facts needed to reproduce it."""
    grid = grid_from_position(position["board"])
    validated = validate_and_score_move(
        lexicon,
        grid,
        str(position["rack"]),
        move_data["placements"],
    )
    placements = validated.placements
    orientation = infer_orientation(placements)
    new_grid = [row.copy() for row in grid]
    for placement in placements:
        new_grid[placement.row][placement.col] = BoardTile(
            placement.letter,
            placement.is_blank,
        )
    new_positions = {(item.row, item.col) for item in placements}
    word_tiles: list[list[tuple[int, int, BoardTile]]] = []
    main_word_index = 0
    if orientation == "across":
        row = placements[0].row
        cols = [item.col for item in placements]
        word_tiles.append(read_word(new_grid, row, min(cols), 0, 1))
        for item in placements:
            cross = read_word(new_grid, item.row, item.col, 1, 0)
            if len(cross) > 1:
                word_tiles.append(cross)
    elif orientation == "down":
        col = placements[0].col
        rows = [item.row for item in placements]
        word_tiles.append(read_word(new_grid, min(rows), col, 1, 0))
        for item in placements:
            cross = read_word(new_grid, item.row, item.col, 0, 1)
            if len(cross) > 1:
                word_tiles.append(cross)
    else:
        item = placements[0]
        horizontal = read_word(new_grid, item.row, item.col, 0, 1)
        vertical = read_word(new_grid, item.row, item.col, 1, 0)
        word_tiles.extend(word for word in (horizontal, vertical) if len(word) > 1)
        if not word_tiles:
            word_tiles.append(horizontal)

    unique_words: list[list[tuple[int, int, BoardTile]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for word in word_tiles:
        key = tuple((row, col) for row, col, _ in word)
        if key not in seen:
            seen.add(key)
            unique_words.append(word)
    audits = [_word_audit(word, new_positions) for word in unique_words]
    word_score = sum(item["score"] for item in audits)
    bingo_bonus = BINGO_BONUS if len(placements) == 7 else 0
    if word_score + bingo_bonus != validated.score:
        raise RuntimeError(
            f"Score audit mismatch for {position['id']}: "
            f"{word_score}+{bingo_bonus}!={validated.score}"
        )
    main = audits[main_word_index]
    existing_contacts = [
        {"row": tile["row"], "col": tile["col"], "letter": tile["letter"]}
        for tile in main["tiles"]
        if not tile["is_new"]
    ]
    return {
        "orientation": orientation or "single-tile",
        "main_word": main["word"],
        "cross_words": [item["word"] for item in audits[1:]],
        "rack_used": ["?" if item.is_blank else item.letter for item in placements],
        "placements": [item.to_dict() for item in placements],
        "existing_contacts": existing_contacts,
        "word_breakdown": audits,
        "word_score": word_score,
        "bingo_bonus": bingo_bonus,
        "total_score": validated.score,
        "rack_ok": True,
        "geometry_ok": True,
        "all_words_valid": True,
        "connected": True,
    }


def _candidate_moves(position: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    candidates = list(position.get("candidate_moves") or position.get("optimal_moves") or [])
    canonical = position["canonical_optimal_move"]
    canonical_key = json.dumps(canonical["placements"], sort_keys=True)
    if all(json.dumps(item["placements"], sort_keys=True) != canonical_key for item in candidates):
        candidates.insert(0, canonical)
    return sorted(
        candidates,
        key=lambda item: (-int(item["score"]), json.dumps(item["placements"], sort_keys=True)),
    )[:limit]


def process_completion(
    position: dict[str, Any],
    lexicon: Lexicon,
    *,
    candidate_limit: int = 3,
) -> str:
    analyses = [
        analyze_legal_move(position, move, lexicon)
        for move in _candidate_moves(position, candidate_limit)
    ]
    chosen = analyze_legal_move(position, position["canonical_optimal_move"], lexicon)
    lines = [
        f"Board reconstruction: {len(position['board'])} occupied squares; rack={position['rack']}.",
        "Candidate verification:",
    ]
    for index, item in enumerate(analyses, start=1):
        word_scores = ", ".join(
            f"{word['word']}={word['letter_subtotal']}x{word['word_multiplier']}={word['score']}"
            for word in item["word_breakdown"]
        )
        placements = ", ".join(
            f"({tile['row']},{tile['col']})={tile['letter']}"
            for tile in item["placements"]
        )
        crosses = ",".join(item["cross_words"]) or "none"
        lines.append(
            f"{index}. {item['main_word']} {item['orientation']}; new={placements}; "
            f"crosses={crosses}; rack_ok=yes; connected=yes; words_valid=yes; "
            f"score=[{word_scores}]+bingo({item['bingo_bonus']})={item['total_score']}."
        )
    lines.extend(
        [
            f"Choice: {chosen['main_word']} for verified maximum {chosen['total_score']}.",
            "Final check: emit only newly placed tiles; coordinates are in bounds, "
            "rack-supplied, collinear, connected, and all formed words are valid.",
            "</think>",
            assistant_payload(position),
        ]
    )
    return "\n".join(lines)


def v3_training_record(
    position: dict[str, Any],
    lexicon: Lexicon,
    transform_name: str = "identity",
    *,
    candidate_limit: int = 3,
) -> dict[str, Any]:
    item = (
        transform_position(position, transform_name)
        if transform_name != "identity"
        else json.loads(json.dumps(position))
    )
    messages = prompt_for_position(item, board_encoding="dense")
    messages.append(
        {
            "role": "assistant",
            "content": process_completion(item, lexicon, candidate_limit=candidate_limit),
        }
    )
    return {
        "id": f"{item['id']}--v3-move",
        "source_id": position["id"],
        "transform": transform_name,
        "position_key": position_key(item),
        "optimal_score": int(item["optimal_score"]),
        "record_type": "v3-process-move",
        "messages": messages,
    }


def v3_audit_record(
    position: dict[str, Any],
    lexicon: Lexicon,
    transform_name: str = "identity",
) -> dict[str, Any]:
    item = (
        transform_position(position, transform_name)
        if transform_name != "identity"
        else json.loads(json.dumps(position))
    )
    audit = analyze_legal_move(item, item["canonical_optimal_move"], lexicon)
    candidate = {
        "rack": list(item["rack"]),
        "board_grid": dense_board_text(item).splitlines(),
        "placements": [
            {"row": p["row"], "col": p["col"], "letter": p["letter"]}
            for p in item["canonical_optimal_move"]["placements"]
        ],
    }
    answer = {
        "legal": True,
        "rack_ok": audit["rack_ok"],
        "geometry_ok": audit["geometry_ok"],
        "connected": audit["connected"],
        "main_word": audit["main_word"],
        "cross_words": audit["cross_words"],
        "score": audit["total_score"],
    }
    crosses = ",".join(audit["cross_words"]) or "none"
    answer_content = "\n".join(
        [
            f"Rack and geometry checks pass. Main word={audit['main_word']}; "
            f"cross words={crosses}; verified score={audit['total_score']}.",
            "</think>",
            json.dumps(answer, separators=(",", ":")),
        ]
    )
    return {
        "id": f"{item['id']}--v3-audit",
        "source_id": position["id"],
        "transform": transform_name,
        "position_key": position_key(item),
        "record_type": "v3-legal-move-audit",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Audit the proposed Scrabble move. Reconstruct the full main and cross "
                    "words and return one JSON object with legality fields and exact score."
                ),
            },
            {"role": "user", "content": json.dumps(candidate, separators=(",", ":"))},
            {"role": "assistant", "content": answer_content},
        ],
    }
