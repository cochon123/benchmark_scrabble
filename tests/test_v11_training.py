from __future__ import annotations

from scrabble_bench.lexicon import Lexicon, TrieNode
from scrabble_bench.v11_training import (
    SQUARE_TOKENS,
    action_text,
    action_to_placements,
    parse_action,
    policy_messages,
    square_token,
    tied_board_text,
    token_square,
)


def tiny_lexicon(*words: str) -> Lexicon:
    root = TrieNode()
    normalized = {word.upper() for word in words}
    for word in normalized:
        node = root
        for letter in word:
            node = node.children.setdefault(letter, TrieNode())
        node.is_word = True
    return Lexicon(normalized, root)


def opening() -> dict:
    return {
        "id": "opening",
        "source_game_id": "test",
        "board": [],
        "rack": "ATXXXXX",
        "optimal_score": 4,
        "canonical_optimal_move": {
            "placements": [
                {"row": 7, "col": 7, "letter": "A"},
                {"row": 7, "col": 8, "letter": "T"},
            ]
        },
    }


def test_square_tokens_are_unique_and_round_trip() -> None:
    assert len(SQUARE_TOKENS) == 225
    assert len(set(SQUARE_TOKENS)) == 225
    for row, col in ((0, 0), (7, 7), (14, 14)):
        assert token_square(square_token(row, col)) == (row, col)


def test_action_round_trip_and_deterministic_renderer() -> None:
    position = opening()
    text = action_text({"word": "AT", "row": 7, "col": 7, "direction": "across"})
    assert parse_action(text) == {"word": "AT", "row": 7, "col": 7, "direction": "across"}
    placements, score = action_to_placements(position, text, tiny_lexicon("AT"))
    assert [(x["row"], x["col"], x["letter"]) for x in placements] == [
        (7, 7, "A"),
        (7, 8, "T"),
    ]
    assert score == 4


def test_board_and_retry_use_the_same_pointer_tokens_without_answer() -> None:
    position = opening()
    board = tied_board_text(position)
    center = square_token(7, 7)
    assert f"{center}:." in board
    messages = policy_messages(position, rejection_history=[f"NO|{center}|A"])
    assert messages[-2]["content"] == f"NO|{center}|A"
    assert "solution" in messages[-1]["content"].lower()
    assert "AT|" not in messages[-1]["content"]
