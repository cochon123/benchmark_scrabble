from __future__ import annotations

import json
import unittest

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import parse_tool_payload
from scrabble_bench.solver import grid_from_position, validate_and_score_move
from scrabble_bench.training import (
    TRANSFORMS,
    recovery_training_record,
    training_record,
    transform_position,
)


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.position = json.loads(DATASET_PATH.read_text(encoding="utf-8"))[0]
        cls.lexicon = Lexicon.from_path(resolve_lexicon_path())

    def test_all_board_symmetries_preserve_move_score(self) -> None:
        for name in TRANSFORMS:
            with self.subTest(name=name):
                position = transform_position(self.position, name)
                move = validate_and_score_move(
                    self.lexicon,
                    grid_from_position(position["board"]),
                    position["rack"],
                    position["canonical_optimal_move"]["placements"],
                )
                self.assertEqual(move.score, self.position["optimal_score"])

    def test_training_record_has_parseable_assistant_payload(self) -> None:
        record = training_record(self.position)
        payload = parse_tool_payload(record["messages"][-1]["content"])
        self.assertEqual(
            payload["arguments"]["placements"],
            [
                {"row": item["row"], "col": item["col"], "letter": item["letter"]}
                for item in self.position["canonical_optimal_move"]["placements"]
            ],
        )

    def test_reasoning_record_ends_with_parseable_payload(self) -> None:
        record = training_record(self.position, include_reasoning=True)
        payload = parse_tool_payload(record["messages"][-1]["content"])
        self.assertEqual(payload["tool"], "play_move")
        self.assertIn("</think>", record["messages"][-1]["content"])

    def test_recovery_record_contains_rejection_and_correct_move(self) -> None:
        record = recovery_training_record(
            self.position,
            self.lexicon,
            include_reasoning=True,
        )
        self.assertEqual(record["record_type"], "invalid-move-recovery")
        self.assertTrue(record["rejection_error"])
        self.assertEqual(len(record["messages"]), 5)
        payload = parse_tool_payload(record["messages"][-1]["content"])
        move = validate_and_score_move(
            self.lexicon,
            grid_from_position(self.position["board"]),
            self.position["rack"],
            payload["arguments"]["placements"],
        )
        self.assertEqual(move.score, self.position["optimal_score"])


if __name__ == "__main__":
    unittest.main()
