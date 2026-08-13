from __future__ import annotations

import json
import unittest
from pathlib import Path

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import parse_tool_payload
from scrabble_bench.solver import grid_from_position, validate_and_score_move
from scrabble_bench.v7_training import (
    AXIS_LABELS,
    axis_token,
    col_record,
    correction_record,
    factorized_plan,
    factorized_plan_text,
    location_record,
    parse_factorized_location,
    parse_factorized_plan,
    plan_record,
    plan_to_move_record,
    plan_to_placements,
    row_record,
    word_record,
)


class V7TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lexicon = Lexicon.from_path(resolve_lexicon_path())
        positions = json.loads(Path("data/general_v6_positions/train_positions.json").read_text())
        cls.position = next(row for row in positions if int(row["band_ply"]) >= 6)

    def test_axis_tokens_cover_board_exactly(self) -> None:
        self.assertEqual(len(AXIS_LABELS), 15)
        self.assertEqual(axis_token("ROW", 0), "ROW_A")
        self.assertEqual(axis_token("COL", 14), "COL_O")
        with self.assertRaises(ValueError):
            axis_token("ROW", 15)

    def test_factorized_plan_round_trip_is_optimal(self) -> None:
        target = factorized_plan(self.position, self.lexicon)
        text = factorized_plan_text(target)
        parsed = parse_factorized_plan(text)
        self.assertEqual(parsed, target)
        placements, score = plan_to_placements(self.position, parsed, self.lexicon)
        self.assertTrue(placements)
        self.assertEqual(score, self.position["optimal_score"])

    def test_component_records_use_fixed_short_grammar(self) -> None:
        row = row_record(self.position, self.lexicon)["messages"][-1]["content"]
        col = col_record(self.position, self.lexicon)["messages"][-1]["content"]
        location = location_record(self.position, self.lexicon)["messages"][-1]["content"]
        word = word_record(self.position, self.lexicon)["messages"][-1]["content"]
        self.assertRegex(row, r"^ROW_[A-O]$")
        self.assertRegex(col, r"^COL_[A-O]$")
        self.assertEqual(parse_factorized_location(location), (
            factorized_plan(self.position, self.lexicon)["row"],
            factorized_plan(self.position, self.lexicon)["col"],
        ))
        self.assertRegex(word, r"^WORD_[A-Z]+$")

    def test_plan_to_move_target_remains_verifier_optimal(self) -> None:
        record = plan_to_move_record(self.position, self.lexicon)
        payload = parse_tool_payload(record["messages"][-1]["content"])
        move = validate_and_score_move(
            self.lexicon,
            grid_from_position(self.position["board"]),
            self.position["rack"],
            payload["arguments"]["placements"],
        )
        self.assertEqual(move.score, self.position["optimal_score"])

    def test_correction_exposes_attempt_but_not_hidden_solver_candidates(self) -> None:
        record = correction_record(
            self.position,
            self.lexicon,
            previous_attempt="ROW_A|COL_A|DIR_A|WORD_BAD",
            first_error="row",
        )
        prompt = record["messages"][-2]["content"]
        self.assertIn("previous_attempt", prompt)
        self.assertIn('"verifier_first_error":"row"', prompt)
        self.assertNotIn("candidate", prompt.lower())
        self.assertEqual(
            record["messages"][-1]["content"],
            plan_record(self.position, self.lexicon)["messages"][-1]["content"],
        )


if __name__ == "__main__":
    unittest.main()
