from __future__ import annotations

import json
import unittest
from pathlib import Path

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import parse_tool_payload
from scrabble_bench.solver import grid_from_position, validate_and_score_move
from scrabble_bench.v41_training import anchor_direction_record, move_plan, plan_copy_record, word_direction_record


class V41TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lexicon = Lexicon.from_path(resolve_lexicon_path())
        positions = json.loads(Path("data/general_v4_pilot/train_positions.json").read_text())
        cls.position = next(item for item in positions if int(item["band_ply"]) > 0)

    def test_plan_reconstructs_canonical_move_metadata(self) -> None:
        plan = move_plan(self.position, self.position["canonical_optimal_move"]["placements"], self.lexicon)
        self.assertIn(plan["direction"], {"across", "down"})
        self.assertTrue(plan["word"])
        self.assertGreaterEqual(plan["anchor_row"], 0)

    def test_partial_scaffold_targets_remain_exactly_optimal(self) -> None:
        for builder in (plan_copy_record, word_direction_record, anchor_direction_record):
            with self.subTest(builder=builder.__name__):
                record = builder(self.position, self.lexicon)
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
