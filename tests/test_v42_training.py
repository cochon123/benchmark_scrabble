from __future__ import annotations

import json
import unittest
from pathlib import Path

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import parse_tool_payload
from scrabble_bench.solver import grid_from_position, validate_and_score_move
from scrabble_bench.v42_training import (
    anchor_mask_record,
    cross_check_record,
    placements_for_start,
    start_candidates,
    start_choice_record,
    start_location_record,
    start_to_move_record,
    word_direction_move_record,
)
from scrabble_bench.v41_training import move_plan


class V42TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lexicon = Lexicon.from_path(resolve_lexicon_path())
        positions = json.loads(Path("data/general_v4_pilot/train_positions.json").read_text())
        cls.position = next(item for item in positions if int(item["band_ply"]) >= 4)

    def test_canonical_start_reconstructs_optimal_move(self) -> None:
        plan = move_plan(self.position, self.position["canonical_optimal_move"]["placements"], self.lexicon)
        placements, score = placements_for_start(
            self.position,
            self.lexicon,
            word=plan["word"],
            direction=plan["direction"],
            start_row=plan["start_row"],
            start_col=plan["start_col"],
        )
        self.assertEqual(score, self.position["optimal_score"])
        self.assertTrue(placements)

    def test_start_candidates_hide_labels_and_retain_target(self) -> None:
        candidates, targets = start_candidates(self.position, self.lexicon, seed=7421)
        self.assertLessEqual(len(candidates), 16)
        self.assertGreaterEqual(len(candidates), 8)
        record = start_choice_record(self.position, self.lexicon, seed=7421)
        prompt = record["messages"][-2]["content"]
        self.assertNotIn('"legal"', prompt)
        self.assertNotIn('"score"', prompt)
        target = json.loads(record["messages"][-1]["content"])["candidate"]
        visible = json.loads(prompt)["candidate_starts"][target]
        self.assertIn((visible["row"], visible["col"]), targets)

    def test_localization_targets_are_short_json(self) -> None:
        locate = start_location_record(self.position, self.lexicon)
        target = json.loads(locate["messages"][-1]["content"])
        self.assertEqual(set(target), {"start_row", "start_col"})
        anchor = anchor_mask_record(self.position, seed=7421)
        self.assertIn("anchor_candidates", json.loads(anchor["messages"][-1]["content"]))
        cross = cross_check_record(self.position, self.lexicon, seed=7421)
        letters = json.loads(cross["messages"][-1]["content"])["allowed_letters"]
        self.assertEqual(letters, "".join(sorted(letters)))

    def test_move_targets_remain_verifier_optimal(self) -> None:
        for builder in (start_to_move_record, word_direction_move_record):
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
