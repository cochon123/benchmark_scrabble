from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from scripts.generate_training_data import _candidate_sample


@dataclass
class Candidate:
    score: int


class V6PipelineTests(unittest.TestCase):
    def test_stratified_candidates_preserve_head_and_reach_tail(self) -> None:
        moves = [Candidate(score=100 - index) for index in range(100)]
        selected = _candidate_sample(moves, 16, "stratified")
        self.assertEqual([item.score for item in selected[:4]], [100, 99, 98, 97])
        self.assertEqual(len(selected), 16)
        self.assertEqual(selected[-1].score, 1)

    def test_top_strategy_is_unchanged(self) -> None:
        moves = [Candidate(score=10 - index) for index in range(10)]
        self.assertEqual(_candidate_sample(moves, 3, "top"), moves[:3])

    def test_invalid_strategy_fails(self) -> None:
        with self.assertRaises(ValueError):
            _candidate_sample([Candidate(score=index) for index in range(5)], 2, "bad")


if __name__ == "__main__":
    unittest.main()
