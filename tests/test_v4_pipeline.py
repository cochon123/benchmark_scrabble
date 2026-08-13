from __future__ import annotations

import json
import unittest

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import parse_tool_payload
from scrabble_bench.v4_training import (
    full_move_record,
    preference_record,
    ranking_record,
    recovery_record,
    verifier_reward,
)
from scripts.train_v4_group_rl import completion_end, standardized_advantages


class V4PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.position = json.loads(DATASET_PATH.read_text(encoding="utf-8"))[0]
        cls.lexicon = Lexicon.from_path(resolve_lexicon_path())

    def test_all_sft_tasks_end_in_optimal_benchmark_json(self) -> None:
        rows = [
            full_move_record(self.position),
            ranking_record(self.position),
            recovery_record(self.position, self.lexicon),
        ]
        for row in rows:
            completion = row["messages"][-1]["content"]
            parse_tool_payload(completion)
            result = verifier_reward(self.position, completion, self.lexicon)
            self.assertTrue(result["legal"], row["record_type"])
            self.assertTrue(result["optimal"], row["record_type"])
            self.assertNotIn("</think>", completion)

    def test_ranking_candidates_are_shuffled_and_hide_scores(self) -> None:
        row = ranking_record(self.position)
        prompt = json.loads(row["messages"][-2]["content"])
        candidates = prompt["candidate_moves"]
        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(all("score" not in item for item in candidates))
        target = parse_tool_payload(row["messages"][-1]["content"])
        self.assertIn(target["arguments"]["placements"], [item["placements"] for item in candidates])

    def test_preference_reward_orders_chosen_above_rejected(self) -> None:
        row = preference_record(self.position, self.lexicon)
        chosen = verifier_reward(self.position, row["chosen"], self.lexicon)
        rejected = verifier_reward(self.position, row["rejected"], self.lexicon)
        self.assertTrue(chosen["optimal"])
        self.assertGreater(chosen["reward"], rejected["reward"])

    def test_reward_ignores_claimed_score_and_rejects_empty_move(self) -> None:
        completion = 'claimed score 999\n{"tool":"play_move","arguments":{"placements":[]}}'
        result = verifier_reward(self.position, completion, self.lexicon)
        self.assertFalse(result["legal"])
        self.assertEqual(result["score"], 0)
        self.assertLess(result["reward"], 0)

    def test_group_advantages_are_centered_and_zero_for_ties(self) -> None:
        advantages = standardized_advantages([-0.7, -0.2, 1.5, 2.0])
        self.assertAlmostEqual(sum(advantages), 0.0, places=6)
        self.assertEqual(standardized_advantages([1.0, 1.0, 1.0, 1.0]), [0.0] * 4)

    def test_completion_end_keeps_first_eos_only(self) -> None:
        self.assertEqual(completion_end([1, 2, 3, 9, 9], 2, 9), 4)
        self.assertEqual(completion_end([1, 2, 3], 2, 9), 3)


if __name__ == "__main__":
    unittest.main()
