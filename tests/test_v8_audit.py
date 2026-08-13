from __future__ import annotations

import json
import unittest
from pathlib import Path

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v8_audit import (
    accepted_plans,
    best_component_repair,
    best_constant_baselines,
    independent_component_matches,
)


class V8AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lexicon = Lexicon.from_path(resolve_lexicon_path())
        cls.rows = json.loads(Path("data/general_v7/rollout_gate.json").read_text())
        cls.row = cls.rows[0]

    def test_optimal_targets_are_set_valued(self) -> None:
        plans = accepted_plans(self.row["position"], self.lexicon)
        self.assertGreaterEqual(len(plans), 1)
        predicted = {name: plans[0][name] for name in ("row", "col", "direction", "word")}
        self.assertTrue(all(independent_component_matches(predicted, plans).values()))

    def test_full_repair_is_legal_and_optimal(self) -> None:
        plans = accepted_plans(self.row["position"], self.lexicon)
        predicted = {"row": 0, "col": 0, "direction": "down", "word": "ZZZ"}
        repaired = best_component_repair(
            self.row["position"],
            predicted,
            plans,
            ("row", "col", "direction", "word"),
            self.lexicon,
        )
        self.assertTrue(repaired.legal)
        self.assertEqual(repaired.score, self.row["position"]["optimal_score"])

    def test_constant_baseline_uses_most_common_target(self) -> None:
        baselines = best_constant_baselines(self.rows)
        self.assertEqual(baselines["row"]["total"], len(self.rows))
        self.assertGreaterEqual(baselines["location"]["correct"], 1)


if __name__ == "__main__":
    unittest.main()
