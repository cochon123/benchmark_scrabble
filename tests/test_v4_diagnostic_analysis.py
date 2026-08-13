from __future__ import annotations

import unittest

from scripts.analyze_v4_diagnostic import binomial_tail, enrich, mcnemar_exact, wilson


class V4DiagnosticAnalysisTests(unittest.TestCase):
    def test_intervals_and_random_tail(self) -> None:
        low, high = wilson(25, 50)
        self.assertLess(low, 50)
        self.assertGreater(high, 50)
        self.assertLess(binomial_tail(20, 50, 0.125), 0.001)
        self.assertLess(mcnemar_exact(12, 1), 0.01)

    def test_diagnosis_prioritizes_copy_failure(self) -> None:
        tasks = {
            name: {"success": 0, "boards": 50, "accuracy_pct": 0.0, "legal_output_pct": 0.0}
            for name in ("copy", "legality", "ranking", "anchor_direction")
        }
        payload = {
            "base": {"by_task": {key: dict(value) for key, value in tasks.items()}},
            "sft": {"by_task": {key: dict(value) for key, value in tasks.items()}},
        }
        self.assertEqual(enrich(payload)["diagnosis"]["classification"], "execution_and_coordinate_copying_failure")

    def test_strong_ranking_signal_identifies_search_bottleneck(self) -> None:
        tasks = {
            name: {"success": 10, "boards": 50, "accuracy_pct": 20.0, "legal_output_pct": 20.0}
            for name in ("copy", "legality", "ranking", "anchor_direction")
        }
        tasks["ranking"].update(success=23, accuracy_pct=46.0, legal_output_pct=88.0)
        payload = {
            "base": {"by_task": {key: dict(value) for key, value in tasks.items()}},
            "sft": {"by_task": {key: dict(value) for key, value in tasks.items()}},
        }
        self.assertEqual(enrich(payload)["diagnosis"]["classification"], "free_search_failure_with_candidate_skill")


if __name__ == "__main__":
    unittest.main()
