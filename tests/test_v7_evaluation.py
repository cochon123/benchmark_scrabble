from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.evaluate_v7_factorized import assess_response
from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v7_training import axis_token


class V7EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lexicon = Lexicon.from_path(resolve_lexicon_path())
        gate = json.loads(Path("data/general_v7/factorized_gate.json").read_text())
        cls.row = next(item for item in gate if item["task"] == "free-plan")

    def test_rejected_plan_retains_first_component_error(self) -> None:
        target = self.row["target_plan"]
        wrong_row = (int(target["row"]) + 1) % 15
        direction = "A" if target["direction"] == "across" else "D"
        response = "|".join(
            [
                axis_token("ROW", wrong_row),
                axis_token("COL", int(target["col"])),
                f"DIR_{direction}",
                f"WORD_{target['word']}",
            ]
        )
        result = assess_response(self.row, response, self.lexicon)
        self.assertTrue(result["parseable"])
        self.assertFalse(result["legal"])
        self.assertEqual(result["first_error"], "row")


if __name__ == "__main__":
    unittest.main()
