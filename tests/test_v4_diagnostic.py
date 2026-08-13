from __future__ import annotations

import json
import unittest
from pathlib import Path

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v4_diagnostic import diagnostic_record, placement_key


class V4DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lexicon = Lexicon.from_path(resolve_lexicon_path())
        positions = json.loads(Path("data/general_v3_merged/test_positions.json").read_text())
        cls.position = next(item for item in positions if int(item["band_ply"]) > 0)

    def test_all_task_families_have_one_target_and_eight_candidates(self) -> None:
        for task in ("copy", "legality", "ranking", "anchor_direction"):
            with self.subTest(task=task):
                record = diagnostic_record(self.position, self.lexicon, task=task, seed=5519)
                self.assertEqual(len(record["candidates"]), 8)
                targets = [item for item in record["candidates"] if item.get("target")]
                self.assertEqual(len(targets), 1)
                self.assertEqual(placement_key(targets[0]["placements"]), placement_key(record["target_placements"]))

    def test_candidate_metadata_is_hidden_from_model_prompt(self) -> None:
        record = diagnostic_record(self.position, self.lexicon, task="ranking", seed=5520)
        prompt = record["messages"][-1]["content"]
        self.assertNotIn('"score"', prompt)
        self.assertNotIn('"legal"', prompt)
        self.assertNotIn('"kind"', prompt)
        self.assertNotIn('"target"', prompt)


if __name__ == "__main__":
    unittest.main()
