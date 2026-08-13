from __future__ import annotations

import json
import unittest

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import parse_tool_payload
from scrabble_bench.solver import grid_from_position, validate_and_score_move
from scrabble_bench.v10_training import (
    compact_reasoning_completion,
    normalize_teacher_completion,
)


class V10TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.position = json.loads(DATASET_PATH.read_text(encoding="utf-8"))[0]
        cls.lexicon = Lexicon.from_path(resolve_lexicon_path())

    def test_compact_trace_closes_template_thinking_and_is_optimal(self) -> None:
        completion = compact_reasoning_completion(self.position, self.lexicon)
        self.assertFalse(completion.startswith("<think>"))
        self.assertIn("SCAN ", completion)
        self.assertIn("TRY ", completion)
        self.assertIn("VERIFY ", completion)
        self.assertIn("\n</think>\n", completion)
        payload = parse_tool_payload(completion)
        move = validate_and_score_move(
            self.lexicon,
            grid_from_position(self.position["board"]),
            self.position["rack"],
            payload["arguments"]["placements"],
        )
        self.assertEqual(move.score, self.position["optimal_score"])

    def test_teacher_trace_is_normalized_for_thinking_template(self) -> None:
        response = (
            "<think>\nSCAN edge anchors\nTRY WORD@1,2A=20\nVERIFY legal\n"
            "CHOOSE WORD=20\n</think>\n"
            '{"tool":"play_move","arguments":{"placements":[]}}'
        )
        normalized = normalize_teacher_completion(response)
        self.assertFalse(normalized.startswith("<think>"))
        self.assertEqual(normalized.count("</think>"), 1)
        self.assertEqual(parse_tool_payload(normalized)["arguments"]["placements"], [])


if __name__ == "__main__":
    unittest.main()
