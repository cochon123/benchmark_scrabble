from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scrabble_bench.subsets import materialize_position_subset


class PositionSubsetTests(unittest.TestCase):
    def test_materialize_preserves_requested_order_and_distinct_games(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "positions.json"
            source.write_text(
                json.dumps(
                    [
                        {"id": "a", "source_game_id": "g1"},
                        {"id": "b", "source_game_id": "g2"},
                    ]
                ),
                encoding="utf-8",
            )
            selected, resolved = materialize_position_subset(
                {"source": "positions.json", "position_ids": ["b", "a"]},
                root,
            )
            self.assertEqual(resolved, source)
            self.assertEqual([position["id"] for position in selected], ["b", "a"])


    def test_materialize_rejects_missing_position(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "positions.json").write_text(
                json.dumps([{"id": "a", "source_game_id": "g1"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing"):
                materialize_position_subset(
                    {"source": "positions.json", "position_ids": ["missing"]},
                    root,
                )


if __name__ == "__main__":
    unittest.main()
