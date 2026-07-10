from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scrabble_bench.cli_agents import iter_word_chunks, parse_codex_stream_event
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.openrouter import normalize_model_for_benchmark
from scrabble_bench.runner import _retry_feedback, parse_tool_payload, prompt_for_position
from scrabble_bench.solver import BoardTile, empty_grid, validate_and_score_move


def lexicon_from_words(words: list[str]) -> Lexicon:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "words.txt"
        path.write_text("\n".join(words), encoding="utf-8")
        return Lexicon.from_path(path)


class ParserTests(unittest.TestCase):
    def test_parses_fenced_json(self) -> None:
        payload = parse_tool_payload(
            '```json\n{"tool":"play_move","arguments":{"placements":[{"row":7,"col":7,"letter":"A"}]}}\n```'
        )
        self.assertEqual(payload["tool"], "play_move")
        self.assertEqual(payload["arguments"]["placements"][0]["letter"], "A")

    def test_parses_json_after_brief_preface(self) -> None:
        payload = parse_tool_payload(
            'Best move found.\nUsing existing tiles only where needed.\n{"tool":"play_move","arguments":{"placements":[{"row":7,"col":7,"letter":"A"}]}}'
        )
        self.assertEqual(payload["tool"], "play_move")
        self.assertEqual(payload["arguments"]["placements"][0]["letter"], "A")

    def test_prompt_requires_only_new_tiles(self) -> None:
        messages = prompt_for_position({"rack": "DHJNSUW", "board": [], "id": "x"})
        self.assertIn("only the new tiles placed this turn", messages[0]["content"])
        self.assertIn("Return only newly placed tiles", messages[1]["content"])

    def test_retry_feedback_mentions_rack_and_new_tiles(self) -> None:
        feedback = _retry_feedback(
            {"rack": "DHJNSUW", "board": [], "id": "x"},
            '{"tool":"play_move","arguments":{"placements":[{"row":7,"col":7,"letter":"S"}]}}',
            "Rack cannot supply letter 'S'.",
        )
        self.assertIn('"rejected"', feedback)
        self.assertIn('"hint"', feedback)
        self.assertIn('"rack"', feedback)
        self.assertIn("Return only newly placed tiles.", feedback)

    def test_no_reasoning_keeps_deepseek_base_model(self) -> None:
        self.assertEqual(
            normalize_model_for_benchmark("deepseek/deepseek-v4-flash:thinking", "none"),
            "deepseek/deepseek-v4-flash",
        )


class SolverTests(unittest.TestCase):
    def test_opening_move_must_cover_center(self) -> None:
        lexicon = lexicon_from_words(["A", "AT"])
        with self.assertRaises(ValueError):
            validate_and_score_move(
                lexicon,
                empty_grid(),
                "A",
                [{"row": 0, "col": 0, "letter": "A"}],
            )

    def test_scores_simple_extension_and_tolerates_existing_reemission(self) -> None:
        lexicon = lexicon_from_words(["A", "AT", "HAT"])
        grid = empty_grid()
        grid[7][7] = BoardTile("A")
        grid[7][8] = BoardTile("T")
        move = validate_and_score_move(
            lexicon,
            grid,
            "H",
            [
                {"row": 7, "col": 6, "letter": "H"},
                {"row": 7, "col": 7, "letter": "A"},
            ],
        )
        self.assertEqual(move.score, 6)
        self.assertEqual(move.words, ["HAT"])

    def test_single_tile_move_can_extend_horizontal_word(self) -> None:
        lexicon = lexicon_from_words(["A", "AX", "AXE"])
        grid = empty_grid()
        grid[7][7] = BoardTile("A")
        grid[7][8] = BoardTile("X")
        move = validate_and_score_move(
            lexicon,
            grid,
            "E",
            [{"row": 7, "col": 9, "letter": "E"}],
        )
        self.assertEqual(move.score, 10)
        self.assertEqual(move.words, ["AXE"])

    def test_single_tile_move_can_extend_vertical_word(self) -> None:
        lexicon = lexicon_from_words(["AXONE", "AXONES"])
        grid = empty_grid()
        for row, letter in enumerate("AXONE", start=3):
            grid[row][7] = BoardTile(letter)
        move = validate_and_score_move(
            lexicon,
            grid,
            "S",
            [{"row": 8, "col": 7, "letter": "S"}],
        )
        self.assertEqual(move.score, 13)
        self.assertEqual(move.words, ["AXONES"])


class LexiconTests(unittest.TestCase):
    def test_enable_lexicon_keeps_full_word_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ENABLE.txt"
            words = [f"AA{i:05d}" for i in range(13050)] + ["JOWS"]
            path.write_text("\n".join(words), encoding="utf-8")
            lexicon = Lexicon.from_path(path)
        self.assertIn("JOWS", lexicon.words)


class CodexStreamTests(unittest.TestCase):
    def test_parse_activity_as_reasoning_before_content(self) -> None:
        lines = [
            '{"type":"thread.started","thread_id":"t1"}',
            '{"type":"turn.started"}',
            '{"type":"item.completed","item":{"id":"item_0","type":"command_execution","command":"ls","status":"completed"}}',
            '{"type":"item.completed","item":{"id":"item_1","type":"reasoning","text":"considering the rack"}}',
            '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"hello stream world"}}',
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":2}}',
        ]
        events = [parse_codex_stream_event(line) for line in lines]
        # First non-empty events should be reasoning crumbs, then content.
        self.assertEqual(events[0][1], "reasoning")
        self.assertIn("thread started", events[0][0])
        self.assertEqual(events[1][1], "reasoning")
        self.assertIn("turn started", events[1][0])
        self.assertEqual(events[2][1], "reasoning")
        self.assertIn("command", events[2][0])
        self.assertEqual(events[3], ("considering the rack", "reasoning", "reasoning"))
        self.assertEqual(events[4], ("hello stream world", "content", "agent_message"))
        before_content = [channel for text, channel, _ in events[:4] if text]
        self.assertTrue(before_content)
        self.assertTrue(all(ch == "reasoning" for ch in before_content))

    def test_word_chunks_fake_stream(self) -> None:
        chunks = iter_word_chunks("hello stream world")
        self.assertEqual(chunks, ["hello ", "stream ", "world"])
        self.assertEqual("".join(chunks), "hello stream world")
        self.assertGreater(len(chunks), 1)

    def test_fake_stream_emits_content_deltas(self) -> None:
        from scrabble_bench.cli_agents import _fake_stream_content

        emitted: list[dict] = []
        _fake_stream_content(
            "hello stream world",
            emitted.append,
            start=0.0,
            last_event_at=0.0,
            trace_events=[],
            source="agent_message",
            word_delay_range=(0.0, 0.0),
        )
        self.assertEqual([e["type"] for e in emitted], ["content_delta"] * 3)
        self.assertEqual("".join(e["text"] for e in emitted), "hello stream world")

    def test_last_agent_message_is_content(self) -> None:
        text, channel, event_type = parse_codex_stream_event(
            '{"type":"turn.completed","last_agent_message":"final answer"}'
        )
        self.assertEqual((text, channel, event_type), ("final answer", "content", "last_agent_message"))


if __name__ == "__main__":
    unittest.main()
