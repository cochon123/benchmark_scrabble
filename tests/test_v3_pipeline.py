from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.evaluation import claimed_score, error_category, summarize_evaluation
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.loss_weighting import completion_loss_weights
from scrabble_bench.runner import dense_board_text, parse_tool_payload, prompt_for_position
from scrabble_bench.solver import grid_from_position, validate_and_score_move
from scrabble_bench.training import transform_position
from scrabble_bench.v3_training import (
    analyze_legal_move,
    process_completion,
    v3_audit_record,
    v3_training_record,
)
from scripts.build_general_v3_data import (
    normalized_hard_negative_row,
    validate_hard_negative_rows,
)
from scripts.evaluate_hf import load_tokenizer


class V3PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.position = json.loads(DATASET_PATH.read_text(encoding="utf-8"))[0]
        cls.lexicon = Lexicon.from_path(resolve_lexicon_path())

    def test_dense_board_has_stable_coordinates_and_all_cells(self) -> None:
        board = dense_board_text(self.position)
        lines = board.splitlines()
        self.assertEqual(len(lines), 16)
        self.assertTrue(lines[0].startswith("   00 01 02"))
        self.assertTrue(lines[-1].startswith("14 "))
        self.assertEqual(sum(line.count(".") for line in lines[1:]), 225 - len(self.position["board"]))

    def test_adapter_tokenizer_inherits_missing_base_chat_template(self) -> None:
        adapter_tokenizer = SimpleNamespace(chat_template=None)
        base_tokenizer = SimpleNamespace(chat_template="base-template")
        with patch(
            "scripts.evaluate_hf.AutoTokenizer.from_pretrained",
            side_effect=[adapter_tokenizer, base_tokenizer],
        ) as from_pretrained:
            loaded = load_tokenizer("base-model", "adapter-path")
        self.assertIs(loaded, adapter_tokenizer)
        self.assertEqual(loaded.chat_template, "base-template")
        self.assertEqual(from_pretrained.call_count, 2)

    def test_dense_prompt_is_explicit_without_changing_sparse_default(self) -> None:
        sparse_payload = json.loads(prompt_for_position(self.position)[1]["content"])
        dense_payload = json.loads(
            prompt_for_position(self.position, board_encoding="dense")[1]["content"]
        )
        self.assertIn("board", sparse_payload)
        self.assertNotIn("board_encoding", sparse_payload)
        self.assertNotIn("board", dense_payload)
        self.assertEqual(dense_payload["board_encoding"], "dense-grid")
        self.assertEqual(len(dense_payload["board_grid"]), 16)
        self.assertTrue(dense_payload["board_grid"][1].startswith("00 "))

    def test_solver_process_trace_reproduces_exact_score_and_payload(self) -> None:
        audit = analyze_legal_move(
            self.position,
            self.position["canonical_optimal_move"],
            self.lexicon,
        )
        self.assertEqual(audit["total_score"], self.position["optimal_score"])
        self.assertEqual(
            sum(word["score"] for word in audit["word_breakdown"]) + audit["bingo_bonus"],
            self.position["optimal_score"],
        )
        completion = process_completion(self.position, self.lexicon)
        payload = parse_tool_payload(completion)
        move = validate_and_score_move(
            self.lexicon,
            grid_from_position(self.position["board"]),
            self.position["rack"],
            payload["arguments"]["placements"],
        )
        self.assertEqual(move.score, self.position["optimal_score"])
        self.assertIn("Candidate verification:", completion)
        self.assertIn("rack_ok=yes", completion)

    def test_candidate_moves_survive_transpose(self) -> None:
        position = json.loads(json.dumps(self.position))
        position["candidate_moves"] = [
            json.loads(json.dumps(position["canonical_optimal_move"]))
        ]
        transformed = transform_position(position, "transpose")
        original = position["candidate_moves"][0]["placements"][0]
        changed = transformed["candidate_moves"][0]["placements"][0]
        self.assertEqual((changed["row"], changed["col"]), (original["col"], original["row"]))
        record = v3_training_record(position, self.lexicon, "transpose")
        self.assertEqual(record["record_type"], "v3-process-move")

    def test_audit_microtask_closes_thinking_before_json(self) -> None:
        record = v3_audit_record(self.position, self.lexicon)
        content = record["messages"][-1]["content"]
        self.assertIn("</think>", content)
        answer = json.loads(content.rsplit("</think>", 1)[1].strip())
        self.assertTrue(answer["legal"])
        self.assertEqual(answer["score"], self.position["optimal_score"])

    def test_final_placement_tokens_receive_largest_weight(self) -> None:
        content = 'analysis\n</think>\n{"tool":"play_move","arguments":{"placements":[{"row":11,"col":3,"letter":"S"}]}}'
        offsets = [(index, index + 1) for index in range(len(content))]
        weights = completion_loss_weights(
            content,
            offsets,
            json_weight=2.0,
            placement_weight=4.0,
        )
        self.assertEqual(weights[0], 1.0)
        self.assertEqual(weights[content.index('{"tool"')], 2.0)
        self.assertEqual(weights[content.index('"row"')], 4.0)

    def test_hard_negative_import_is_restricted_to_train_sources(self) -> None:
        valid = {
            "id": "recovery-1",
            "source_id": "train-position",
            "messages": [{"role": "assistant", "content": "corrected move"}],
        }
        validate_hard_negative_rows([valid], {"train-position"})
        invalid = dict(valid, source_id="validation-position")
        with self.assertRaisesRegex(RuntimeError, "non-train"):
            validate_hard_negative_rows([invalid], {"train-position"})

    def test_hard_negative_import_requires_an_assistant_target(self) -> None:
        malformed = {
            "id": "recovery-1",
            "source_id": "train-position",
            "messages": [{"role": "user", "content": "feedback"}],
        }
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            validate_hard_negative_rows([malformed], {"train-position"})

    def test_hard_negative_import_normalizes_to_clean_training_schema(self) -> None:
        row = {
            "id": "recovery-1",
            "source_id": "train-position",
            "position_key": "position-key",
            "record_type": "v3-on-policy-recovery",
            "rejection_error": "audit-only detail",
            "error_category": "collision",
            "messages": [{"role": "assistant", "content": "corrected move"}],
        }
        normalized = normalized_hard_negative_row(
            row,
            {"train-position": {"optimal_score": 42, "position_key": "position-key"}},
        )
        self.assertEqual(
            set(normalized),
            {
                "id",
                "source_id",
                "transform",
                "position_key",
                "optimal_score",
                "record_type",
                "messages",
            },
        )
        self.assertEqual(normalized["optimal_score"], 42)

    def test_legality_report_tracks_retry_recovery_and_score_claims(self) -> None:
        payload = {
            "results": [
                {
                    "score": 40,
                    "optimal_score": 40,
                    "is_optimal": True,
                    "error": None,
                    "raw_response": 'verified maximum 40\n{"tool":"play_move","arguments":{"placements":[]}}',
                    "attempts": [
                        {"raw_response": "bad", "error": "Response did not contain JSON."},
                        {
                            "raw_response": '{"tool":"play_move","arguments":{"placements":[]}}',
                            "error": None,
                        },
                    ],
                },
                {
                    "score": 0,
                    "optimal_score": 20,
                    "is_optimal": False,
                    "error": "Cross word is invalid: ZZ",
                    "raw_response": '{"tool":"play_move","arguments":{"placements":[]}}',
                    "attempts": [
                        {
                            "raw_response": '{"tool":"play_move","arguments":{"placements":[]}}',
                            "error": "Cross word is invalid: ZZ",
                        }
                    ],
                },
            ]
        }
        report = summarize_evaluation(payload)
        self.assertEqual(report["final"]["legal_pct"], 50.0)
        self.assertEqual(report["final"]["recovered_after_rejection"], 1)
        self.assertEqual(report["final"]["exact_score_claims"], 1)
        self.assertEqual(error_category("Cross word is invalid: ZZ"), "cross_word")
        self.assertEqual(claimed_score("verified maximum 40"), 40)


if __name__ == "__main__":
    unittest.main()
