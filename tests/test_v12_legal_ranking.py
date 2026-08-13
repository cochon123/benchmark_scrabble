from __future__ import annotations

import pytest

from scrabble_bench.legal_ranking import (
    assert_score_blind,
    baseline_metrics,
    evaluate_ranker,
    select_game_separated,
    train_pairwise_ranker,
)


def test_score_blind_audit_rejects_nested_solver_fields():
    assert_score_blind({"position": {"rack": "AEINRST"}, "candidates": [{"word": "STAINER"}]})
    with pytest.raises(ValueError, match="score"):
        assert_score_blind({"candidates": [{"word": "STAINER", "score": 70}]})
    with pytest.raises(ValueError, match="optimal_score"):
        assert_score_blind({"position": {"optimal_score": 70}})


def test_game_selection_is_balanced_and_lineage_unique():
    rows = [
        {"id": f"{game}-{band}", "source_game_id": game, "band_ply": band}
        for game in ("a", "b", "c", "d")
        for band in (6, 7)
    ]
    selected = select_game_separated(rows, 4, seed=12)
    assert len(selected) == 4
    assert len({row["source_game_id"] for row in selected}) == 4
    assert {row["band_ply"] for row in selected} == {6, 7}


def test_fair_tie_baselines_do_not_use_candidate_order():
    inputs = [{
        "board_id": "b",
        "position": {"id": "b", "source_game_id": "g", "board": [], "rack": "AEINRST"},
        "candidates": [
            {"candidate_id": "a", "word_length": 3, "word_face_value": 3},
            {"candidate_id": "b", "word_length": 3, "word_face_value": 3},
        ],
    }]
    labels = [{
        "board_id": "b",
        "candidate_ids": ["a", "b"],
        "scores": [10, 20],
        "optimal_score": 20,
        "optimal_candidate_ids": ["b"],
    }]
    result = baseline_metrics(inputs, labels)
    assert result["longest_word"]["expected_exact_optimal_top1_pct"] == 50
    assert result["face_value"]["expected_point_ratio_pct"] == 75


def test_pairwise_ranker_trains_and_evaluates_without_score_features():
    candidates = [
        {
            "candidate_id": "low",
            "word": "AA",
            "word_length": 2,
            "word_face_value": 2,
            "new_tile_count": 2,
            "new_tile_face_value": 2,
            "blank_count": 0,
            "bingo": False,
            "cross_words": [],
            "placements": [
                {"letter": "A", "premium": "none"},
                {"letter": "A", "premium": "none"},
            ],
            "premium_counts": {},
        },
        {
            "candidate_id": "high",
            "word": "QUIZ",
            "word_length": 4,
            "word_face_value": 22,
            "new_tile_count": 4,
            "new_tile_face_value": 22,
            "blank_count": 0,
            "bingo": False,
            "cross_words": [],
            "placements": [
                {"letter": letter, "premium": "none"} for letter in "QUIZ"
            ],
            "premium_counts": {},
        },
    ]
    inputs = [{"board_id": "b", "position": {"rack": "AEINRST"}, "candidates": candidates}]
    labels = [{
        "board_id": "b",
        "candidate_ids": ["low", "high"],
        "scores": [2, 22],
        "optimal_score": 22,
        "optimal_candidate_ids": ["high"],
    }]
    model = train_pairwise_ranker(inputs, labels, epochs=3, negatives_per_board=1)
    result = evaluate_ranker(inputs, labels, model)
    assert result["metrics"]["exact_optimal_top1_pct"] == 100
