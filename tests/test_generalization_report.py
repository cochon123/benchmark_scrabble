from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_generalization_report import score_interval, wilson  # noqa: E402


def test_wilson_interval_contains_observed_proportion() -> None:
    low, high = wilson(44, 55)
    assert low < 80 < high
    assert 0 <= low <= high <= 100


def test_score_interval_is_deterministic_and_clusters_games() -> None:
    results = [
        {"id": "train-400000-4", "score": 10, "optimal_score": 20},
        {"id": "train-400000-5", "score": 15, "optimal_score": 20},
        {"id": "train-400001-4", "score": 20, "optimal_score": 20},
        {"id": "train-400001-5", "score": 20, "optimal_score": 20},
    ]
    first = score_interval(results)
    second = score_interval(results)
    assert first == second
    assert first[0] == 62.5
    assert first[1] == 100.0
