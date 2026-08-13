from __future__ import annotations

import hashlib
import json
import random
import statistics
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .constants import LETTER_MULTIPLIERS, LETTER_VALUES, WORD_MULTIPLIERS
from .lexicon import Lexicon
from .solver import enumerate_moves, grid_from_position, validate_and_score_move
from .v41_training import move_plan


FORBIDDEN_INPUT_KEYS = {
    "candidate_moves",
    "canonical_optimal_move",
    "optimal_moves",
    "optimal_score",
    "score",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, *parts: object) -> str:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_game_separated(rows: list[dict[str, Any]], boards: int, *, seed: int) -> list[dict[str, Any]]:
    """Select at most one board per game while round-robining through ply bands."""
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["band_ply"]),
            stable_key(seed, row["source_game_id"], row["id"]),
        ),
    )
    bands = sorted({int(row["band_ply"]) for row in ordered})
    queues = {band: [row for row in ordered if int(row["band_ply"]) == band] for band in bands}
    selected: list[dict[str, Any]] = []
    games: set[str] = set()
    target = min(boards, len({str(row["source_game_id"]) for row in rows}))
    while len(selected) < target:
        progressed = False
        for band in bands:
            while queues[band]:
                row = queues[band].pop(0)
                game = str(row["source_game_id"])
                if game in games:
                    continue
                selected.append(row)
                games.add(game)
                progressed = True
                break
            if len(selected) >= target:
                break
        if not progressed:
            break
    return selected


def public_position(position: dict[str, Any]) -> dict[str, Any]:
    """Keep deployment-visible state and provenance, stripping every solver annotation."""
    return {
        key: position[key]
        for key in (
            "id",
            "source_game_id",
            "source_seed",
            "band_ply",
            "player_to_move",
            "tiles_played",
            "bag_count",
            "board",
            "rack",
        )
        if key in position
    }


def _premium_name(row: int, col: int) -> str:
    word = WORD_MULTIPLIERS[row][col]
    letter = LETTER_MULTIPLIERS[row][col]
    if word == 3:
        return "triple_word"
    if word == 2:
        return "double_word"
    if letter == 3:
        return "triple_letter"
    if letter == 2:
        return "double_letter"
    return "none"


def plan_placements(position: dict[str, Any], plan: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    board = {(int(item["row"]), int(item["col"])): str(item["letter"]).upper() for item in position["board"]}
    dr, dc = ((0, 1) if plan["direction"] == "across" else (1, 0))
    placements = []
    for offset, letter in enumerate(str(plan["word"])):
        row = int(plan["start_row"]) + dr * offset
        col = int(plan["start_col"]) + dc * offset
        if not (0 <= row < 15 and 0 <= col < 15):
            raise ValueError(f"Plan runs out of bounds: {plan}")
        existing = board.get((row, col))
        if existing is not None:
            if existing != letter:
                raise ValueError(f"Plan conflicts with board at ({row}, {col}): {existing} != {letter}")
            continue
        placements.append({"row": row, "col": col, "letter": letter})
    rack = Counter(str(position["rack"]))
    blanks = 0
    for placement in placements:
        letter = placement["letter"]
        if rack[letter]:
            rack[letter] -= 1
        elif rack["?"]:
            rack["?"] -= 1
            blanks += 1
        else:
            raise ValueError(f"Rack cannot supply plan letter {letter}: {plan}")
    if not placements:
        raise ValueError(f"Plan places no new tiles: {plan}")
    return placements, blanks


def candidate_record(
    position: dict[str, Any],
    plan: dict[str, Any],
    formed_words: list[str],
) -> dict[str, Any]:
    raw_placements, blank_count = plan_placements(position, plan)
    placements = [
        {
            "row": int(item["row"]),
            "col": int(item["col"]),
            "letter": str(item["letter"]),
            "premium": _premium_name(int(item["row"]), int(item["col"])),
        }
        for item in raw_placements
    ]
    formed_words = [str(word) for word in formed_words]
    main_word = str(plan["word"])
    cross_words = list(formed_words)
    if main_word in cross_words:
        cross_words.remove(main_word)
    candidate_id = hashlib.sha256(
        f"{main_word}|{plan['start_row']}|{plan['start_col']}|{plan['direction']}".encode("utf-8")
    ).hexdigest()[:16]
    premiums = Counter(item["premium"] for item in placements if item["premium"] != "none")
    return {
        "candidate_id": candidate_id,
        "word": main_word,
        "start_row": int(plan["start_row"]),
        "start_col": int(plan["start_col"]),
        "direction": str(plan["direction"]),
        "placements": placements,
        "formed_words": formed_words,
        "cross_words": cross_words,
        "word_length": len(main_word),
        "new_tile_count": len(placements),
        "blank_count": blank_count,
        "bingo": len(placements) == 7,
        "word_face_value": sum(LETTER_VALUES[letter] for letter in main_word),
        "new_tile_face_value": sum(LETTER_VALUES[item["letter"]] for item in placements),
        "premium_counts": dict(sorted(premiums.items())),
    }


def enumerate_board(position: dict[str, Any], lexicon: Lexicon) -> tuple[dict[str, Any], dict[str, Any]]:
    moves = enumerate_moves(lexicon, grid_from_position(position["board"]), str(position["rack"]))
    by_plan: dict[tuple[str, int, int, str], tuple[dict[str, Any], int]] = {}
    for move in moves:
        raw = move.to_dict()
        plan = move_plan(position, raw["placements"], lexicon)
        candidate = candidate_record(position, plan, raw.get("words", []))
        key = (
            candidate["word"],
            candidate["start_row"],
            candidate["start_col"],
            candidate["direction"],
        )
        current = by_plan.get(key)
        if current is None or int(raw["score"]) > current[1]:
            by_plan[key] = (candidate, int(raw["score"]))
    if not by_plan:
        raise RuntimeError(f"No legal moves for {position['id']}")
    ordered = sorted(by_plan.values(), key=lambda item: item[0]["candidate_id"])
    candidates = [item[0] for item in ordered]
    scores = [item[1] for item in ordered]
    optimum = max(scores)
    declared = int(position["optimal_score"])
    if optimum != declared:
        raise RuntimeError(f"Solver disagreement for {position['id']}: enumerated={optimum}, declared={declared}")
    board_id = str(position["id"])
    inputs = {
        "board_id": board_id,
        "position": public_position(position),
        "candidates": candidates,
    }
    labels = {
        "board_id": board_id,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "scores": scores,
        "optimal_score": optimum,
        "optimal_candidate_ids": [
            candidate["candidate_id"] for candidate, score in ordered if score == optimum
        ],
    }
    assert_score_blind(inputs)
    return inputs, labels


def convert_frozen_board(row: dict[str, Any], lexicon: Lexicon) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert the immutable V8 full-space artifact without re-enumerating or changing order."""
    position = row["position"]
    public_candidates = []
    scores = []
    for raw in row["candidates"]:
        plan = {
            "word": raw["word"],
            "start_row": raw["start_row"],
            "start_col": raw["start_col"],
            "direction": raw["direction"],
        }
        raw_placements, _ = plan_placements(position, plan)
        verified = validate_and_score_move(
            lexicon,
            grid_from_position(position["board"]),
            str(position["rack"]),
            raw_placements,
        )
        public_candidates.append(candidate_record(position, plan, verified.words))
        scores.append(int(raw["score"]))
    optimum = max(scores)
    inputs = {"board_id": str(row["id"]), "position": public_position(position), "candidates": public_candidates}
    labels = {
        "board_id": str(row["id"]),
        "candidate_ids": [item["candidate_id"] for item in public_candidates],
        "scores": scores,
        "optimal_score": optimum,
        "optimal_candidate_ids": [
            item["candidate_id"] for item, score in zip(public_candidates, scores, strict=True) if score == optimum
        ],
    }
    if len(set(labels["candidate_ids"])) != len(labels["candidate_ids"]):
        raise RuntimeError(f"Duplicate frozen candidate plans for {position['id']}")
    assert_score_blind(inputs)
    return inputs, labels


def assert_score_blind(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_INPUT_KEYS.intersection(value)
        if forbidden:
            raise ValueError(f"Score-blind input contains forbidden keys at {path}: {sorted(forbidden)}")
        for key, child in value.items():
            assert_score_blind(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_score_blind(child, f"{path}[{index}]")


def _fair_tie_metrics(
    inputs: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    value,
) -> dict[str, float]:
    top1: list[float] = []
    selected_scores: list[float] = []
    optimal_scores: list[int] = []
    for board, label in zip(inputs, labels, strict=True):
        values = [value(candidate) for candidate in board["candidates"]]
        best = max(values)
        tied = [index for index, item in enumerate(values) if item == best]
        scores = [int(label["scores"][index]) for index in tied]
        optimal = int(label["optimal_score"])
        top1.append(sum(score == optimal for score in scores) / len(scores))
        selected_scores.append(statistics.mean(scores))
        optimal_scores.append(optimal)
    return {
        "expected_exact_optimal_top1_pct": 100 * statistics.mean(top1),
        "expected_point_ratio_pct": 100 * sum(selected_scores) / sum(optimal_scores),
    }


def baseline_metrics(inputs: list[dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, Any]:
    if len(inputs) != len(labels):
        raise ValueError("Input/label board counts differ")
    random_metrics = _fair_tie_metrics(inputs, labels, lambda _: 0)
    longest = _fair_tie_metrics(inputs, labels, lambda candidate: int(candidate["word_length"]))
    face_value = _fair_tie_metrics(inputs, labels, lambda candidate: int(candidate["word_face_value"]))
    counts = [len(board["candidates"]) for board in inputs]
    return {
        "boards": len(inputs),
        "candidate_count": {
            "min": min(counts),
            "median": statistics.median(counts),
            "mean": statistics.mean(counts),
            "max": max(counts),
            "total": sum(counts),
        },
        "random": random_metrics,
        "longest_word": longest,
        "face_value": face_value,
    }


FEATURE_NAMES = (
    "word_length",
    "word_face_value",
    "new_tile_count",
    "new_tile_face_value",
    "blank_count",
    "bingo",
    "cross_count",
    "cross_word_length",
    "cross_face_value",
    "double_letter_count",
    "triple_letter_count",
    "double_word_count",
    "triple_word_count",
    "letter_premium_proxy",
    "main_word_proxy",
)


def candidate_features(candidate: dict[str, Any]) -> list[float]:
    premiums = candidate.get("premium_counts", {})
    cross_words = candidate.get("cross_words", [])
    cross_face = sum(LETTER_VALUES[letter] for word in cross_words for letter in word)
    cross_length = sum(len(word) for word in cross_words)
    premium_multiplier = {"none": 1, "double_letter": 2, "triple_letter": 3, "double_word": 1, "triple_word": 1}
    letter_proxy = sum(
        LETTER_VALUES[item["letter"]] * premium_multiplier[item["premium"]]
        for item in candidate["placements"]
    )
    word_multiplier = (2 ** int(premiums.get("double_word", 0))) * (3 ** int(premiums.get("triple_word", 0)))
    main_proxy = (int(candidate["word_face_value"]) + letter_proxy - int(candidate["new_tile_face_value"])) * word_multiplier
    values = {
        "word_length": int(candidate["word_length"]),
        "word_face_value": int(candidate["word_face_value"]),
        "new_tile_count": int(candidate["new_tile_count"]),
        "new_tile_face_value": int(candidate["new_tile_face_value"]),
        "blank_count": int(candidate["blank_count"]),
        "bingo": int(bool(candidate["bingo"])),
        "cross_count": len(cross_words),
        "cross_word_length": cross_length,
        "cross_face_value": cross_face,
        "double_letter_count": int(premiums.get("double_letter", 0)),
        "triple_letter_count": int(premiums.get("triple_letter", 0)),
        "double_word_count": int(premiums.get("double_word", 0)),
        "triple_word_count": int(premiums.get("triple_word", 0)),
        "letter_premium_proxy": letter_proxy,
        "main_word_proxy": main_proxy,
    }
    return [float(values[name]) for name in FEATURE_NAMES]


def train_pairwise_ranker(
    inputs: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    seed: int = 12012,
    epochs: int = 30,
    negatives_per_board: int = 64,
    learning_rate: float = 0.04,
    l2: float = 0.0005,
) -> dict[str, Any]:
    all_features = [candidate_features(candidate) for board in inputs for candidate in board["candidates"]]
    means = [statistics.mean(row[index] for row in all_features) for index in range(len(FEATURE_NAMES))]
    stds = [statistics.pstdev(row[index] for row in all_features) or 1.0 for index in range(len(FEATURE_NAMES))]

    def normalized(candidate: dict[str, Any]) -> list[float]:
        raw = candidate_features(candidate)
        return [(value - mean) / std for value, mean, std in zip(raw, means, stds, strict=True)]

    pairs: list[list[float]] = []
    for board, label in zip(inputs, labels, strict=True):
        optimal = int(label["optimal_score"])
        positives = [index for index, score in enumerate(label["scores"]) if int(score) == optimal]
        negatives = [index for index, score in enumerate(label["scores"]) if int(score) < optimal]
        negatives.sort(key=lambda index: stable_key(seed, board["board_id"], label["candidate_ids"][index]))
        negatives = negatives[:negatives_per_board]
        positive = positives[int(stable_key(seed, board["board_id"]), 16) % len(positives)]
        positive_features = normalized(board["candidates"][positive])
        for negative in negatives:
            negative_features = normalized(board["candidates"][negative])
            pairs.append([a - b for a, b in zip(positive_features, negative_features, strict=True)])

    weights = [0.0] * len(FEATURE_NAMES)
    rng = random.Random(seed)
    for epoch in range(epochs):
        rng.shuffle(pairs)
        rate = learning_rate / math.sqrt(epoch + 1)
        for difference in pairs:
            margin = sum(weight * value for weight, value in zip(weights, difference, strict=True))
            probability = 1 / (1 + math.exp(min(40.0, max(-40.0, margin))))
            for index, value in enumerate(difference):
                weights[index] += rate * (probability * value - l2 * weights[index])
    return {
        "version": "v12-linear-pairwise-1",
        "feature_names": list(FEATURE_NAMES),
        "means": means,
        "stds": stds,
        "weights": weights,
        "training": {
            "boards": len(inputs),
            "pairs": len(pairs),
            "seed": seed,
            "epochs": epochs,
            "negatives_per_board": negatives_per_board,
            "learning_rate": learning_rate,
            "l2": l2,
        },
    }


def ranker_score(candidate: dict[str, Any], model: dict[str, Any]) -> float:
    raw = candidate_features(candidate)
    normalized = [
        (value - mean) / std
        for value, mean, std in zip(raw, model["means"], model["stds"], strict=True)
    ]
    return sum(weight * value for weight, value in zip(model["weights"], normalized, strict=True))


def _bootstrap_delta(primary: list[float], baseline: list[float], *, seed: int, samples: int = 4000) -> dict[str, float]:
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(len(primary)) for _ in primary]
        deltas.append(statistics.mean(primary[index] - baseline[index] for index in indices))
    deltas.sort()
    return {
        "delta": statistics.mean(primary) - statistics.mean(baseline),
        "lower_95": deltas[int(samples * 0.025)],
        "upper_95": deltas[int(samples * 0.975) - 1],
    }


def evaluate_ranker(
    inputs: list[dict[str, Any]], labels: list[dict[str, Any]], model: dict[str, Any], *, seed: int = 12012
) -> dict[str, Any]:
    details = []
    face_expected = []
    for board, label in zip(inputs, labels, strict=True):
        ranked = sorted(
            range(len(board["candidates"])),
            key=lambda index: (
                -ranker_score(board["candidates"][index], model),
                board["candidates"][index]["candidate_id"],
            ),
        )
        optimal = int(label["optimal_score"])
        selected_score = int(label["scores"][ranked[0]])
        first_optimal = next(rank for rank, index in enumerate(ranked, 1) if int(label["scores"][index]) == optimal)
        face_values = [int(candidate["word_face_value"]) for candidate in board["candidates"]]
        best_face = max(face_values)
        face_ties = [index for index, value in enumerate(face_values) if value == best_face]
        face_expected.append(sum(int(label["scores"][index]) == optimal for index in face_ties) / len(face_ties))
        details.append({
            "board_id": board["board_id"],
            "candidate_count": len(ranked),
            "selected_score": selected_score,
            "optimal_score": optimal,
            "optimal_top1": selected_score == optimal,
            "point_ratio": selected_score / optimal,
            "optimal_rank": first_optimal,
            "recall_at_32": first_optimal <= 32,
        })

    def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "boards": len(rows),
            "exact_optimal_top1_pct": 100 * statistics.mean(row["optimal_top1"] for row in rows),
            "point_ratio_pct": 100 * sum(row["selected_score"] for row in rows) / sum(row["optimal_score"] for row in rows),
            "optimal_recall_at_32_pct": 100 * statistics.mean(row["recall_at_32"] for row in rows),
        }

    bins = {
        "le_128": [row for row in details if row["candidate_count"] <= 128],
        "129_256": [row for row in details if 129 <= row["candidate_count"] <= 256],
        "257_512": [row for row in details if 257 <= row["candidate_count"] <= 512],
        "gt_512": [row for row in details if row["candidate_count"] > 512],
    }
    primary = metrics(details)
    comparison = _bootstrap_delta(
        [float(row["optimal_top1"]) for row in details], face_expected, seed=seed
    )
    gates = {
        "exact_optimal_top1_at_least_25pct": primary["exact_optimal_top1_pct"] >= 25,
        "point_ratio_at_least_70pct": primary["point_ratio_pct"] >= 70,
        "optimal_recall_at_32_at_least_80pct": primary["optimal_recall_at_32_pct"] >= 80,
        "paired_95pct_lower_bound_beats_face_value": comparison["lower_95"] > 0,
    }
    return {
        "metrics": primary,
        "by_candidate_count": {name: metrics(rows) for name, rows in bins.items() if rows},
        "top1_vs_face_value": comparison,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "details": details,
    }


def game_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row["source_game_id"]) for row in rows}
