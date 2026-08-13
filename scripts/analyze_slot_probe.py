#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v41_training import move_plan


def slots(position: dict, lexicon: Lexicon) -> set[int]:
    result = set()
    for move in [position["canonical_optimal_move"], *(position.get("optimal_moves") or [])]:
        plan = move_plan(position, move["placements"], lexicon)
        result.add((0 if plan["direction"] == "across" else 1) * 225 + int(plan["start_row"]) * 15 + int(plan["start_col"]))
    return result


def choose(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda item: (int(item["band_ply"]), item["id"]))
    selected = []; games = set(); bands = sorted({int(row["band_ply"]) for row in ordered})
    queues = {band: [row for row in ordered if int(row["band_ply"]) == band] for band in bands}
    while len(selected) < len({row["source_game_id"] for row in rows}):
        progressed = False
        for band in bands:
            while queues[band]:
                row = queues[band].pop(0)
                if row["source_game_id"] not in games:
                    selected.append(row); games.add(row["source_game_id"]); progressed = True; break
        if not progressed: break
    return selected


def factor_sets(target: set[int]) -> tuple[set[int], set[int], set[int]]:
    decoded = [(slot // 225, (slot % 225) // 15, slot % 15) for slot in target]
    return ({row for _, row, _ in decoded}, {col for _, _, col in decoded}, {direction for direction, _, _ in decoded})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); lexicon = Lexicon.from_path(resolve_lexicon_path())
    train = json.loads(args.train.read_text()); test = choose(json.loads(args.test.read_text()))
    slot_counts = Counter(); row_counts = Counter(); col_counts = Counter(); direction_counts = Counter()
    for position in train:
        target = slots(position, lexicon); rows, cols, directions = factor_sets(target)
        slot_counts.update(target); row_counts.update(rows); col_counts.update(cols); direction_counts.update(directions)
    slot_order = [slot for slot, _ in slot_counts.most_common()] + [slot for slot in range(450) if slot not in slot_counts]
    best_row = row_counts.most_common(1)[0][0]; best_col = col_counts.most_common(1)[0][0]; best_direction = direction_counts.most_common(1)[0][0]
    targets = [slots(position, lexicon) for position in test]
    factors = [factor_sets(target) for target in targets]
    constant = {}
    for k in (1, 8, 32): constant[f"recall_at_{k}_pct"] = 100 * statistics.mean(bool(target & set(slot_order[:k])) for target in targets)
    constant.update({
        "row_top1_pct": 100 * statistics.mean(best_row in factor[0] for factor in factors),
        "col_top1_pct": 100 * statistics.mean(best_col in factor[1] for factor in factors),
        "direction_top1_pct": 100 * statistics.mean(best_direction in factor[2] for factor in factors),
        "best_row": best_row, "best_col": best_col, "best_direction": "across" if best_direction == 0 else "down",
    })
    random_expected = {}
    for k in (1, 8, 32):
        random_expected[f"recall_at_{k}_pct"] = 100 * statistics.mean(
            1 - math.comb(450 - len(target), k) / math.comb(450, k) for target in targets
        )
    random_expected.update({
        "row_top1_pct": 100 * statistics.mean(len(factor[0]) / 15 for factor in factors),
        "col_top1_pct": 100 * statistics.mean(len(factor[1]) / 15 for factor in factors),
        "direction_top1_pct": 100 * statistics.mean(len(factor[2]) / 2 for factor in factors),
    })
    payload = {
        "boards": len(test), "unique_games": len({row["source_game_id"] for row in test}),
        "optimal_slot_count_mean": statistics.mean(len(target) for target in targets),
        "optimal_slot_count_median": statistics.median(len(target) for target in targets),
        "random_expected": random_expected, "best_constant_from_train": constant,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
