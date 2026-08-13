#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.constants import TILE_DISTRIBUTION
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.solver import BoardTile, empty_grid, enumerate_moves, grid_from_position, grid_to_cells
from scrabble_bench.v11_training import gate_row, move_action, training_record


ROOT = Path(__file__).resolve().parents[1]
Prepared = tuple[dict[str, Any], str, int]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def tile_bag() -> list[str]:
    return [letter for letter, count in TILE_DISTRIBUTION.items() for _ in range(count)]


def synthetic_positions(
    *, category: str, count: int, seed: int, lexicon: Lexicon
) -> list[Prepared]:
    if category not in {"opening", "one_anchor"}:
        raise ValueError(category)
    rng = random.Random(seed)
    result: list[Prepared] = []
    attempts = 0
    while len(result) < count:
        attempts += 1
        if attempts > count * 30:
            raise RuntimeError(f"Could not generate {count} {category} positions")
        bag = tile_bag()
        rng.shuffle(bag)
        board: list[dict[str, Any]] = []
        if category == "one_anchor":
            anchor = bag.pop()
            if anchor == "?":
                continue
            board = [
                {
                    "row": rng.randrange(2, 13),
                    "col": rng.randrange(2, 13),
                    "letter": anchor,
                    "is_blank": False,
                    "is_existing": True,
                }
            ]
        rack = "".join(sorted(bag[-7:]))
        provisional = {
            "id": f"v11-{category}-{seed}-{len(result):04d}",
            "source_game_id": f"v11-{category}-{seed}-{len(result):04d}",
            "band_ply": 0 if category == "opening" else 1,
            "board": board,
            "rack": rack,
        }
        moves = enumerate_moves(lexicon, grid_from_position(board), rack)
        if not moves:
            continue
        best = moves[0]
        provisional.update(
            optimal_score=int(best.score),
            optimal_moves=[best.to_dict()],
            canonical_optimal_move=best.to_dict(),
        )
        result.append((provisional, easiest_action(provisional, moves, lexicon), len(moves)))
    return result


def synthetic_controlled_gate(*, count: int, seed: int, lexicon: Lexicon) -> list[dict[str, Any]]:
    """Make held-out, reachable boards containing exactly one opening word."""
    rng = random.Random(seed)
    result: list[dict[str, Any]] = []
    attempts = 0
    while len(result) < count:
        attempts += 1
        if attempts > count * 30:
            raise RuntimeError(f"Could not generate {count} controlled gate positions")
        bag = tile_bag()
        rng.shuffle(bag)
        rack1 = "".join(sorted(bag.pop() for _ in range(7)))
        opening_moves = enumerate_moves(lexicon, empty_grid(), rack1)
        if not opening_moves:
            continue
        opening = rng.choice(opening_moves[: min(20, len(opening_moves))])
        board = empty_grid()
        for placement in opening.placements:
            board[placement.row][placement.col] = BoardTile(
                placement.letter, placement.is_blank
            )
        rack2 = "".join(sorted(bag.pop() for _ in range(7)))
        legal = enumerate_moves(lexicon, board, rack2)
        if not legal:
            continue
        result.append(
            {
                "id": f"v11-controlled-gate-{seed}-{len(result):04d}",
                "source_game_id": f"v11-controlled-gate-{seed}-{len(result):04d}",
                "band_ply": 1,
                "board": grid_to_cells(board),
                "rack": rack2,
                "optimal_score": int(legal[0].score),
                "optimal_moves": [legal[0].to_dict()],
                "canonical_optimal_move": legal[0].to_dict(),
            }
        )
    return result


def easiest_action(position: dict[str, Any], moves: list[Any], lexicon: Lexicon) -> str:
    # The SFT policy is deliberately easy and deterministic: legality first, not
    # imitation of Quackle's exact optimum. The verifier still accepts every legal move.
    ordered = sorted(
        moves,
        key=lambda move: (
            len(move.placements),
            int(move.score),
            tuple((p.row, p.col, p.letter) for p in move.placements),
        ),
    )
    last_error: Exception | None = None
    for move in ordered:
        try:
            return move_action(position, move, lexicon)
        except (RuntimeError, ValueError) as error:
            last_error = error
    raise RuntimeError(f"No move could be encoded for {position['id']}: {last_error}")


def records_for(
    items: list[Prepared], category: str
) -> list[dict[str, Any]]:
    return [
        training_record(
            position,
            action,
            stage=category,
            legal_move_count=legal_count,
        )
        for position, action, legal_count in items
    ]


def materialize_existing(
    positions: list[dict[str, Any]], lexicon: Lexicon
) -> list[Prepared]:
    result: list[Prepared] = []
    for index, position in enumerate(positions, 1):
        moves = enumerate_moves(lexicon, grid_from_position(position["board"]), str(position["rack"]))
        if moves:
            result.append((position, easiest_action(position, moves, lexicon), len(moves)))
        if index % 100 == 0:
            print(f"enumerated {index}/{len(positions)} existing positions", flush=True)
    return result


def mixed(primary: list[dict[str, Any]], *replay: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(primary)
    replay_budget = max(1, len(primary) // 3)
    for source in replay:
        rows.extend(rng.sample(source, min(replay_budget, len(source))))
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V11 atomic-pointer legality curriculum.")
    parser.add_argument(
        "--positions",
        type=Path,
        default=ROOT / "data/slot_pilot_v1/train_positions.json",
    )
    parser.add_argument(
        "--benchmark-validation",
        type=Path,
        default=ROOT / "data/slot_pilot_v1/validation_positions.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/v11_atomic_policy")
    parser.add_argument("--synthetic-train", type=int, default=500)
    parser.add_argument("--synthetic-validation", type=int, default=50)
    parser.add_argument("--synthetic-gate", type=int, default=100)
    parser.add_argument("--controlled-train", type=int, default=300)
    parser.add_argument("--full-train", type=int, default=700)
    parser.add_argument("--seed", type=int, default=11001)
    args = parser.parse_args()

    lexicon = Lexicon.from_path(resolve_lexicon_path())
    source = json.loads(args.positions.read_text(encoding="utf-8"))
    benchmark_validation = json.loads(args.benchmark_validation.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    synthetic_total = args.synthetic_train + args.synthetic_validation + args.synthetic_gate
    opening_all = synthetic_positions(
        category="opening", count=synthetic_total, seed=args.seed + 1, lexicon=lexicon
    )
    anchor_all = synthetic_positions(
        category="one_anchor", count=synthetic_total, seed=args.seed + 2, lexicon=lexicon
    )
    controlled_gate_positions = synthetic_controlled_gate(
        count=40, seed=args.seed + 3, lexicon=lexicon
    )

    by_ply: dict[int, list[dict[str, Any]]] = {}
    for position in source:
        by_ply.setdefault(int(position["band_ply"]), []).append(position)
    controlled_pool = list(by_ply.get(1, [])) + list(by_ply.get(2, []))
    rng.shuffle(controlled_pool)
    controlled_count = min(args.controlled_train + 40, len(controlled_pool))
    controlled_all = materialize_existing(controlled_pool[:controlled_count], lexicon)
    full_pool = [position for position in source if int(position["band_ply"]) >= 6]
    rng.shuffle(full_pool)
    full_all = materialize_existing(full_pool[: args.full_train + 60], lexicon)
    full_gate_items = benchmark_validation

    nt, nv = args.synthetic_train, args.synthetic_validation
    opening_train = records_for(opening_all[:nt], "opening")
    opening_validation = records_for(opening_all[nt : nt + nv], "opening")
    opening_gate = opening_all[nt + nv :]
    anchor_train = records_for(anchor_all[:nt], "one_anchor")
    anchor_validation = records_for(anchor_all[nt : nt + nv], "one_anchor")
    anchor_gate = anchor_all[nt + nv :]

    controlled_train = records_for(controlled_all[: args.controlled_train], "controlled_cross")
    controlled_validation = records_for(controlled_all[args.controlled_train :], "controlled_cross")
    full_train = records_for(full_all[: args.full_train], "full")
    full_validation = records_for(full_all[args.full_train :], "full")

    stages = {
        "stage0_opening": (opening_train, opening_validation),
        "stage1_anchor": (
            mixed(anchor_train, opening_train, seed=args.seed + 10),
            anchor_validation + opening_validation,
        ),
        "stage2_cross": (
            mixed(controlled_train, anchor_train, opening_train, seed=args.seed + 11),
            controlled_validation + anchor_validation + opening_validation,
        ),
        "stage3_full": (
            mixed(full_train, controlled_train, anchor_train, opening_train, seed=args.seed + 12),
            full_validation + controlled_validation + anchor_validation + opening_validation,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stage, (train, validation) in stages.items():
        write_jsonl(args.output_dir / f"{stage}_train.jsonl", train)
        write_jsonl(args.output_dir / f"{stage}_validation.jsonl", validation)

    gate = []
    gate.extend(gate_row(position, category="opening") for position, _, _ in opening_gate)
    gate.extend(gate_row(position, category="one_anchor") for position, _, _ in anchor_gate)
    gate.extend(
        gate_row(position, category="controlled_cross")
        for position in controlled_gate_positions
    )
    gate.extend(gate_row(position, category="full") for position in full_gate_items)
    (args.output_dir / "frozen_gate.json").write_text(
        json.dumps(gate, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )

    manifest = {
        "version": "v11-atomic-pointer-legality-1",
        "seed": args.seed,
        "policy": "deterministic easiest legal SFT target; verifier accepts any legal action",
        "category_counts": {
            "opening_train": len(opening_train),
            "opening_gate": len(opening_gate),
            "one_anchor_train": len(anchor_train),
            "one_anchor_gate": len(anchor_gate),
            "controlled_train": len(controlled_train),
            "controlled_gate": len(controlled_gate_positions),
            "full_train": len(full_train),
            "full_gate": len(full_gate_items),
        },
        "stage_counts": {
            stage: {"train": len(train), "validation": len(validation)}
            for stage, (train, validation) in stages.items()
        },
        "legal_move_counts": dict(
            Counter(
                min(100, legal_count)
                for items in (opening_all, anchor_all, controlled_all, full_all)
                for _, _, legal_count in items
            )
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
