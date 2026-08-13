#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
from collections import Counter
from pathlib import Path
from typing import Any

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.constants import RACK_SIZE, TILE_DISTRIBUTION
from scrabble_bench.dataset import _canonical_move
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.solver import (
    BoardTile,
    empty_grid,
    enumerate_moves,
    grid_from_position,
    grid_to_cells,
    validate_and_score_move,
)
from scrabble_bench.training import TRANSFORMS, position_key, training_record, transform_position


_LEXICON: Lexicon | None = None
_MIN_PLY = 4
_MAX_PLY = 14
_CANDIDATE_LIMIT = 0
_CANDIDATE_STRATEGY = "top"


def _init_worker(
    lexicon_path: str,
    min_ply: int = 4,
    max_ply: int = 14,
    candidate_limit: int = 0,
    candidate_strategy: str = "top",
) -> None:
    global _CANDIDATE_LIMIT, _CANDIDATE_STRATEGY, _LEXICON, _MAX_PLY, _MIN_PLY
    _LEXICON = Lexicon.from_path(Path(lexicon_path))
    _MIN_PLY = min_ply
    _MAX_PLY = max_ply
    _CANDIDATE_LIMIT = candidate_limit
    _CANDIDATE_STRATEGY = candidate_strategy


def _candidate_sample(moves: list[Any], limit: int, strategy: str) -> list[Any]:
    """Retain a deterministic, score-diverse subset without losing the optimum."""
    if limit <= 0:
        return []
    if len(moves) <= limit:
        return moves[:]
    if strategy == "top":
        return moves[:limit]
    if strategy != "stratified":
        raise ValueError(f"Unsupported candidate strategy: {strategy}")

    head_count = min(4, limit)
    indices = list(range(head_count))
    remaining = limit - head_count
    if remaining:
        low = head_count
        high = len(moves) - 1
        quantiles = (
            [high]
            if remaining == 1
            else [round(low + i * (high - low) / (remaining - 1)) for i in range(remaining)]
        )
        indices.extend(int(index) for index in quantiles)
    selected: list[Any] = []
    seen: set[int] = set()
    for index in indices:
        if index not in seen:
            selected.append(moves[index])
            seen.add(index)
    if len(selected) < limit:
        for index, move in enumerate(moves):
            if index not in seen:
                selected.append(move)
                seen.add(index)
            if len(selected) == limit:
                break
    return selected


def _generate_game(seed: int) -> list[dict[str, Any]]:
    assert _LEXICON is not None
    rng = random.Random(seed)
    board = empty_grid()
    bag = [letter for letter, count in TILE_DISTRIBUTION.items() for _ in range(count)]
    rng.shuffle(bag)
    racks = [_draw_tiles(bag, []), _draw_tiles(bag, [])]
    player_to_move = 0
    positions: list[dict[str, Any]] = []

    # Reuse each self-play move search as a supervised optimal-move label.
    # Stop at the configured last ply so curriculum shards do no unused searches.
    for tiles_played in range(_MAX_PLY + 1):
        rack = racks[player_to_move]
        moves = enumerate_moves(_LEXICON, board, rack)
        if not moves:
            break
        top_score = moves[0].score
        optimal_moves = [move for move in moves if move.score == top_score]
        if _MIN_PLY <= tiles_played <= _MAX_PLY:
            canonical = _canonical_move(optimal_moves)
            position = {
                    "id": f"train-{seed}-{tiles_played}",
                    "band_ply": tiles_played,
                    "source_game_id": f"training-selfplay-{seed}",
                    "source_seed": seed,
                    "board": grid_to_cells(board),
                    "rack": rack,
                    "player_to_move": player_to_move,
                    "bag_count": len(bag),
                    "tiles_played": tiles_played,
                    "optimal_score": canonical.score,
                    "optimal_moves": [move.to_dict() for move in optimal_moves],
                    "canonical_optimal_move": canonical.to_dict(),
                }
            if _CANDIDATE_LIMIT:
                position["candidate_moves"] = [
                    move.to_dict()
                    for move in _candidate_sample(
                        moves, _CANDIDATE_LIMIT, _CANDIDATE_STRATEGY
                    )
                ]
            positions.append(position)
        if tiles_played == _MAX_PLY:
            break
        move = rng.choice(moves[: min(12, len(moves))])
        applied = validate_and_score_move(
            _LEXICON,
            board,
            rack,
            [placement.to_dict() for placement in move.placements],
        )
        for placement in applied.placements:
            board[placement.row][placement.col] = BoardTile(placement.letter, placement.is_blank)
        racks[player_to_move] = _draw_tiles(bag, _consume_rack(rack, applied.placements))
        player_to_move = 1 - player_to_move
    return positions


def _draw_tiles(bag: list[str], rack_letters: list[str]) -> str:
    rack = list(rack_letters)
    while len(rack) < RACK_SIZE and bag:
        rack.append(bag.pop())
    rack.sort()
    return "".join(rack)


def _consume_rack(rack: str, placements: list[Any]) -> list[str]:
    counts = Counter(rack)
    for placement in placements:
        token = "?" if placement.is_blank else placement.letter
        if counts[token] > 0:
            counts[token] -= 1
        elif token != "?" and counts["?"] > 0:
            counts["?"] -= 1
    letters: list[str] = []
    for letter, count in counts.items():
        letters.extend(letter for _ in range(count))
    return sorted(letters)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _validate_transforms(position: dict[str, Any], lexicon: Lexicon) -> None:
    for name in TRANSFORMS:
        item = transform_position(position, name)
        move = validate_and_score_move(
            lexicon,
            grid_from_position(item["board"]),
            item["rack"],
            item["canonical_optimal_move"]["placements"],
        )
        if move.score != int(position["optimal_score"]):
            raise RuntimeError(
                f"Symmetry validation failed for {position['id']} {name}: "
                f"{move.score} != {position['optimal_score']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate leakage-safe Scrabble SFT data from independent self-play games."
    )
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=max(1, min(4, mp.cpu_count())))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--min-ply", type=int, default=4)
    parser.add_argument("--max-ply", type=int, default=14)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=0,
        help="Persist the top N legal candidates per position for process supervision.",
    )
    parser.add_argument(
        "--candidate-strategy",
        choices=("top", "stratified"),
        default="top",
        help="Save either the top moves or a score-stratified deterministic subset.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable transpose augmentation for train/validation.",
    )
    args = parser.parse_args()
    if args.games < 3:
        raise SystemExit("--games must be at least 3")
    if not 0 <= args.min_ply <= args.max_ply <= 14:
        raise SystemExit("Ply bounds must satisfy 0 <= min-ply <= max-ply <= 14")
    if args.candidate_limit < 0:
        raise SystemExit("--candidate-limit must be non-negative")
    if args.train_fraction <= 0 or args.validation_fraction < 0:
        raise SystemExit("Split fractions must be non-negative and train must be positive")
    if args.train_fraction + args.validation_fraction >= 1:
        raise SystemExit("train + validation fractions must leave a test split")

    benchmark = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    benchmark_keys = {position_key(item) for item in benchmark}
    seeds = list(range(args.seed_start, args.seed_start + args.games))
    print(f"Generating {len(seeds)} games with {args.workers} workers...", flush=True)
    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(
            str(resolve_lexicon_path()),
            args.min_ply,
            args.max_ply,
            args.candidate_limit,
            args.candidate_strategy,
        ),
    ) as pool:
        games = list(pool.imap(_generate_game, seeds))

    lexicon = Lexicon.from_path(resolve_lexicon_path())
    clean_games: list[list[dict[str, Any]]] = []
    seen = set(benchmark_keys)
    dropped = 0
    for game in games:
        clean: list[dict[str, Any]] = []
        for position in game:
            key = position_key(position)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            # Identity-only pilots do not materialize transformed records, so
            # validating all eight symmetry transforms would add substantial
            # solver cost without testing anything used by this dataset.
            if not args.no_augment:
                _validate_transforms(position, lexicon)
            clean.append(position)
        if clean:
            clean_games.append(clean)

    train_end = min(
        max(1, int(len(clean_games) * args.train_fraction)),
        len(clean_games) - 2,
    )
    validation_count = min(
        max(1, int(len(clean_games) * args.validation_fraction)),
        len(clean_games) - train_end - 1,
    )
    validation_end = train_end + validation_count
    split_games = {
        "train": clean_games[:train_end],
        "validation": clean_games[train_end:validation_end],
        "test": clean_games[validation_end:],
    }
    transforms = ["identity"] if args.no_augment else list(TRANSFORMS)
    manifest: dict[str, Any] = {
        "seed_start": args.seed_start,
        "requested_games": args.games,
        "generated_games": len(clean_games),
        "benchmark_positions_excluded": len(benchmark_keys),
        "duplicate_positions_dropped": dropped,
        "augmentation": transforms,
        "min_ply": args.min_ply,
        "max_ply": args.max_ply,
        "candidate_limit": args.candidate_limit,
        "candidate_strategy": args.candidate_strategy,
        "splits": {},
    }

    for split, grouped_positions in split_games.items():
        raw_positions = [item for group in grouped_positions for item in group]
        split_transforms = transforms if split != "test" else ["identity"]
        records = [
            training_record(position, transform_name)
            for position in raw_positions
            for transform_name in split_transforms
        ]
        _write_jsonl(args.output_dir / f"{split}.jsonl", records)
        positions_path = args.output_dir / f"{split}_positions.json"
        positions_payload = json.dumps(raw_positions, indent=2)
        positions_path.write_text(positions_payload, encoding="utf-8")
        manifest["splits"][split] = {
            "games": len(grouped_positions),
            "raw_positions": len(raw_positions),
            "records": len(records),
            "positions_sha256": hashlib.sha256(positions_payload.encode("utf-8")).hexdigest(),
            "positions_by_ply": dict(
                sorted(Counter(int(item["band_ply"]) for item in raw_positions).items())
            ),
        }
        print(
            f"{split}: {len(grouped_positions)} games, "
            f"{len(raw_positions)} positions, {len(records)} records",
            flush=True,
        )

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote dataset to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
