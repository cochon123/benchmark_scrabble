from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .config import DATASET_PATH, ensure_directories, resolve_lexicon_path
from .constants import PRESET_BOARD_COUNTS, RACK_SIZE, TILE_DISTRIBUTION
from .lexicon import Lexicon
from .solver import Grid, best_moves, empty_grid, enumerate_moves, grid_to_cells, validate_and_score_move


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        generate_dataset(path=path)
    return json.loads(path.read_text(encoding="utf-8"))


def generate_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    ensure_directories()
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    rng = random.Random(20260412)
    target_bands = {6: 34, 10: 33, 14: 33}
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    game_index = 0

    while any(sum(1 for item in accepted if item["band_ply"] == band) < count for band, count in target_bands.items()):
        game_index += 1
        for snapshot in _self_play_snapshots(lexicon, rng, game_index):
            band = snapshot["band_ply"]
            if sum(1 for item in accepted if item["band_ply"] == band) >= target_bands[band]:
                continue
            key = _position_dedupe_key(snapshot)
            if key in seen:
                continue
            seen.add(key)
            accepted.append(snapshot)
            if all(sum(1 for item in accepted if item["band_ply"] == band) >= count for band, count in target_bands.items()):
                break

    accepted.sort(key=lambda item: (item["band_ply"], item["id"]))
    path.write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    return accepted


def dataset_subset(mode: str, boards: int | None = None) -> list[dict[str, Any]]:
    dataset = load_dataset()
    if boards is None:
        boards = PRESET_BOARD_COUNTS.get(mode, len(dataset))
    return dataset[:boards]


def _self_play_snapshots(lexicon: Lexicon, rng: random.Random, seed: int) -> list[dict[str, Any]]:
    board = empty_grid()
    bag = _build_bag(rng)
    racks = [_draw_tiles(bag, [], rng), _draw_tiles(bag, [], rng)]
    player_to_move = 0
    snapshots: list[dict[str, Any]] = []

    for ply in range(1, 15):
        rack = racks[player_to_move]
        moves = enumerate_moves(lexicon, board, rack)
        if not moves:
            return []
        candidate_pool = moves[: min(12, len(moves))]
        move = rng.choice(candidate_pool)
        applied = validate_and_score_move(
            lexicon,
            board,
            rack,
            [placement.to_dict() for placement in move.placements],
        )
        for placement in applied.placements:
            from .solver import BoardTile

            board[placement.row][placement.col] = BoardTile(placement.letter, placement.is_blank)
        racks[player_to_move] = _draw_tiles(bag, _consume_rack(rack, applied.placements), rng)
        player_to_move = 1 - player_to_move

        if ply in {6, 10, 14}:
            next_rack = racks[player_to_move]
            optimal_moves = best_moves(lexicon, board, next_rack)
            if not optimal_moves:
                continue
            canonical = _canonical_move(optimal_moves)
            snapshots.append(
                {
                    "id": f"pos-{seed}-{ply}",
                    "band_ply": ply,
                    "source_game_id": f"selfplay-{seed}",
                    "source_seed": seed,
                    "board": grid_to_cells(board),
                    "rack": next_rack,
                    "player_to_move": player_to_move,
                    "bag_count": len(bag),
                    "tiles_played": ply,
                    "optimal_score": canonical.score,
                    "optimal_moves": [move.to_dict() for move in optimal_moves],
                    "canonical_optimal_move": canonical.to_dict(),
                }
            )
    return snapshots


def _build_bag(rng: random.Random) -> list[str]:
    bag = [letter for letter, count in TILE_DISTRIBUTION.items() for _ in range(count)]
    rng.shuffle(bag)
    return bag


def _draw_tiles(bag: list[str], rack_letters: list[str], rng: random.Random) -> str:
    rack = list(rack_letters)
    while len(rack) < RACK_SIZE and bag:
        rack.append(bag.pop())
    rack.sort()
    return "".join(rack)


def _consume_rack(rack: str, placements: list) -> list[str]:
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
    letters.sort()
    return letters


def _canonical_move(moves):
    return sorted(
        moves,
        key=lambda move: (
            -move.score,
            len(move.placements),
            [(p.row, p.col, p.letter, p.is_blank) for p in move.placements],
        ),
    )[0]


def _position_dedupe_key(position: dict[str, Any]) -> str:
    cells = sorted((cell["row"], cell["col"], cell["letter"], cell.get("is_blank", False)) for cell in position["board"])
    return json.dumps({"cells": cells, "rack": position["rack"]}, sort_keys=True)

