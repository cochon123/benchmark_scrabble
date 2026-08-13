#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.slot_model import SlotDataset, SlotModel, decode_slot, optimal_slots, slot_metrics
from scrabble_bench.solver import enumerate_moves, grid_from_position
from scrabble_bench.v41_training import move_plan


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(rows: list[dict], boards: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (int(row["band_ply"]), row["id"]))
    selected = []; games = set(); bands = sorted({int(row["band_ply"]) for row in ordered})
    queues = {band: [row for row in ordered if int(row["band_ply"]) == band] for band in bands}
    while len(selected) < boards:
        progressed = False
        for band in bands:
            while queues[band]:
                row = queues[band].pop(0)
                if row["source_game_id"] not in games:
                    selected.append(row); games.add(row["source_game_id"]); progressed = True; break
            if len(selected) >= boards: break
        if not progressed: break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--boards", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    rows = choose(json.loads(args.positions.read_text()), args.boards)
    models = []
    for checkpoint in args.checkpoint:
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model = SlotModel(**saved["config"]).to(device); model.load_state_dict(saved["state_dict"]); model.eval(); models.append(model)
    details = []; started = time.time()
    for index, row in enumerate(rows, 1):
        board, premium, rack, _ = SlotDataset([row], lexicon)[0]
        logits = []
        with torch.inference_mode():
            for model in models:
                logits.append(model(board[None].to(device), premium[None].to(device), rack[None].to(device)).cpu())
        ensemble = torch.stack(logits).mean(dim=0)[0]
        order = ensemble.argsort(descending=True).tolist()
        targets = set(optimal_slots(row, lexicon))
        legal = enumerate_moves(lexicon, grid_from_position(row["board"]), row["rack"])
        legal_slots = set()
        for move in legal:
            plan = move_plan(row, move.to_dict()["placements"], lexicon)
            legal_slots.add((0 if plan["direction"] == "across" else 1) * 225 + int(plan["start_row"]) * 15 + int(plan["start_col"]))
        rank = next((place + 1 for place, slot in enumerate(order) if slot in targets), 451)
        details.append({
            "id": row["id"], "source_game_id": row["source_game_id"], "band_ply": int(row["band_ply"]), "legal_slot_count": len(legal_slots),
            "optimal_slot_count": len(targets), "optimal_rank": rank,
            "top1_optimal": order[0] in targets, "top1_legal": order[0] in legal_slots,
            "recall_at_8": any(slot in targets for slot in order[:8]),
            "recall_at_32": any(slot in targets for slot in order[:32]),
            "top1_slot": decode_slot(order[0]),
        })
        if index == 1 or index % 50 == 0 or index == len(rows): print(f"evaluated {index}/{len(rows)}", flush=True)
    summary = {
        "boards": len(details), "unique_games": len({row["source_game_id"] for row in details}),
        "top1_optimal_pct": 100 * statistics.mean(row["top1_optimal"] for row in details),
        "top1_legal_pct": 100 * statistics.mean(row["top1_legal"] for row in details),
        "optimal_recall_at_8_pct": 100 * statistics.mean(row["recall_at_8"] for row in details),
        "optimal_recall_at_32_pct": 100 * statistics.mean(row["recall_at_32"] for row in details),
        "candidate_slot_count_median": statistics.median(row["legal_slot_count"] for row in details),
        "band_counts": dict(sorted(Counter(str(row["band_ply"]) for row in details).items())),
        "device": str(device), "elapsed_seconds": time.time() - started,
        "checkpoints": [{"path": str(path), "sha256": sha256(path)} for path in args.checkpoint],
    }
    payload = {"summary": summary, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
