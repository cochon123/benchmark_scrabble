#!/usr/bin/env python3
"""Analyze seed ensemble, oracle union, and saved candidate-slot shortlist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.slot_model import SlotDataset, decode_slot, factor_metrics, slot_index, slot_metrics
from scrabble_bench.v41_training import move_plan
from scrabble_bench.v8_joint import encode_position
from scrabble_bench.slot_model import SpatialSlotModel


def choose(rows):
    ordered = sorted(rows, key=lambda item: (int(item["band_ply"]), item["id"]))
    selected, games = [], set()
    bands = sorted({int(row["band_ply"]) for row in ordered})
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


def candidate_slots(row, lexicon):
    slots = set()
    for move in row.get("candidate_moves") or []:
        plan = move_plan(row, move["placements"], lexicon)
        slots.add(slot_index(plan["start_row"], plan["start_col"], plan["direction"]))
    return slots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    rows = choose(json.loads(args.positions.read_text()))
    models = []
    for path in args.checkpoint:
        saved = torch.load(path, map_location=device, weights_only=True)
        model = SpatialSlotModel(**{k: v for k, v in saved["config"].items() if k != "model_type"}).to(device)
        model.load_state_dict(saved["state_dict"]); model.eval(); models.append(model)
    logits_by_model = [[] for _ in models]
    target_masks, candidate_masks = [], []
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            board, premium, rack, targets = SlotDataset([row], lexicon)[0]
            board, premium, rack = board[None].to(device), premium[None].to(device), rack[None].to(device)
            for index, model in enumerate(models):
                logits_by_model[index].append(model(board, premium, rack).cpu()[0])
            target = torch.zeros(450, dtype=torch.bool); target[targets] = True; target_masks.append(target)
            candidate = torch.zeros(450, dtype=torch.bool); candidate[list(candidate_slots(row, lexicon))] = True; candidate_masks.append(candidate)
            if i == 1 or i % 50 == 0 or i == len(rows): print(f"evaluated {i}/{len(rows)}", flush=True)
    target = torch.stack(target_masks); candidate = torch.stack(candidate_masks)
    per = [torch.stack(values) for values in logits_by_model]
    ensemble = torch.stack(per).mean(dim=0)
    output = {"boards": len(rows), "unique_games": len({r["source_game_id"] for r in rows}), "checkpoints": [str(p) for p in args.checkpoint]}
    output["ensemble"] = {**slot_metrics(ensemble, target), **factor_metrics(ensemble, target)}
    output["seeds"] = [{"checkpoint": str(p), **slot_metrics(logits, target)} for p, logits in zip(args.checkpoint, per, strict=True)]
    for k in (1, 8, 32, 96):
        top = torch.stack([x.topk(k, dim=1).indices for x in per], dim=0)
        top = top.permute(1, 0, 2).reshape(len(rows), -1)
        union_hit = target.gather(1, top).any(dim=1).float().mean() * 100
        output.setdefault("oracle_union_pct", {})[f"recall_at_{k}"] = float(union_hit)
    # The saved candidate_moves are a 16-move solver shortlist, not the full legal space.
    restricted = ensemble.masked_fill(~candidate, -torch.inf)
    output["candidate_shortlist_coverage_pct"] = float(((target & candidate).any(dim=1).float().mean()) * 100)
    output["candidate_shortlist_ranked"] = {**slot_metrics(restricted, target), **factor_metrics(restricted, target)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2) + "\n"); print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
