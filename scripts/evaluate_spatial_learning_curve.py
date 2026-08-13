#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.slot_model import SlotDataset, SpatialSlotModel, factor_metrics, slot_metrics


def choose(rows):
    ordered = sorted(rows, key=lambda item: (int(item["band_ply"]), item["id"])); selected = []; games = set(); bands = sorted({int(row["band_ply"]) for row in ordered}); queues = {band: [row for row in ordered if int(row["band_ply"]) == band] for band in bands}
    while len(selected) < len({row["source_game_id"] for row in rows}):
        progressed = False
        for band in bands:
            while queues[band]:
                row = queues[band].pop(0)
                if row["source_game_id"] not in games: selected.append(row); games.add(row["source_game_id"]); progressed = True; break
        if not progressed: break
    return selected


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, action="append", required=True); parser.add_argument("--positions", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); lexicon = Lexicon.from_path(resolve_lexicon_path()); rows = choose(json.loads(args.positions.read_text()))
    models = []
    for path in args.checkpoint:
        saved = torch.load(path, map_location=device, weights_only=True); model = SpatialSlotModel(**{key: value for key, value in saved["config"].items() if key != "model_type"}).to(device); model.load_state_dict(saved["state_dict"]); model.eval(); models.append(model)
    logits_all = []; masks_all = []
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            board, premium, rack, slots = SlotDataset([row], lexicon)[0]; board = board[None].to(device); premium = premium[None].to(device); rack = rack[None].to(device)
            logits_all.append(torch.stack([model(board, premium, rack).cpu()[0] for model in models]).mean(dim=0, keepdim=True)); mask = torch.zeros(1, 450, dtype=torch.bool); mask[0, slots] = True; masks_all.append(mask)
            if index == 1 or index % 50 == 0 or index == len(rows): print(f"evaluated {index}/{len(rows)}", flush=True)
    logits = torch.cat(logits_all); masks = torch.cat(masks_all); summary = {"boards": len(rows), "unique_games": len({row["source_game_id"] for row in rows}), "metrics": {**slot_metrics(logits, masks), **factor_metrics(logits, masks)}, "checkpoints": [str(path) for path in args.checkpoint]}; args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
