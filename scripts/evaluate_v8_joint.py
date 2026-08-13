#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v8_joint import JointMoveModel, assess_plan, beam_plans


def select(rows, boards):
    chosen, games = [], set()
    bands = sorted({int(row["band_ply"]) for row in rows})
    queues = {band: [row for row in rows if int(row["band_ply"]) == band] for band in bands}
    while len(chosen) < boards:
        progressed = False
        for band in bands:
            while queues[band]:
                row = queues[band].pop(0)
                if row["source_game_id"] not in games:
                    chosen.append(row); games.add(row["source_game_id"]); progressed = True
                    break
            if len(chosen) == boards: break
        if not progressed: break
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--boards", type=int, default=100)
    parser.add_argument("--beam-size", type=int, default=32)
    parser.add_argument("--lexicon-constrained", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = JointMoveModel(**saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"]); model.eval()
    rows = select(json.loads(args.positions.read_text()), args.boards)
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    results = []
    for index, row in enumerate(rows, 1):
        plans = beam_plans(
            model, row, args.beam_size, device,
            lexicon=lexicon if args.lexicon_constrained else None,
        )
        assessed = [{"plan": plan, "log_probability": probability, **assess_plan(row, plan, lexicon)} for plan, probability in plans]
        top = assessed[0] if assessed else {"legal": False, "optimal": False, "score": 0}
        results.append({"id": row["id"], "source_game_id": row["source_game_id"], "band_ply": row["band_ply"], "optimal_score": row["optimal_score"], "top1": top, "beams": assessed})
        print(f"{index}/{len(rows)}", flush=True)
    optimum = sum(row["optimal_score"] for row in results)
    summary = {
        "boards": len(results), "beam_size": args.beam_size,
        "lexicon_constrained": args.lexicon_constrained,
        "top1_legal_pct": 100 * sum(row["top1"]["legal"] for row in results) / len(results),
        "top1_optimal_pct": 100 * sum(row["top1"]["optimal"] for row in results) / len(results),
        "top1_score_pct": 100 * sum(row["top1"]["score"] for row in results) / optimum,
        "pass32_legal_pct": 100 * sum(any(move["legal"] for move in row["beams"]) for row in results) / len(results),
        "pass32_optimal_pct": 100 * sum(any(move["optimal"] for move in row["beams"]) for row in results) / len(results),
        "unique_games": len({row["source_game_id"] for row in results}),
    }
    payload = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
