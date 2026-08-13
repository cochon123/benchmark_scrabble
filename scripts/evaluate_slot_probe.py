#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.slot_model import LocalSlotProbe, SlotDataset, SlotModel, factor_metrics, slot_metrics


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, action="append", required=True)
    parser.add_argument("--probe", type=Path, action="append", required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.base) != len(args.probe): raise SystemExit("Need one probe per base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lexicon = Lexicon.from_path(resolve_lexicon_path()); rows = choose(json.loads(args.positions.read_text()))
    bases = []; probes = []
    for base_path, probe_path in zip(args.base, args.probe, strict=True):
        saved = torch.load(base_path, map_location=device, weights_only=True)
        base = SlotModel(**saved["config"]).to(device); base.load_state_dict(saved["state_dict"]); base.eval(); bases.append(base)
        saved_probe = torch.load(probe_path, map_location=device, weights_only=True)
        probe = LocalSlotProbe(saved_probe["d_model"]).to(device); probe.load_state_dict(saved_probe["state_dict"]); probe.eval(); probes.append(probe)
    global_logits = []; local_logits = []; masks = []
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            board, premium, rack, slots = SlotDataset([row], lexicon)[0]
            board = board[None].to(device); premium = premium[None].to(device); rack = rack[None].to(device)
            globals_one = []; locals_one = []
            for base, probe in zip(bases, probes, strict=True):
                encoded = base.encode(board, premium, rack)
                globals_one.append(base.head(base.norm(encoded.mean(dim=1))).cpu())
                locals_one.append(probe(encoded).cpu())
            global_logits.append(torch.stack(globals_one).mean(dim=0)); local_logits.append(torch.stack(locals_one).mean(dim=0))
            mask = torch.zeros(1, 450, dtype=torch.bool); mask[0, slots] = True; masks.append(mask)
            if index == 1 or index % 50 == 0 or index == len(rows): print(f"evaluated {index}/{len(rows)}", flush=True)
    global_logits = torch.cat(global_logits); local_logits = torch.cat(local_logits); masks = torch.cat(masks)
    summary = {
        "boards": len(rows), "unique_games": len({row["source_game_id"] for row in rows}),
        "global_head": {**slot_metrics(global_logits, masks), **factor_metrics(global_logits, masks)},
        "local_probe": {**slot_metrics(local_logits, masks), **factor_metrics(local_logits, masks)},
        "base_checkpoints": [{"path": str(path), "sha256": sha256(path)} for path in args.base],
        "probe_checkpoints": [{"path": str(path), "sha256": sha256(path)} for path in args.probe],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps({"summary": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
