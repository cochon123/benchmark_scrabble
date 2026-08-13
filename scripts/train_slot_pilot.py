#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.slot_model import SlotDataset, SlotModel, collate, set_nll, slot_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    train = SlotDataset(json.loads(args.train.read_text()), lexicon)
    validation = SlotDataset(json.loads(args.validation.read_text()), lexicon)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate)
    model = SlotModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr / 10)
    iterator = iter(loader); history = []; started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        try: batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = set_nll(model(batch["board"], batch["premium"], batch["rack"]), batch["target_mask"])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            history.append({"step": step, "train_loss": float(loss), "lr": scheduler.get_last_lr()[0]})
            print(json.dumps(history[-1]), flush=True)
    model.eval(); losses = []; logits_all = []; masks_all = []
    with torch.inference_mode():
        for batch in val_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch["board"], batch["premium"], batch["rack"])
            losses.append(float(set_nll(logits, batch["target_mask"])))
            logits_all.append(logits.cpu()); masks_all.append(batch["target_mask"].cpu())
    val_logits = torch.cat(logits_all); val_masks = torch.cat(masks_all)
    metrics = {
        "seed": args.seed, "steps": args.steps, "parameters": sum(p.numel() for p in model.parameters()),
        "validation_loss": sum(losses) / len(losses), "validation": slot_metrics(val_logits, val_masks),
        "device": str(device), "elapsed_seconds": time.time() - started, "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, args.output)
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
