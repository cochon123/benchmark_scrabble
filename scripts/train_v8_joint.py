#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v8_joint import JointMoveDataset, JointMoveModel, PAD, collator


def loss_for(model, batch, device):
    batch = {key: value.to(device) for key, value in batch.items()}
    logits = model(batch["board"], batch["premium"], batch["rack"], batch["target"])
    labels = batch["target"][:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=PAD)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--objective", choices=("canonical", "set-valued"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    train_rows = json.loads(args.train.read_text(encoding="utf-8"))
    validation_rows = json.loads(args.validation.read_text(encoding="utf-8"))
    train_data = JointMoveDataset(train_rows, lexicon, args.objective)
    validation_data = JointMoveDataset(validation_rows, lexicon, "canonical")
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator(args.seed))
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, collate_fn=collator(0))
    model = JointMoveModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr / 10)
    history = []
    iterator = iter(loader)
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, batch, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            event = {"step": step, "train_loss": float(loss), "lr": scheduler.get_last_lr()[0]}
            history.append(event)
            print(json.dumps(event), flush=True)
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in validation_loader:
            losses.append(float(loss_for(model, batch, device)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, args.output)
    metrics = {
        "objective": args.objective, "seed": args.seed, "steps": args.steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "validation_loss": sum(losses) / len(losses),
        "elapsed_seconds": time.time() - started, "device": str(device), "history": history,
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
