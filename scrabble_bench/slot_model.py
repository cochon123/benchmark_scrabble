from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from .constants import BOARD_SIZE
from .lexicon import Lexicon
from .v41_training import move_plan
from .v8_joint import encode_position


SLOT_COUNT = BOARD_SIZE * BOARD_SIZE * 2


def slot_index(row: int, col: int, direction: str) -> int:
    return (0 if str(direction) == "across" else 1) * BOARD_SIZE * BOARD_SIZE + int(row) * BOARD_SIZE + int(col)


def decode_slot(index: int) -> dict[str, Any]:
    direction, remainder = divmod(int(index), BOARD_SIZE * BOARD_SIZE)
    row, col = divmod(remainder, BOARD_SIZE)
    return {"row": row, "col": col, "direction": "across" if direction == 0 else "down"}


def optimal_slots(position: dict[str, Any], lexicon: Lexicon) -> list[int]:
    moves = [position["canonical_optimal_move"], *(position.get("optimal_moves") or [])]
    slots = set()
    for move in moves:
        plan = move_plan(position, move["placements"], lexicon)
        slots.add(slot_index(plan["start_row"], plan["start_col"], plan["direction"]))
    return sorted(slots)


class SlotDataset(Dataset):
    def __init__(self, positions: list[dict[str, Any]], lexicon: Lexicon) -> None:
        self.items = []
        for position in positions:
            board, premium, rack = encode_position(position)
            self.items.append((board, premium, rack, optimal_slots(position, lexicon)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Any:
        return self.items[index]


def collate(batch: list[Any]) -> dict[str, torch.Tensor]:
    board, premium, rack, slots = zip(*batch, strict=True)
    mask = torch.zeros(len(batch), SLOT_COUNT, dtype=torch.bool)
    for index, target in enumerate(slots):
        mask[index, target] = True
    return {"board": torch.stack(board), "premium": torch.stack(premium), "rack": torch.stack(rack), "target_mask": mask}


class SlotModel(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8, layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.config = {"d_model": d_model, "nhead": nhead, "layers": layers, "dropout": dropout}
        self.square = nn.Embedding(53, d_model)
        self.premium = nn.Embedding(5, d_model)
        self.row = nn.Embedding(BOARD_SIZE, d_model)
        self.col = nn.Embedding(BOARD_SIZE, d_model)
        self.rack = nn.Sequential(nn.Linear(27, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, SLOT_COUNT)
        rows = torch.arange(BOARD_SIZE).repeat_interleave(BOARD_SIZE)
        cols = torch.arange(BOARD_SIZE).repeat(BOARD_SIZE)
        self.register_buffer("row_ids", rows, persistent=False)
        self.register_buffer("col_ids", cols, persistent=False)

    def encode(self, board: torch.Tensor, premium: torch.Tensor, rack: torch.Tensor) -> torch.Tensor:
        hidden = (
            self.square(board) + self.premium(premium)
            + self.row(self.row_ids)[None] + self.col(self.col_ids)[None]
            + self.rack(rack)[:, None]
        )
        return self.encoder(hidden)

    def forward(self, board: torch.Tensor, premium: torch.Tensor, rack: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(board, premium, rack)
        pooled = encoded.mean(dim=1)
        return self.head(self.norm(pooled))


class LocalSlotProbe(nn.Module):
    """Shared per-square direction head over a frozen spatial encoder."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        scores = self.head(self.norm(encoded))
        return torch.cat((scores[:, :, 0], scores[:, :, 1]), dim=1)


class SpatialSlotModel(SlotModel):
    """End-to-end encoder trained with the local per-square slot head."""

    def __init__(self, d_model: int = 128, nhead: int = 8, layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__(d_model=d_model, nhead=nhead, layers=layers, dropout=dropout)
        self.local_head = LocalSlotProbe(d_model)
        self.config["model_type"] = "spatial-local"

    def forward(self, board: torch.Tensor, premium: torch.Tensor, rack: torch.Tensor) -> torch.Tensor:
        return self.local_head(self.encode(board, premium, rack))


def set_nll(logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    log_partition = torch.logsumexp(logits, dim=-1)
    selected = logits.masked_fill(~target_mask, -math.inf)
    log_selected_mean = torch.logsumexp(selected, dim=-1) - target_mask.sum(dim=-1).float().log()
    return (log_partition - log_selected_mean).mean()


def slot_metrics(logits: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float]:
    order = logits.argsort(dim=-1, descending=True)
    output: dict[str, float] = {}
    for k in (1, 8, 32):
        hits = target_mask.gather(1, order[:, :k]).any(dim=1).float().mean()
        output[f"optimal_recall_at_{k}_pct"] = float(hits * 100)
    output["top1_optimal_pct"] = output["optimal_recall_at_1_pct"]
    return output


def factor_metrics(logits: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float]:
    scores = logits.view(-1, 2, BOARD_SIZE, BOARD_SIZE)
    targets = target_mask.view(-1, 2, BOARD_SIZE, BOARD_SIZE)
    row_scores = torch.logsumexp(torch.logsumexp(scores, dim=1), dim=-1)
    col_scores = torch.logsumexp(torch.logsumexp(scores, dim=1), dim=1)
    direction_scores = torch.logsumexp(scores.flatten(2), dim=-1)
    row_targets = targets.any(dim=1).any(dim=-1)
    col_targets = targets.any(dim=1).any(dim=1)
    direction_targets = targets.flatten(2).any(dim=-1)
    return {
        "row_top1_pct": float(row_targets.gather(1, row_scores.argmax(dim=1, keepdim=True)).float().mean() * 100),
        "col_top1_pct": float(col_targets.gather(1, col_scores.argmax(dim=1, keepdim=True)).float().mean() * 100),
        "direction_top1_pct": float(direction_targets.gather(1, direction_scores.argmax(dim=1, keepdim=True)).float().mean() * 100),
    }
