from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from .constants import BOARD_SIZE, LETTER_MULTIPLIERS, WORD_MULTIPLIERS
from .lexicon import Lexicon
from .solver import grid_from_position, validate_and_score_move
from .v41_training import move_plan
from .v42_training import placements_for_start


PAD, BOS, SEP, EOS = range(4)
LETTER_OFFSET = 4
ROW_OFFSET = LETTER_OFFSET + 26
COL_OFFSET = ROW_OFFSET + BOARD_SIZE
DIR_OFFSET = COL_OFFSET + BOARD_SIZE
VOCAB_SIZE = DIR_OFFSET + 2


def premium_code(row: int, col: int) -> int:
    lm, wm = LETTER_MULTIPLIERS[row][col], WORD_MULTIPLIERS[row][col]
    return {2: 1, 3: 2}.get(lm, {2: 3, 3: 4}.get(wm, 0))


def encode_position(position: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    board = torch.zeros(BOARD_SIZE * BOARD_SIZE, dtype=torch.long)
    for cell in position["board"]:
        letter = ord(str(cell["letter"]).upper()) - ord("A") + 1
        if cell.get("is_blank", False):
            letter += 26
        board[int(cell["row"]) * BOARD_SIZE + int(cell["col"])] = letter
    premium = torch.tensor(
        [premium_code(row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE)],
        dtype=torch.long,
    )
    rack = torch.zeros(27, dtype=torch.float32)
    for letter in str(position["rack"]):
        rack[26 if letter == "?" else ord(letter) - ord("A")] += 1
    return board, premium, rack


def plan_tokens(plan: dict[str, Any]) -> list[int]:
    direction = 0 if str(plan["direction"]) == "across" else 1
    return (
        [BOS]
        + [LETTER_OFFSET + ord(letter) - ord("A") for letter in str(plan["word"]).upper()]
        + [
            SEP,
            ROW_OFFSET + int(plan["start_row"]),
            COL_OFFSET + int(plan["start_col"]),
            DIR_OFFSET + direction,
            EOS,
        ]
    )


def tokens_plan(tokens: list[int]) -> dict[str, Any]:
    if not tokens or tokens[0] != BOS or SEP not in tokens:
        raise ValueError("Malformed sequence")
    split = tokens.index(SEP)
    word_ids = tokens[1:split]
    tail = tokens[split + 1 :]
    if len(word_ids) < 2 or len(tail) < 4 or tail[3] != EOS:
        raise ValueError("Incomplete sequence")
    if any(not LETTER_OFFSET <= token < ROW_OFFSET for token in word_ids):
        raise ValueError("Invalid word token")
    row, col, direction = tail[:3]
    if not ROW_OFFSET <= row < COL_OFFSET or not COL_OFFSET <= col < DIR_OFFSET:
        raise ValueError("Invalid coordinate token")
    if direction not in (DIR_OFFSET, DIR_OFFSET + 1):
        raise ValueError("Invalid direction token")
    return {
        "word": "".join(chr(ord("A") + token - LETTER_OFFSET) for token in word_ids),
        "start_row": row - ROW_OFFSET,
        "start_col": col - COL_OFFSET,
        "direction": "across" if direction == DIR_OFFSET else "down",
    }


def target_plans(
    position: dict[str, Any], lexicon: Lexicon, objective: str, threshold: float = 0.8
) -> tuple[list[list[int]], list[float]]:
    if objective == "canonical":
        moves = [position["canonical_optimal_move"]]
    elif objective == "set-valued":
        cutoff = float(position["optimal_score"]) * threshold
        moves = [move for move in position.get("candidate_moves", []) if float(move["score"]) >= cutoff]
        moves.extend(position.get("optimal_moves", []))
    else:
        raise ValueError(f"Unknown objective: {objective}")
    unique: dict[tuple[int, ...], float] = {}
    for move in moves:
        plan = move_plan(position, move["placements"], lexicon)
        tokens = tuple(plan_tokens(plan))
        unique[tokens] = max(unique.get(tokens, -math.inf), float(move["score"]))
    ordered = sorted(unique.items(), key=lambda item: (-item[1], item[0]))
    return [list(tokens) for tokens, _ in ordered], [score for _, score in ordered]


class JointMoveDataset(Dataset):
    def __init__(self, positions: list[dict[str, Any]], lexicon: Lexicon, objective: str) -> None:
        self.items = []
        for position in positions:
            plans, scores = target_plans(position, lexicon, objective)
            self.items.append((position, *encode_position(position), plans, scores))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Any:
        return self.items[index]


def collator(seed: int, temperature: float = 6.0):
    rng = random.Random(seed)

    def collate(batch: list[Any]) -> dict[str, Any]:
        boards, premiums, racks, targets = [], [], [], []
        for _, board, premium, rack, plans, scores in batch:
            weights = [math.exp((score - max(scores)) / temperature) for score in scores]
            target = rng.choices(plans, weights=weights, k=1)[0]
            boards.append(board)
            premiums.append(premium)
            racks.append(rack)
            targets.append(torch.tensor(target, dtype=torch.long))
        width = max(len(target) for target in targets)
        padded = torch.full((len(targets), width), PAD, dtype=torch.long)
        for index, target in enumerate(targets):
            padded[index, : len(target)] = target
        return {
            "board": torch.stack(boards),
            "premium": torch.stack(premiums),
            "rack": torch.stack(racks),
            "target": padded,
        }

    return collate


class JointMoveModel(nn.Module):
    def __init__(
        self, d_model: int = 192, nhead: int = 8, encoder_layers: int = 4,
        decoder_layers: int = 4, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = dict(
            d_model=d_model, nhead=nhead, encoder_layers=encoder_layers,
            decoder_layers=decoder_layers, dropout=dropout,
        )
        self.square = nn.Embedding(53, d_model)
        self.premium = nn.Embedding(5, d_model)
        self.row = nn.Embedding(BOARD_SIZE, d_model)
        self.col = nn.Embedding(BOARD_SIZE, d_model)
        self.rack = nn.Sequential(nn.Linear(27, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_layers, enable_nested_tensor=False)
        self.output_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.output_position = nn.Embedding(24, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE)
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

    def decode(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.output_embedding(tokens) + self.output_position(positions)[None]
        mask = torch.triu(
            torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        decoded = self.decoder(hidden, memory, tgt_mask=mask, tgt_key_padding_mask=tokens.eq(PAD))
        return self.head(self.norm(decoded))

    def forward(
        self, board: torch.Tensor, premium: torch.Tensor, rack: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.decode(self.encode(board, premium, rack), target[:, :-1])


def allowed_tokens(sequence: list[int], lexicon: Lexicon | None = None) -> list[int]:
    if SEP not in sequence:
        word_length = len(sequence) - 1
        if lexicon is None:
            letters = list(range(LETTER_OFFSET, ROW_OFFSET))
            terminal = word_length >= 2
        else:
            node = lexicon.root
            for token in sequence[1:]:
                node = node.children.get(chr(ord("A") + token - LETTER_OFFSET))
                if node is None:
                    return []
            letters = [LETTER_OFFSET + ord(letter) - ord("A") for letter in sorted(node.children)]
            terminal = node.is_word
        if terminal:
            letters.append(SEP)
        return [SEP] if word_length >= 15 and terminal else ([] if word_length >= 15 else letters)
    tail = len(sequence) - sequence.index(SEP) - 1
    return [
        list(range(ROW_OFFSET, COL_OFFSET)),
        list(range(COL_OFFSET, DIR_OFFSET)),
        [DIR_OFFSET, DIR_OFFSET + 1],
        [EOS],
    ][min(tail, 3)]


@torch.inference_mode()
def beam_plans(
    model: JointMoveModel, position: dict[str, Any], beam_size: int, device: torch.device,
    lexicon: Lexicon | None = None,
) -> list[tuple[dict[str, Any], float]]:
    board, premium, rack = encode_position(position)
    memory = model.encode(board[None].to(device), premium[None].to(device), rack[None].to(device))
    beams: list[tuple[list[int], float]] = [([BOS], 0.0)]
    complete: list[tuple[list[int], float]] = []
    for _ in range(20):
        active = [(sequence, score) for sequence, score in beams if sequence[-1] != EOS]
        complete.extend((sequence, score) for sequence, score in beams if sequence[-1] == EOS)
        if not active:
            break
        width = max(len(sequence) for sequence, _ in active)
        tokens = torch.full((len(active), width), PAD, dtype=torch.long, device=device)
        for index, (sequence, _) in enumerate(active):
            tokens[index, : len(sequence)] = torch.tensor(sequence, device=device)
        logits = model.decode(memory.expand(len(active), -1, -1), tokens)
        candidates: list[tuple[list[int], float]] = []
        for index, (sequence, score) in enumerate(active):
            allowed = allowed_tokens(sequence, lexicon)
            if not allowed:
                continue
            log_probs = torch.log_softmax(logits[index, len(sequence) - 1, allowed], dim=-1)
            count = min(beam_size, len(allowed))
            values, indices = torch.topk(log_probs, count)
            for value, choice in zip(values.tolist(), indices.tolist()):
                candidates.append((sequence + [allowed[choice]], score + value))
        candidates.extend(complete)
        beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]
        complete = []
    decoded = []
    seen = set()
    for tokens, score in beams:
        try:
            plan = tokens_plan(tokens)
            key = tuple(plan.values())
            if key not in seen:
                decoded.append((plan, score))
                seen.add(key)
        except ValueError:
            pass
    return decoded


def assess_plan(position: dict[str, Any], plan: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    try:
        placements, _ = placements_for_start(
            position, lexicon, word=plan["word"], direction=plan["direction"],
            start_row=plan["start_row"], start_col=plan["start_col"],
        )
        move = validate_and_score_move(
            lexicon, grid_from_position(position["board"]), position["rack"], placements
        )
        return {"legal": True, "score": move.score, "optimal": move.score == position["optimal_score"]}
    except Exception as error:
        return {"legal": False, "score": 0, "optimal": False, "error": f"{type(error).__name__}: {error}"}
