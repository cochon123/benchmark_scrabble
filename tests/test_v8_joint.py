from __future__ import annotations

import json

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v41_training import move_plan
from scrabble_bench.v8_joint import (
    BOS,
    EOS,
    SEP,
    allowed_tokens,
    assess_plan,
    encode_position,
    plan_tokens,
    target_plans,
    tokens_plan,
)


def _position():
    return json.loads(DATASET_PATH.read_text())[0]


def test_plan_token_round_trip() -> None:
    position = _position()
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    plan = move_plan(position, position["canonical_optimal_move"]["placements"], lexicon)
    decoded = tokens_plan(plan_tokens(plan))
    assert decoded == {key: plan[key] for key in ("word", "start_row", "start_col", "direction")}


def test_position_encoding_and_canonical_assessment() -> None:
    position = _position()
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    board, premium, rack = encode_position(position)
    assert board.shape == (225,)
    assert premium.shape == (225,)
    assert rack.sum().item() == len(position["rack"])
    plan = move_plan(position, position["canonical_optimal_move"]["placements"], lexicon)
    result = assess_plan(position, plan, lexicon)
    assert result["legal"] and result["optimal"]


def test_set_valued_targets_and_lexicon_mask() -> None:
    position = _position()
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    canonical, _ = target_plans(position, lexicon, "canonical")
    targets, _ = target_plans(position, lexicon, "set-valued")
    assert canonical[0] in targets
    assert SEP not in allowed_tokens([BOS], lexicon)
    first = plan_tokens(move_plan(position, position["canonical_optimal_move"]["placements"], lexicon))[1]
    assert first in allowed_tokens([BOS], lexicon)
    complete_word = canonical[0][:-5]
    assert SEP in allowed_tokens(complete_word, lexicon)
    assert EOS == canonical[0][-1]
