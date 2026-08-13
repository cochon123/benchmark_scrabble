#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.insert(0, "/content/scrabble_ai")

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import parse_tool_payload, prompt_for_position
from scrabble_bench.solver import grid_from_position, validate_and_score_move


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
ADAPTER = Path("/content/checkpoint-100")
DATASET = Path("/content/scrabble_ai/data/general_v2_sft/selection_positions.json")
OUTPUT = Path("/content/general-snapshot-step100-cpu.json")


def score_response(position: dict[str, Any], response: str, lexicon: Lexicon) -> dict[str, Any]:
    result: dict[str, Any] = {
        "raw_response": response.strip(),
        "score": 0,
        "optimal_score": int(position["optimal_score"]),
        "legal": False,
        "optimal": False,
        "placements": [],
        "error": None,
    }
    try:
        payload = parse_tool_payload(result["raw_response"])
        move = validate_and_score_move(
            lexicon,
            grid_from_position(position["board"]),
            position["rack"],
            payload["arguments"]["placements"],
        )
        result.update(
            score=move.score,
            legal=True,
            optimal=move.score == int(position["optimal_score"]),
            placements=[item.to_dict() for item in move.placements],
        )
    except Exception as error:
        result["error"] = str(error)
    return result


def main() -> None:
    os.chdir("/content/scrabble_ai")
    with tarfile.open("/content/scrabble_checkpoint_100.tar.gz", "r:gz") as archive:
        archive.extractall("/content", filter="data")
    set_seed(3407)
    position = json.loads(DATASET.read_text(encoding="utf-8"))[0]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = tokenizer.apply_chat_template(
        prompt_for_position(position),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    results: dict[str, Any] = {
        "protocol": {
            "purpose": "qualitative diagnostic only; excluded from checkpoint selection",
            "dataset": str(DATASET),
            "position_index": 0,
            "position_id": position["id"],
            "checkpoint": 100,
            "decoding": "greedy",
            "max_new_tokens": 192,
            "seed": 3407,
            "device": "cpu",
            "dtype": "bfloat16",
        },
        "position": position,
        "generations": {},
    }
    for name, disable_adapter in (("base", True), ("checkpoint-100", False)):
        started = time.time()
        context = model.disable_adapter() if disable_adapter else contextlib.nullcontext()
        with context, torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        scored = score_response(position, response, lexicon)
        scored["elapsed_seconds"] = time.time() - started
        results["generations"][name] = scored
        print(json.dumps({"name": name, **scored}, indent=2), flush=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved snapshot comparison to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
