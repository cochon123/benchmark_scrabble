# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.2",
#   "bitsandbytes>=0.45",
#   "huggingface-hub>=1.0",
#   "peft>=0.14",
#   "torch>=2.6",
#   "transformers>=4.55",
# ]
# ///
from __future__ import annotations

import gc
import hashlib
import json
import os
import statistics
import sys
import tarfile
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


INPUT = Path("/input")
MODEL_REPO = Path("/model")
WORK = Path("/workspace")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_json(api: HfApi, repo: str, path: Path, repo_path: str, value: Any, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=repo_path,
        repo_id=repo,
        commit_message=message,
    )


def load_tokenizer(model_name: str, adapter: Path) -> Any:
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=True)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = AutoTokenizer.from_pretrained(model_name, use_fast=True).chat_template
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name: str, adapter: Path | None) -> Any:
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        torch_dtype=dtype,
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        ),
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def evaluate(label: str, model: Any, tokenizer: Any, records: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    from scrabble_bench.config import resolve_lexicon_path
    from scrabble_bench.evaluation import error_category
    from scrabble_bench.lexicon import Lexicon
    from scrabble_bench.runner import parse_tool_payload
    from scrabble_bench.solver import grid_from_position, validate_and_score_move
    from scrabble_bench.v4_diagnostic import placement_key

    lexicon = Lexicon.from_path(resolve_lexicon_path())
    results: list[dict[str, Any]] = []
    started = time.time()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=True)
            for record in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        responses = tokenizer.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        for record, response in zip(batch, responses, strict=True):
            response = response.strip()
            parsed = False
            output_key = None
            output_error = None
            output_score = 0
            output_legal = False
            try:
                payload = parse_tool_payload(response)
                parsed = True
                placements = payload["arguments"]["placements"]
                output_key = placement_key(placements)
                move = validate_and_score_move(
                    lexicon,
                    grid_from_position(record["position"]["board"]),
                    str(record["position"]["rack"]),
                    placements,
                )
                output_legal = True
                output_score = move.score
            except Exception as error:
                output_error = str(error)

            matched_index = None
            matched = None
            if output_key is not None:
                for index, candidate in enumerate(record["candidates"]):
                    if placement_key(candidate["placements"]) == output_key:
                        matched_index = index
                        matched = candidate
                        break
            target_key = placement_key(record["target_placements"])
            if record["task"] == "ranking":
                success = bool(
                    matched is not None
                    and bool(matched["legal"])
                    and int(matched["score"]) == int(record["position"]["optimal_score"])
                )
            else:
                success = output_key == target_key
            results.append(
                {
                    "id": record["id"],
                    "source_id": record["source_id"],
                    "source_game_id": record["source_game_id"],
                    "band_ply": record["band_ply"],
                    "task": record["task"],
                    "success": success,
                    "parseable": parsed,
                    "candidate_adherent": matched is not None,
                    "selected_candidate": matched_index,
                    "selected_kind": matched.get("kind") if matched else None,
                    "selected_candidate_error_category": matched.get("error_category") if matched else None,
                    "output_legal": output_legal,
                    "output_score": output_score,
                    "output_optimal": output_score == int(record["position"]["optimal_score"]),
                    "output_error": output_error,
                    "output_error_category": None if output_legal else error_category(output_error),
                    "completion_tokens": len(tokenizer(response, add_special_tokens=False)["input_ids"]),
                    "raw_response": response,
                }
            )
        print(f"[{label}] {min(start + batch_size, len(records))}/{len(records)}", flush=True)

    by_task: dict[str, Any] = {}
    for task in sorted({item["task"] for item in results}):
        rows = [item for item in results if item["task"] == task]
        by_task[task] = {
            "boards": len(rows),
            "success": sum(item["success"] for item in rows),
            "accuracy_pct": 100 * sum(item["success"] for item in rows) / len(rows),
            "parseable_pct": 100 * sum(item["parseable"] for item in rows) / len(rows),
            "candidate_adherence_pct": 100 * sum(item["candidate_adherent"] for item in rows) / len(rows),
            "legal_output_pct": 100 * sum(item["output_legal"] for item in rows) / len(rows),
            "optimal_output_pct": 100 * sum(item["output_optimal"] for item in rows) / len(rows),
            "selected_kinds": dict(sorted(Counter(str(item["selected_kind"]) for item in rows).items())),
            "selected_candidate_error_categories": dict(
                sorted(Counter(str(item["selected_candidate_error_category"]) for item in rows).items())
            ),
            "output_error_categories": dict(sorted(Counter(str(item["output_error_category"]) for item in rows).items())),
        }
    lengths = [item["completion_tokens"] for item in results]
    return {
        "summary": {
            "label": label,
            "records": len(results),
            "random_choice_baseline_pct": 12.5,
            "by_task": by_task,
            "overall_parseable_pct": 100 * sum(item["parseable"] for item in results) / len(results),
            "overall_candidate_adherence_pct": 100 * sum(item["candidate_adherent"] for item in results) / len(results),
            "completion_tokens_mean": statistics.mean(lengths),
            "completion_tokens_median": statistics.median(lengths),
            "completion_tokens_max": max(lengths),
            "elapsed_seconds": time.time() - started,
        },
        "results": results,
    }


def main() -> None:
    started = time.time()
    manifest = json.loads((INPUT / "job_manifest.json").read_text(encoding="utf-8"))
    output_repo = str(manifest["output_repo"])
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    status: dict[str, Any] = {"state": "starting", "started_at_utc": utc_now(), "manifest": manifest}
    try:
        for filename in ("diagnostic.json", "manifest.json", "source.tar.gz"):
            actual = sha256(INPUT / filename)
            if actual != manifest["sha256"][filename]:
                raise RuntimeError(f"Hash mismatch for {filename}: {actual}")
        source = WORK / "scrabble_ai"
        source.mkdir(parents=True, exist_ok=True)
        with tarfile.open(INPUT / "source.tar.gz", "r:gz") as archive:
            archive.extractall(source, filter="data")
        sys.path.insert(0, str(source))
        os.chdir(source)
        records = json.loads((INPUT / "diagnostic.json").read_text(encoding="utf-8"))
        adapter = MODEL_REPO / "sft" / "final"
        if not (adapter / "adapter_model.safetensors").exists():
            raise FileNotFoundError(adapter / "adapter_model.safetensors")
        tokenizer = load_tokenizer(str(manifest["base_model"]), adapter)
        outputs: dict[str, Any] = {}
        for label, adapter_path in (("sft", adapter), ("base", None)):
            status.update(state="evaluating", label=label, completed_labels=list(outputs))
            upload_json(api, output_repo, WORK / "status.json", "diagnostic/status.json", status, f"V4 diagnostic: start {label}")
            model = load_model(str(manifest["base_model"]), adapter_path)
            payload = evaluate(label, model, tokenizer, records, int(manifest["batch_size"]))
            outputs[label] = payload["summary"]
            upload_json(
                api,
                output_repo,
                WORK / f"{label}.json",
                f"diagnostic/{label}.json",
                payload,
                f"Upload V4 {label} ability diagnostic",
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        comparison = {
            "protocol": {
                "positions": len(records),
                "tasks": dict(sorted(Counter(item["task"] for item in records).items())),
                "candidates_per_task": 8,
                "decoding": "greedy",
                "max_new_tokens": 128,
                "seed": manifest["seed"],
            },
            "base": outputs["base"],
            "sft": outputs["sft"],
            "accuracy_delta_pct": {
                task: outputs["sft"]["by_task"][task]["accuracy_pct"] - outputs["base"]["by_task"][task]["accuracy_pct"]
                for task in outputs["sft"]["by_task"]
            },
        }
        upload_json(
            api,
            output_repo,
            WORK / "comparison.json",
            "diagnostic/comparison.json",
            comparison,
            "Upload V4 ability diagnostic comparison",
        )
        status.update(
            state="complete",
            completed_at_utc=utc_now(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(manifest["cost_per_hour_usd"]),
            comparison=comparison,
        )
    except Exception as error:
        status.update(
            state="error",
            failed_at_utc=utc_now(),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            elapsed_seconds=time.time() - started,
            estimated_cost_usd=(time.time() - started) / 3600 * float(manifest["cost_per_hour_usd"]),
        )
        raise
    finally:
        upload_json(api, output_repo, WORK / "status.json", "diagnostic/status.json", status, "Update V4 diagnostic status")
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    set_seed(5519)
    main()
