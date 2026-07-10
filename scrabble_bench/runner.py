from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from .cli_agents import cli_metadata, run_cli_completion
from .config import DATASET_PATH, get_openrouter_api_key, normalize_reasoning_effort, resolve_lexicon_path
from .dataset import dataset_subset, load_dataset
from .lexicon import Lexicon
from .models import Placement
from .openrouter import chat_completion, fetch_model_metadata
from .solver import grid_from_position, validate_and_score_move
from .storage import append_log, create_run, record_board_result, should_stop, update_run_status

SPINNER_FRAMES = ["|", "/", "-", "\\"]
ANSI_RESET = "\033[0m"
ANSI_BLUE = "\033[94m"
ANSI_GREEN = "\033[92m"


def _use_color() -> bool:
    return sys.stdout.isatty()


def _colorize(text: str, color: str) -> str:
    if not _use_color():
        return text
    return f"{color}{text}{ANSI_RESET}"


def _board_ascii(position: dict[str, Any]) -> list[str]:
    grid = [["." for _ in range(15)] for _ in range(15)]
    for cell in position["board"]:
        grid[cell["row"]][cell["col"]] = cell["letter"]
    return [f"{row:02d} {' '.join(chars)}" for row, chars in enumerate(grid)]


def _board_display(position: dict[str, Any], placements: list[Placement] | None = None, highlight_color: str | None = None) -> list[str]:
    grid = [["." for _ in range(15)] for _ in range(15)]
    for cell in position["board"]:
        grid[cell["row"]][cell["col"]] = cell["letter"]
    if placements:
        for placement in placements:
            letter = placement.letter.lower() if placement.is_blank else placement.letter
            grid[placement.row][placement.col] = _colorize(letter, highlight_color or "")
    return [f"{row:02d} {' '.join(chars)}" for row, chars in enumerate(grid)]


def _print_board_context(position: dict[str, Any]) -> None:
    print("    rack:", position["rack"], flush=True)
    for line in _board_display(position):
        print(f"    {line}", flush=True)


def _print_scored_board(position: dict[str, Any], placements: list[Placement], is_optimal: bool) -> None:
    highlight_color = ANSI_GREEN if is_optimal else ANSI_BLUE
    label = "best move" if is_optimal else "model move"
    print(f"    {label}:", flush=True)
    for line in _board_display(position, placements, highlight_color):
        print(f"    {line}", flush=True)


def _format_attempted_placements(raw_response: str) -> str:
    try:
        payload = parse_tool_payload(raw_response)
    except Exception:
        return "(could not parse placements from previous response)"
    placements = payload["arguments"]["placements"]
    if not placements:
        return "(none)"
    return ", ".join(
        f"({int(item['row'])},{int(item['col'])})={str(item['letter']).upper()}"
        for item in placements
        if isinstance(item, dict)
    )


def _retry_hint(error: str) -> str:
    if "Rack cannot supply letter" in error:
        return "Use only letters available in the rack. If you need a missing letter, you must consume a ? blank."
    if "collides with existing tile" in error:
        return "Do not place on top of an occupied square unless the letter already matches there. Return only newly placed tiles."
    if "No new tiles were provided" in error:
        return "Return at least one newly placed tile. Do not answer with only existing board letters."
    if "same row or column" in error or "gap in its main word" in error:
        return "All new placements must lie on one line and form one contiguous word with no gaps."
    if "Main word is invalid" in error or "Cross word is invalid" in error or "Word is invalid" in error:
        return "Check the full word formed on the board after your placement, including all touched existing letters."
    if "does not connect to the existing board" in error:
        return "Your move must connect to the existing board through adjacency or overlap with matching existing letters."
    return "Return a legal move using only newly placed tiles and only letters from the rack."


def _retry_feedback(position: dict[str, Any], raw_response: str, error: str) -> str:
    return json.dumps(
        {
            "tool_result": {
                "tool": "play_move",
                "status": "rejected",
                "reason": error,
                "attempted_placements": _format_attempted_placements(raw_response),
                "rack": list(position["rack"]),
                "hint": _retry_hint(error),
                "reminder": [
                    "Return only newly placed tiles.",
                    "Do not re-emit crossing letters already on the board.",
                    "Do not invent letters outside the rack.",
                    "Reply with one raw JSON object and nothing else.",
                ],
            }
        }
    )


def prompt_for_position(
    position: dict[str, Any],
    retry_state: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    system = "\n".join(
        [
            "You are a Scrabble benchmarking agent.",
            "The game is played in English and the loaded dictionary expects valid English words.",
            "Return your move as one final JSON object.",
            'Format: {"tool":"play_move","arguments":{"placements":[{"row":number,"col":number,"letter":"A"}]}}',
            "All row/col coordinates are 0-indexed.",
            "The center square is row 7, col 7.",
            "Goal: return the legal move with the highest immediate raw Scrabble score.",
            "With play_move, you must return only the new tiles placed this turn.",
            "If your word crosses a letter already present on the board, do not return that square in placements.",
            "Never invent letters outside the rack. A ? in the rack may stand for one missing letter.",
            "Keep the visible answer to the JSON object only.",
            "Do not include prose before or after the JSON object.",
        ]
    )
    payload = {
        "benchmark": "highest-immediate-score-only",
        "board_size": 15,
        "rack": list(position["rack"]),
        "board": position["board"],
        "instruction": (
            "Return the legal move with the highest immediate raw Scrabble score. "
            "Return only newly placed tiles in placements."
        ),
        "example": {
            "board_has_existing_word": "AERIE",
            "desired_full_word": "AERIED",
            "correct_response": {"tool": "play_move", "arguments": {"placements": [{"row": 6, "col": 13, "letter": "D"}]}},
        },
    }
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload)},
    ]
    if retry_state:
        messages.extend(
            [
                {"role": "assistant", "content": retry_state["raw_response"]},
                {"role": "user", "content": _retry_feedback(position, retry_state["raw_response"], retry_state["error"])},
            ]
        )
    return messages


def parse_tool_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Response did not contain JSON.")
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict) or payload.get("tool") != "play_move":
        raise ValueError("Response did not use the play_move tool envelope.")
    arguments = payload.get("arguments") or {}
    placements = arguments.get("placements")
    if not isinstance(placements, list):
        raise ValueError("Tool payload is missing arguments.placements.")
    return {"tool": "play_move", "arguments": {"placements": placements}}


def prepare_run(model: str, mode: str, boards: int | None, reasoning_effort: str | None = None) -> dict[str, Any]:
    if mode == "custom" and boards is None:
        raise RuntimeError("Custom runs require an explicit --boards value.")
    board_count = boards if boards is not None else {"smoke": 5, "full": 100}.get(mode, 100)
    effort = normalize_reasoning_effort(reasoning_effort)
    metadata = fetch_model_metadata(model, effort)
    is_local_gateway = metadata["model_id"].startswith("cli2api/")
    if metadata["model_id"] != "demo/mock" and not is_local_gateway and not get_openrouter_api_key():
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to the root .env file before starting a provider run.")
    return create_run(
        model_id=metadata["model_id"],
        model_name=metadata["model_name"],
        company_slug=metadata["company_slug"],
        release_date=metadata["release_date"],
        reasoning_effort=effort,
        mode=mode,
        board_count=board_count,
    )


def prepare_cli_run(
    agent: str,
    mode: str,
    boards: int | None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    if mode == "custom" and boards is None:
        raise RuntimeError("Custom runs require an explicit --boards value.")
    board_count = boards if boards is not None else {"smoke": 5, "full": 100}.get(mode, 100)
    metadata = cli_metadata(agent, model, reasoning_effort=reasoning_effort)
    return create_run(
        model_id=metadata["model_id"],
        model_name=metadata["model_name"],
        company_slug=metadata["company_slug"],
        release_date=metadata["release_date"],
        reasoning_effort=metadata.get("reasoning_effort") or "cli",
        mode=mode,
        board_count=board_count,
    )


def execute_run(run: dict[str, Any], concurrency: int = 1, cli_agent: str | None = None, cli_model: str | None = None) -> None:
    dataset = dataset_subset(run["mode"], run["board_count"])
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    worker_count = 1 if cli_agent else max(1, min(int(concurrency or 1), len(dataset) or 1))
    update_run_status(run["id"], "running")
    append_log(
        run["id"],
        {
            "type": "run_started",
            "run_id": run["id"],
            "model": run["model_id"],
            "reasoning_effort": run.get("reasoning_effort", "medium"),
            "board_count": len(dataset),
            "concurrency": worker_count,
            "runner": "cli" if cli_agent else "api",
            "cli_agent": cli_agent,
            "cli_model": cli_model,
        },
    )

    try:
        if worker_count == 1:
            for index, position in enumerate(dataset, start=1):
                if should_stop(run["id"]):
                    _cancel_run(run["id"])
                    return
                board_result = _run_position(
                    run["id"],
                    run["model_id"],
                    run.get("reasoning_effort", "high"),
                    lexicon,
                    position,
                    index,
                    len(dataset),
                    cli_agent=cli_agent,
                    cli_model=cli_model,
                )
                _record_position_result(run["id"], index, len(dataset), position, board_result)
        else:
            print(f"Running {len(dataset)} boards with concurrency={worker_count}", flush=True)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                pending: dict[Future[dict[str, Any]], tuple[int, dict[str, Any]]] = {}
                next_position = 0

                def submit_next() -> None:
                    nonlocal next_position
                    if next_position >= len(dataset):
                        return
                    index = next_position + 1
                    position = dataset[next_position]
                    next_position += 1
                    pending[
                        executor.submit(
                            _run_position,
                            run["id"],
                            run["model_id"],
                            run.get("reasoning_effort", "high"),
                            lexicon,
                            position,
                            index,
                            len(dataset),
                            cli_agent=None,
                            cli_model=None,
                        )
                    ] = (index, position)

                for _ in range(worker_count):
                    submit_next()

                while pending:
                    if should_stop(run["id"]):
                        for future in pending:
                            future.cancel()
                        _cancel_run(run["id"])
                        return
                    done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    if not done:
                        continue
                    for future in done:
                        index, position = pending.pop(future)
                        board_result = future.result()
                        _record_position_result(run["id"], index, len(dataset), position, board_result)
                        submit_next()
        update_run_status(run["id"], "completed")
        append_log(run["id"], {"type": "run_completed", "run_id": run["id"]})
    except KeyboardInterrupt:
        update_run_status(run["id"], "cancelled", "Interrupted by user.")
        append_log(run["id"], {"type": "run_cancelled", "run_id": run["id"], "error": "Interrupted by user."})
        print("\nInterrupted by user. Marking run as cancelled.", flush=True)
        raise SystemExit(130)
    except Exception as exc:
        update_run_status(run["id"], "failed", str(exc))
        append_log(run["id"], {"type": "run_failed", "run_id": run["id"], "error": str(exc)})
        raise


def _cancel_run(run_id: str) -> None:
    update_run_status(run_id, "cancelled")
    append_log(run_id, {"type": "run_cancelled", "run_id": run_id})
    print("Run cancelled.", flush=True)


def _record_position_result(
    run_id: str,
    index: int,
    total: int,
    position: dict[str, Any],
    board_result: dict[str, Any],
) -> None:
    record_board_result(run_id, board_result)
    append_log(
        run_id,
        {
            "type": "board_result",
            "run_id": run_id,
            "index": index,
            "board_count": total,
            "position_id": position["id"],
            "move_score": board_result["move_score"],
            "optimal_score": board_result["optimal_score"],
            "retry_used": board_result["retry_used"],
            "validation_error": board_result["validation_error"],
            "total_tokens": board_result["total_tokens"],
            "latency_ms": board_result["latency_ms"],
        },
    )
    print(
        f"[{index}/{total}] {position['id']}: "
        f"{board_result['move_score']}/{board_result['optimal_score']} points",
        flush=True,
    )


def _run_position(
    run_id: str,
    model_id: str,
    reasoning_effort: str,
    lexicon: Lexicon,
    position: dict[str, Any],
    index: int,
    total: int,
    cli_agent: str | None = None,
    cli_model: str | None = None,
) -> dict[str, Any]:
    grid = grid_from_position(position["board"])
    attempt_trace: list[dict[str, Any]] = []
    final_error = None
    parsed_move = None
    last_response = ""
    last_reasoning = ""
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    latency_ms = 0

    for attempt_index in (1, 2):
        retry_state = None
        if attempt_index == 2 and final_error is not None:
            retry_state = {"raw_response": last_response, "error": final_error}
        messages = prompt_for_position(position, retry_state)
        append_log(
            run_id,
            {
                "type": "attempt_started",
                "run_id": run_id,
                "index": index,
                "board_count": total,
                "position_id": position["id"],
                "attempt_index": attempt_index,
                "model": model_id,
                "reasoning_effort": reasoning_effort,
            },
        )
        print(f"[{index}/{total}] {position['id']} attempt {attempt_index} output:", flush=True)
        _print_board_context(position)

        progress = {"phase": "sending request", "tick": 0}
        progress_lock = threading.Lock()
        stop_progress = threading.Event()
        started_at = time.perf_counter()

        def progress_printer() -> None:
            last_render = ""
            while not stop_progress.wait(0.2):
                with progress_lock:
                    phase = progress["phase"]
                    tick = progress["tick"]
                elapsed = time.perf_counter() - started_at
                frame = SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]
                line = f"[{index}/{total}] {position['id']} attempt {attempt_index} {frame} {phase} ({elapsed:.1f}s)"
                if sys.stdout.isatty():
                    sys.stdout.write("\r" + line.ljust(max(len(last_render), len(line))))
                    sys.stdout.flush()
                    last_render = line
                elif tick % 10 == 0:
                    print(line, flush=True)
                with progress_lock:
                    progress["tick"] += 1
            if sys.stdout.isatty():
                sys.stdout.write("\r" + (" " * max(len(last_render), 1)) + "\r")
                sys.stdout.flush()

        progress_thread = threading.Thread(target=progress_printer, daemon=True)
        progress_thread.start()

        def on_stream_event(event: dict[str, Any]) -> None:
            event_type = event.get("type")
            text = event.get("text", "")
            append_log(
                run_id,
                {
                    "type": "output_delta",
                    "run_id": run_id,
                    "index": index,
                    "board_count": total,
                    "position_id": position["id"],
                    "attempt_index": attempt_index,
                    "channel": "reasoning" if event_type == "reasoning_delta" else "content",
                    "text": text,
                    "elapsed_ms": event.get("elapsed_ms"),
                    "delta_ms": event.get("delta_ms"),
                    "chars": event.get("chars"),
                },
            )
            if event_type == "content_delta" and text:
                with progress_lock:
                    progress["phase"] = "receiving response"
            elif event_type == "reasoning_delta" and text:
                with progress_lock:
                    progress["phase"] = "receiving reasoning"

        def on_status(message: str) -> None:
            with progress_lock:
                if message.startswith("running "):
                    progress["phase"] = message
                elif message.startswith("POST "):
                    progress["phase"] = "contacting API"
                elif message.startswith("stream opened"):
                    progress["phase"] = "stream opened"
                elif message.startswith("first stream bytes"):
                    progress["phase"] = "waiting for model output"
                elif message.startswith("retrying without streaming"):
                    progress["phase"] = "waiting for model output"
                else:
                    progress["phase"] = message

        def on_status_event(event: dict[str, Any]) -> None:
            status = str(event.get("status") or "").strip()
            detail = event.get("detail")
            if status:
                with progress_lock:
                    progress["phase"] = status if not detail else f"{status}: {str(detail)[:80]}"
            # Status is also mirrored into reasoning_delta by the CLI runner via on_stream_event.

        try:
            if cli_agent:
                response = run_cli_completion(
                    cli_agent,
                    cli_model,
                    messages,
                    on_stream_event=on_stream_event,
                    on_status_event=on_status_event,
                    on_status=on_status,
                    reasoning_effort=reasoning_effort,
                )
            else:
                response = chat_completion(
                    model_id,
                    messages,
                    reasoning_effort=reasoning_effort,
                    on_stream_event=on_stream_event,
                    on_status=on_status,
                )
        except KeyboardInterrupt:
            stop_progress.set()
            progress_thread.join(timeout=0.1)
            raise
        except Exception as exc:
            stop_progress.set()
            progress_thread.join(timeout=0.1)
            final_error = str(exc)
            attempt_trace.append(
                {
                    "attempt": attempt_index,
                    "raw_response": "",
                    "reasoning": "",
                    "status": "invalid",
                    "error": final_error,
                }
            )
            append_log(
                run_id,
                {
                    "type": "attempt_invalid",
                    "run_id": run_id,
                    "index": index,
                    "board_count": total,
                    "position_id": position["id"],
                    "attempt_index": attempt_index,
                    "error": final_error,
                },
            )
            print(
                f"[{index}/{total}] {position['id']} attempt {attempt_index} failed before a valid response: {final_error}",
                flush=True,
            )
            continue
        else:
            stop_progress.set()
            progress_thread.join(timeout=0.1)
        content = response["content"] or ""
        last_reasoning = response.get("reasoning", "") or ""
        reasoning_trace = response.get("reasoning_trace") or {"events": [], "summary": {}}
        usage = response.get("usage") or {}
        token_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        latency_ms = int(response.get("latency_ms", 0) or 0)
        last_response = content
        print(
            f"[{index}/{total}] {position['id']} attempt {attempt_index} done in {latency_ms}ms",
            flush=True,
        )
        try:
            parsed_move = parse_tool_payload(content)
            validated = validate_and_score_move(lexicon, grid, position["rack"], parsed_move["arguments"]["placements"])
            is_optimal = validated.score == position["optimal_score"]
            _print_scored_board(position, validated.placements, is_optimal)
            return {
                "position_id": position["id"],
                "attempt_index": attempt_index,
                "raw_response": content,
                "parsed_move": parsed_move,
                "validation_error": None,
                "attempt_trace": attempt_trace
                + [
                    {
                        "attempt": attempt_index,
                        "raw_response": content,
                        "reasoning": last_reasoning,
                        "reasoning_trace": reasoning_trace,
                        "status": "ok",
                    }
                ],
                "retry_used": attempt_index == 2,
                "move_score": validated.score,
                "optimal_score": position["optimal_score"],
                "is_optimal": int(is_optimal),
                **token_usage,
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            final_error = str(exc)
            attempt_trace.append(
                {
                    "attempt": attempt_index,
                    "raw_response": content,
                    "reasoning": last_reasoning,
                    "reasoning_trace": reasoning_trace,
                    "status": "invalid",
                    "error": final_error,
                }
            )
            append_log(
                run_id,
                {
                    "type": "attempt_invalid",
                    "run_id": run_id,
                    "index": index,
                    "board_count": total,
                    "position_id": position["id"],
                    "attempt_index": attempt_index,
                    "error": final_error,
                },
            )
            print(
                f"[{index}/{total}] {position['id']} attempt {attempt_index} invalid: {final_error}",
                flush=True,
            )

    return {
        "position_id": position["id"],
        "attempt_index": 2,
        "raw_response": last_response,
        "parsed_move": parsed_move,
        "validation_error": final_error,
        "attempt_trace": attempt_trace,
        "retry_used": True,
        "move_score": 0,
        "optimal_score": position["optimal_score"],
        "is_optimal": 0,
        **token_usage,
        "latency_ms": latency_ms,
    }
