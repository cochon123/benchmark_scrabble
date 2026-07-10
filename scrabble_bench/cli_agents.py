from __future__ import annotations

import json
import queue
import random
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


CLI_TIMEOUT_SECONDS = 300
# Delay between fake-streamed words so the UI animates (Codex emits whole messages).
WORD_STREAM_DELAY_RANGE = (0.015, 0.040)
_CONTENT_ITEM_TYPES = frozenset({"agent_message", "assistant_message", "message"})
_REASONING_ITEM_TYPES = frozenset({"reasoning", "thought", "thinking"})


@dataclass(frozen=True)
class CliAgent:
    id: str
    name: str
    executable: str


CLI_AGENTS = {
    "claude": CliAgent("claude", "Claude Code", "claude"),
    "opencode": CliAgent("opencode", "OpenCode", "opencode"),
    "codex": CliAgent("codex", "Codex CLI", "codex"),
}


def normalize_agent_id(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    aliases = {
        "claude-code": "claude",
        "claude": "claude",
        "opencode": "opencode",
        "open-code": "opencode",
        "codex-cli": "codex",
        "codex": "codex",
    }
    if normalized not in aliases:
        known = ", ".join(agent.name for agent in CLI_AGENTS.values())
        raise RuntimeError(f"Unknown CLI agent '{value}'. Expected one of: {known}.")
    return aliases[normalized]


def list_cli_agents() -> list[dict[str, Any]]:
    return [agent_payload(agent) for agent in CLI_AGENTS.values()]


def agent_payload(agent: CliAgent) -> dict[str, Any]:
    available = shutil.which(agent.executable) is not None
    return {
        "id": agent.id,
        "name": agent.name,
        "executable": agent.executable,
        "available": available,
        "models": list_cli_models(agent.id) if available else [],
    }


def list_cli_models(agent_id: str) -> list[dict[str, str]]:
    agent = CLI_AGENTS[normalize_agent_id(agent_id)]
    if not shutil.which(agent.executable):
        return []
    if agent.id == "opencode":
        result = subprocess.run(
            [agent.executable, "models"],
            cwd=tempfile.gettempdir(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        models = []
        for line in result.stdout.splitlines():
            model = line.strip()
            if model and "/" in model:
                models.append({"id": model, "name": model})
        return models
    if agent.id == "codex":
        return _list_codex_models()
    if agent.id == "claude":
        # Claude Code model ids are free-form; expose a few common aliases.
        return [
            {"id": "sonnet", "name": "Claude Sonnet (default alias)"},
            {"id": "opus", "name": "Claude Opus"},
            {"id": "haiku", "name": "Claude Haiku"},
        ]
    return []


def _list_codex_models() -> list[dict[str, str]]:
    cache = Path.home() / ".codex" / "models_cache.json"
    if not cache.exists():
        return [{"id": "gpt-5.6-terra", "name": "GPT-5.6-Terra"}]
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [{"id": "gpt-5.6-terra", "name": "GPT-5.6-Terra"}]

    raw_models: list[Any]
    if isinstance(payload, list):
        raw_models = payload
    elif isinstance(payload, dict):
        raw_models = payload.get("models") or payload.get("data") or []
        if not raw_models:
            for value in payload.values():
                if isinstance(value, list) and value and isinstance(value[0], dict) and "slug" in value[0]:
                    raw_models = value
                    break
    else:
        raw_models = []

    models: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug") or item.get("id")
        if not isinstance(slug, str) or not slug or slug in seen:
            continue
        # Skip non-listable / internal entries when visibility is present.
        visibility = item.get("visibility")
        if visibility is not None and visibility not in {"list", "default", "visible", True}:
            continue
        name = item.get("display_name") or item.get("name") or slug
        models.append({"id": slug, "name": str(name)})
        seen.add(slug)
    if "gpt-5.6-terra" not in seen:
        models.insert(0, {"id": "gpt-5.6-terra", "name": "GPT-5.6-Terra"})
    return models


def parse_cli_model_slug(slug: str) -> tuple[str, str | None]:
    """Parse `cli/<agent>` or `cli/<agent>/<model...>` into agent id + model."""
    parts = slug.strip().split("/", 2)
    if len(parts) < 2 or parts[0] != "cli":
        raise RuntimeError(f"Not a CLI model slug: {slug}")
    agent = normalize_agent_id(parts[1])
    model = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
    return agent, model


def search_cli_models(query: str) -> list[dict[str, Any]]:
    """Autocomplete rows shaped like OpenRouter search results."""
    needle = query.strip().lower()
    if not needle:
        return []
    results: list[dict[str, Any]] = []
    for agent in CLI_AGENTS.values():
        available = shutil.which(agent.executable) is not None
        if not available:
            continue
        # Always include the bare agent entry.
        bare = {
            "slug": f"cli/{agent.id}",
            "name": f"{agent.name} (default model)",
            "author": "cli",
            "source": "cli",
        }
        hay_bare = f"{bare['slug']} {bare['name']} {agent.id}".lower()
        if needle in hay_bare:
            results.append(bare)
        for model in list_cli_models(agent.id):
            slug = f"cli/{agent.id}/{model['id']}"
            row = {
                "slug": slug,
                "name": f"{agent.name} · {model['name']}",
                "author": "cli",
                "source": "cli",
            }
            hay = f"{slug} {row['name']} {model['id']}".lower()
            if needle in hay:
                results.append(row)
    # Prefer exact-ish matches first.
    results.sort(key=lambda row: (0 if needle in row["slug"].lower() else 1, row["slug"]))
    return results


def cli_model_id(agent_name: str, model: str | None = None) -> str:
    agent = CLI_AGENTS[normalize_agent_id(agent_name)]
    suffix = f"/{model.strip()}" if model and model.strip() else ""
    return f"cli/{agent.id}{suffix}"


def cli_metadata(
    agent_name: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    agent = CLI_AGENTS[normalize_agent_id(agent_name)]
    label = model.strip() if model and model.strip() else "default"
    effort = (reasoning_effort or "").strip().lower() or None
    name = f"{agent.name} ({label})"
    if effort:
        name = f"{name} [{effort}]"
    return {
        "model_id": cli_model_id(agent.name, model),
        "model_name": name,
        "company_slug": "cli",
        "release_date": None,
        "supports_reasoning": bool(effort),
        "reasoning_effort": effort or "cli",
    }


def _guarded_prompt(messages: list[dict[str, str]]) -> str:
    sections = [
        "You are participating in a Scrabble benchmark.",
        "Do not use tools, files, shell commands, web search, network lookups, code execution, or external references.",
        "Use only the board and rack text in this prompt.",
        "Return exactly one raw JSON object and no prose.",
        "",
    ]
    for message in messages:
        sections.append(f"{message['role'].upper()}:\n{message['content']}")
        sections.append("")
    return "\n".join(sections)


def run_cli_completion(
    agent_name: str,
    model: str | None,
    messages: list[dict[str, str]],
    on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    on_status_event: Callable[[dict[str, Any]], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    agent = CLI_AGENTS[normalize_agent_id(agent_name)]
    if not shutil.which(agent.executable):
        raise RuntimeError(f"{agent.name} executable '{agent.executable}' was not found on PATH.")

    prompt = _guarded_prompt(messages)
    command = _command_for_agent(agent, model, prompt, reasoning_effort=reasoning_effort)
    start = time.perf_counter()
    early_trace: list[dict[str, Any]] = []
    _emit_cli_status(
        on_status_event,
        f"starting {agent.name}",
        0,
        on_stream_event=on_stream_event,
        trace_events=early_trace,
        start=start,
    )
    if on_status is not None:
        on_status(f"running {agent.name}")
    with tempfile.TemporaryDirectory(prefix="scrabble-bench-cli-") as workdir:
        content, reasoning, stderr_text, returncode, trace_events = _run_streaming_process(
            command,
            workdir,
            agent,
            on_stream_event,
            on_status_event,
            start,
        )
    if early_trace:
        trace_events = [*early_trace, *trace_events]
    latency_ms = int((time.perf_counter() - start) * 1000)
    if returncode != 0:
        details = stderr_text.strip() or content.strip() or f"exit code {returncode}"
        raise RuntimeError(f"{agent.name} failed: {details}")
    return {
        "content": content,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "latency_ms": latency_ms,
        "provider": agent.id,
        "model": model or "default",
        "reasoning": reasoning,
        "reasoning_trace": {
            "events": trace_events or [{"type": "cli_response", "elapsed_ms": latency_ms, "chars": len(content)}],
            "summary": {
                "latency_ms": latency_ms,
                "content_chars": len(content),
                "reasoning_chars": len(reasoning),
            },
        },
    }


def _run_streaming_process(
    command: list[str],
    workdir: str,
    agent: CliAgent,
    on_stream_event: Callable[[dict[str, Any]], None] | None,
    on_status_event: Callable[[dict[str, Any]], None] | None,
    start: float,
    word_delay_range: tuple[float, float] = WORD_STREAM_DELAY_RANGE,
) -> tuple[str, str, str, int, list[dict[str, Any]]]:
    proc = subprocess.Popen(
        command,
        cwd=workdir,
        text=True,
        stdin=subprocess.DEVNULL,  # Codex waits forever if stdin stays open
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    events: queue.Queue[tuple[str, str]] = queue.Queue()
    stdout_parts: list[str] = []
    reasoning_parts: list[str] = []
    stderr_parts: list[str] = []
    trace_events: list[dict[str, Any]] = []

    def reader(name: str, stream: Any) -> None:
        try:
            for line in stream:
                events.put((name, line))
        finally:
            events.put((name, ""))

    assert proc.stdout is not None and proc.stderr is not None
    _emit_cli_status(
        on_status_event,
        f"{agent.name} process started",
        int((time.perf_counter() - start) * 1000),
        on_stream_event=on_stream_event,
        trace_events=trace_events,
        start=start,
        reasoning_parts=reasoning_parts,
    )
    threading.Thread(target=reader, args=("stdout", proc.stdout), daemon=True).start()
    threading.Thread(target=reader, args=("stderr", proc.stderr), daemon=True).start()

    finished_streams: set[str] = set()
    deadline = time.monotonic() + CLI_TIMEOUT_SECONDS
    last_event_at = start
    last_status_at = time.monotonic()

    while len(finished_streams) < 2:
        if time.monotonic() > deadline:
            proc.kill()
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            _emit_cli_status(
                on_status_event,
                f"{agent.name} timed out after {CLI_TIMEOUT_SECONDS}s",
                elapsed_ms,
                on_stream_event=on_stream_event,
                trace_events=trace_events,
                start=start,
                reasoning_parts=reasoning_parts,
            )
            raise TimeoutError(f"{agent.name} did not finish within {CLI_TIMEOUT_SECONDS} seconds.")
        try:
            stream_name, line = events.get(timeout=0.2)
        except queue.Empty:
            if time.monotonic() - last_status_at >= 2:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                _emit_cli_status(
                    on_status_event,
                    f"waiting for {agent.name} output",
                    elapsed_ms,
                    on_stream_event=on_stream_event,
                    trace_events=trace_events,
                    start=start,
                    reasoning_parts=reasoning_parts,
                )
                last_status_at = time.monotonic()
            if proc.poll() is not None and len(finished_streams) == 2:
                break
            continue
        if line == "":
            finished_streams.add(stream_name)
            continue
        now = time.perf_counter()
        elapsed_ms = int((now - start) * 1000)
        delta_ms = int((now - last_event_at) * 1000)
        last_event_at = now
        if stream_name == "stderr":
            stderr_parts.append(line)
            detail = line.strip()
            _emit_cli_status(
                on_status_event,
                f"{agent.name} stderr",
                elapsed_ms,
                detail=detail,
                on_stream_event=on_stream_event,
                trace_events=trace_events,
                start=start,
                reasoning_parts=reasoning_parts,
            )
            continue
        if agent.id == "opencode":
            text, channel, event_type = _parse_opencode_event(line)
            if text:
                if channel == "content":
                    stdout_parts.append(text)
                else:
                    reasoning_parts.append(text)
                _emit_cli_event(on_stream_event, f"{channel}_delta", text, elapsed_ms, delta_ms, trace_events, event_type)
            elif event_type:
                crumb = f"{event_type}\n"
                reasoning_parts.append(crumb)
                _emit_cli_event(on_stream_event, "reasoning_delta", crumb, elapsed_ms, delta_ms, trace_events, event_type)
                _emit_cli_status(
                    on_status_event,
                    f"{agent.name} {event_type}",
                    elapsed_ms,
                )
        elif agent.id == "codex":
            text, channel, event_type = parse_codex_stream_event(line)
            if not text:
                if event_type:
                    _emit_cli_status(
                        on_status_event,
                        f"{agent.name} {event_type}",
                        elapsed_ms,
                        on_stream_event=on_stream_event,
                        trace_events=trace_events,
                        start=start,
                        reasoning_parts=reasoning_parts,
                    )
                continue
            if channel == "content":
                # Skip duplicate final last_agent_message when we already streamed content.
                if event_type == "last_agent_message" and stdout_parts:
                    continue
                # Codex JSONL emits the full agent message when complete —
                # keep only the latest complete message, then fake-stream words.
                if event_type in _CONTENT_ITEM_TYPES | {"item.completed", "last_agent_message"}:
                    stdout_parts.clear()
                stdout_parts.append(text)
                last_event_at = _fake_stream_content(
                    text,
                    on_stream_event,
                    start,
                    last_event_at,
                    trace_events,
                    event_type,
                    word_delay_range=word_delay_range,
                )
            else:
                crumb = text if text.endswith("\n") else f"{text}\n"
                reasoning_parts.append(crumb)
                _emit_cli_event(
                    on_stream_event,
                    "reasoning_delta",
                    crumb,
                    elapsed_ms,
                    delta_ms,
                    trace_events,
                    event_type,
                )
        else:
            # Claude Code: plain stdout is the answer; status crumbs go to reasoning.
            stdout_parts.append(line)
            _emit_cli_event(on_stream_event, "content_delta", line, elapsed_ms, delta_ms, trace_events, "stdout")

    returncode = proc.wait(timeout=5)
    return (
        "".join(stdout_parts).strip(),
        "".join(reasoning_parts).strip(),
        "".join(stderr_parts),
        returncode,
        trace_events,
    )


def _fake_stream_content(
    text: str,
    on_stream_event: Callable[[dict[str, Any]], None] | None,
    start: float,
    last_event_at: float,
    trace_events: list[dict[str, Any]],
    source: str,
    word_delay_range: tuple[float, float] = WORD_STREAM_DELAY_RANGE,
) -> float:
    """Emit content word-by-word so the UI animates a full Codex agent_message blob."""
    cursor = last_event_at
    lo, hi = word_delay_range
    for chunk in iter_word_chunks(text):
        now = time.perf_counter()
        elapsed_ms = int((now - start) * 1000)
        delta_ms = int((now - cursor) * 1000)
        _emit_cli_event(on_stream_event, "content_delta", chunk, elapsed_ms, delta_ms, trace_events, source)
        cursor = now
        if hi > 0:
            time.sleep(random.uniform(lo, hi) if hi > lo else lo)
    return cursor


def iter_word_chunks(text: str) -> list[str]:
    """Split text into word chunks (spaces preserved on non-final words)."""
    if not text:
        return []
    parts = text.split(" ")
    if len(parts) == 1:
        return [text]
    return [f"{part} " for part in parts[:-1]] + [parts[-1]]


def _emit_cli_status(
    on_status_event: Callable[[dict[str, Any]], None] | None,
    status: str,
    elapsed_ms: int,
    detail: str | None = None,
    *,
    on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    trace_events: list[dict[str, Any]] | None = None,
    start: float | None = None,
    reasoning_parts: list[str] | None = None,
) -> None:
    if on_status_event is not None:
        event: dict[str, Any] = {
            "status": status,
            "elapsed_ms": elapsed_ms,
        }
        if detail:
            event["detail"] = detail
        on_status_event(event)
    # Mirror useful status into the reasoning channel so the UI updates while waiting.
    if on_stream_event is not None and trace_events is not None:
        detail_text = detail
        if detail_text and len(detail_text) > 240:
            detail_text = f"{detail_text[:237]}..."
        crumb = status if not detail_text else f"{status}: {detail_text}"
        line = crumb if crumb.endswith("\n") else f"{crumb}\n"
        if reasoning_parts is not None:
            reasoning_parts.append(line)
        prev_elapsed = int(trace_events[-1].get("elapsed_ms", 0) or 0) if trace_events else 0
        delta_ms = max(0, elapsed_ms - prev_elapsed)
        _emit_cli_event(on_stream_event, "reasoning_delta", line, elapsed_ms, delta_ms, trace_events, "status")


def _emit_cli_event(
    on_stream_event: Callable[[dict[str, Any]], None] | None,
    event_type: str,
    text: str,
    elapsed_ms: int,
    delta_ms: int,
    trace_events: list[dict[str, Any]],
    source: str,
) -> None:
    event = {
        "type": event_type,
        "elapsed_ms": elapsed_ms,
        "delta_ms": delta_ms,
        "chars": len(text),
        "source": source,
    }
    trace_events.append(event)
    if on_stream_event is not None:
        on_stream_event({**event, "text": text})


def _parse_opencode_event(line: str) -> tuple[str, str, str]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return line, "content", "stdout"
    event_type = str(event.get("type") or "")
    part = event.get("part") if isinstance(event.get("part"), dict) else {}
    text = part.get("text") if isinstance(part.get("text"), str) else ""
    if not text:
        for key in ("text", "content"):
            value = event.get(key)
            if isinstance(value, str):
                text = value
                break
    channel = "reasoning" if "reason" in event_type or "think" in event_type else "content"
    return text, channel, event_type


def parse_codex_stream_event(line: str) -> tuple[str, str, str]:
    """Map one Codex JSONL line to (text, channel, event_type).

    Content is reserved for agent/assistant messages. Everything else that is
    useful for the live UI becomes a short reasoning crumb (not raw JSON).
    """
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        stripped = line.strip()
        return (stripped, "content", "stdout") if stripped else ("", "content", "stdout")
    if not isinstance(event, dict):
        return "", "content", "stdout"

    event_type = str(event.get("type") or "")

    if isinstance(event.get("last_agent_message"), str) and event["last_agent_message"]:
        return event["last_agent_message"], "content", "last_agent_message"

    item = event.get("item") if isinstance(event.get("item"), dict) else None
    if item is not None:
        item_type = str(item.get("type") or "")
        item_text = _codex_item_text(item)
        if item_type in _CONTENT_ITEM_TYPES or (
            event_type in {"item.completed", "agent_message", "message"} and item_type in _CONTENT_ITEM_TYPES
        ):
            if item_text:
                return item_text, "content", item_type or event_type
        if item_type in _REASONING_ITEM_TYPES or "reason" in item_type or "think" in item_type:
            if item_text:
                return item_text, "reasoning", item_type or event_type
        crumb = _codex_activity_crumb(event_type, item)
        if crumb:
            return crumb, "reasoning", item_type or event_type
        return "", "reasoning", item_type or event_type

    # Top-level agent_message.message (legacy)
    if event_type in _CONTENT_ITEM_TYPES:
        if isinstance(event.get("message"), str) and event["message"]:
            return event["message"], "content", event_type
        if isinstance(event.get("text"), str) and event["text"]:
            return event["text"], "content", event_type

    if event_type == "error":
        message = event.get("message")
        if isinstance(message, str) and message:
            if message.lower().startswith("reconnecting"):
                return "", "reasoning", event_type
            return message, "reasoning", event_type

    crumb = _codex_activity_crumb(event_type, None)
    if crumb:
        return crumb, "reasoning", event_type or "status"

    if isinstance(event.get("message"), str) and event["message"]:
        channel = "reasoning" if "reason" in event_type or "think" in event_type else "content"
        return event["message"], channel, event_type or "message"
    if isinstance(event.get("text"), str) and event["text"]:
        channel = "reasoning" if "reason" in event_type or "think" in event_type else "content"
        return event["text"], channel, event_type or "message"

    return "", "reasoning", event_type


def _codex_item_text(item: dict[str, Any]) -> str:
    for key in ("text", "content", "message"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _codex_activity_crumb(event_type: str, item: dict[str, Any] | None) -> str:
    """Short human-readable status for turn/tool lifecycle — not raw JSON."""
    if event_type == "thread.started":
        return "thread started"
    if event_type == "turn.started":
        return "turn started"
    if event_type == "turn.completed":
        return "turn completed"
    if event_type == "turn.failed":
        return "turn failed"

    if item is None:
        if event_type:
            return event_type.replace(".", " ")
        return ""

    item_type = str(item.get("type") or "")
    status = str(item.get("status") or "")
    phase = event_type.replace("item.", "").replace(".", " ") or "item"

    if item_type == "command_execution":
        command = str(item.get("command") or "").strip()
        label = status or phase
        if command:
            short = command if len(command) <= 120 else f"{command[:117]}..."
            return f"command {label}: {short}"
        return f"command {label}"

    if item_type == "file_change":
        path = item.get("path") or item.get("file") or ""
        label = status or phase
        if path:
            return f"file change ({label}): {path}"
        return f"file change ({label})"

    if item_type == "todo_list":
        return f"todo list {phase}"

    if item_type and item_type not in _CONTENT_ITEM_TYPES:
        label = f"{item_type} {phase}".strip()
        return label

    if event_type and event_type not in {"item.completed", "agent_message", "message"}:
        return event_type.replace(".", " ")
    return ""


# Back-compat alias used by older call sites / tests.
def _parse_codex_event(line: str) -> tuple[str, str, str]:
    return parse_codex_stream_event(line)


def _command_for_agent(
    agent: CliAgent,
    model: str | None,
    prompt: str,
    reasoning_effort: str | None = None,
) -> list[str]:
    if agent.id == "opencode":
        command = [agent.executable, "run", "--pure", "--format", "json"]
        if model:
            command.extend(["--model", model])
        command.append(prompt)
        return command
    if agent.id == "claude":
        command = [agent.executable, "-p", "--allowedTools", ""]
        if model:
            command.extend(["--model", model])
        command.append(prompt)
        return command
    # Codex CLI — local subscription, no OpenRouter key
    command = [
        agent.executable,
        "exec",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        "never",
        "--json",
    ]
    if model:
        command.extend(["--model", model])
    effort = (reasoning_effort or "").strip().lower()
    if effort and effort not in {"cli", "none"}:
        command.extend(["-c", f"model_reasoning_effort={effort}"])
    command.append(prompt)
    return command


def _content_from_opencode_json(output: str) -> str:
    if not output:
        return output
    parts: list[str] = []
    fallback = ""
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "text":
            part = event.get("part") or {}
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        for key in ("text", "content"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                fallback = value.strip()
        message = event.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                fallback = content.strip()
    return "".join(parts).strip() or fallback or output.strip()
