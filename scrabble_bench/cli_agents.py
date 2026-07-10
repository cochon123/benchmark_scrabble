from __future__ import annotations

import json
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


CLI_TIMEOUT_SECONDS = 300


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
    return []


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
    _emit_cli_status(on_status_event, f"starting {agent.name}", 0)
    if on_status is not None:
        on_status(f"running {agent.name}")
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="scrabble-bench-cli-") as workdir:
        content, stderr_text, returncode, trace_events = _run_streaming_process(
            command,
            workdir,
            agent,
            on_stream_event,
            on_status_event,
            start,
        )
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
        "reasoning": "",
        "reasoning_trace": {
            "events": trace_events or [{"type": "cli_response", "elapsed_ms": latency_ms, "chars": len(content)}],
            "summary": {"latency_ms": latency_ms, "content_chars": len(content)},
        },
    }


def _run_streaming_process(
    command: list[str],
    workdir: str,
    agent: CliAgent,
    on_stream_event: Callable[[dict[str, Any]], None] | None,
    on_status_event: Callable[[dict[str, Any]], None] | None,
    start: float,
) -> tuple[str, str, int, list[dict[str, Any]]]:
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
    stderr_parts: list[str] = []
    trace_events: list[dict[str, Any]] = []

    def reader(name: str, stream: Any) -> None:
        try:
            for line in stream:
                events.put((name, line))
        finally:
            events.put((name, ""))

    assert proc.stdout is not None and proc.stderr is not None
    _emit_cli_status(on_status_event, f"{agent.name} process started", int((time.perf_counter() - start) * 1000))
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
            _emit_cli_status(on_status_event, f"{agent.name} timed out after {CLI_TIMEOUT_SECONDS}s", elapsed_ms)
            raise TimeoutError(f"{agent.name} did not finish within {CLI_TIMEOUT_SECONDS} seconds.")
        try:
            stream_name, line = events.get(timeout=0.2)
        except queue.Empty:
            if time.monotonic() - last_status_at >= 2:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                _emit_cli_status(on_status_event, f"waiting for {agent.name} output", elapsed_ms)
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
            _emit_cli_status(on_status_event, f"{agent.name} stderr", elapsed_ms, detail=line.strip())
            continue
        if agent.id == "opencode":
            text, channel, event_type = _parse_opencode_event(line)
            if text:
                if channel == "content":
                    stdout_parts.append(text)
                _emit_cli_event(on_stream_event, f"{channel}_delta", text, elapsed_ms, delta_ms, trace_events, event_type)
            elif event_type:
                _emit_cli_status(on_status_event, f"{agent.name} {event_type}", elapsed_ms)
        elif agent.id == "codex":
            text, channel, event_type = _parse_codex_event(line)
            if text:
                if channel == "content":
                    # Codex JSONL emits the full agent message when complete —
                    # keep only the latest complete message.
                    if event_type in {"agent_message", "item.completed", "message"}:
                        stdout_parts.clear()
                    stdout_parts.append(text)
                _emit_cli_event(on_stream_event, f"{channel}_delta", text, elapsed_ms, delta_ms, trace_events, event_type)
            elif event_type:
                _emit_cli_status(on_status_event, f"{agent.name} {event_type}", elapsed_ms)
        else:
            stdout_parts.append(line)
            _emit_cli_event(on_stream_event, "content_delta", line, elapsed_ms, delta_ms, trace_events, "stdout")

    returncode = proc.wait(timeout=5)
    return "".join(stdout_parts).strip(), "".join(stderr_parts), returncode, trace_events


def _emit_cli_status(
    on_status_event: Callable[[dict[str, Any]], None] | None,
    status: str,
    elapsed_ms: int,
    detail: str | None = None,
) -> None:
    if on_status_event is None:
        return
    event: dict[str, Any] = {
        "status": status,
        "elapsed_ms": elapsed_ms,
    }
    if detail:
        event["detail"] = detail
    on_status_event(event)


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


def _parse_codex_event(line: str) -> tuple[str, str, str]:
    """Extract assistant text from Codex exec --json JSONL events."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return line, "content", "stdout"
    if not isinstance(event, dict):
        return "", "content", "stdout"

    event_type = str(event.get("type") or "")
    text = ""

    if isinstance(event.get("last_agent_message"), str):
        text = event["last_agent_message"]
        return text, "content", "agent_message"

    item = event.get("item") if isinstance(event.get("item"), dict) else None
    if item:
        item_type = str(item.get("type") or "")
        if isinstance(item.get("text"), str):
            text = item["text"]
        elif isinstance(item.get("content"), str):
            text = item["content"]
        if text:
            channel = "reasoning" if "reason" in item_type or "think" in item_type else "content"
            return text, channel, item_type or event_type

    if isinstance(event.get("message"), str):
        text = event["message"]
    elif isinstance(event.get("text"), str):
        text = event["text"]

    if not text:
        return "", "content", event_type

    channel = "reasoning" if "reason" in event_type or "think" in event_type else "content"
    return text, channel, event_type or "message"


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
