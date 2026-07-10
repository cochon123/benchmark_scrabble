from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from urllib.error import HTTPError
from typing import Any, Callable

from .config import (
    get_cli2api_base_url,
    get_cli2api_token,
    get_openrouter_api_key,
    get_openrouter_reasoning_effort,
    normalize_reasoning_effort,
)


OPENROUTER_BASE = "https://openrouter.ai"
STALL_WITHOUT_MODEL_TOKENS_SECONDS = 60
_MODEL_SEARCH_CACHE: dict[str, list[dict[str, Any]]] = {}
_MODEL_DETAILS_CACHE: dict[str, dict[str, Any]] = {}


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                elif isinstance(item.get("summary"), str):
                    parts.append(item["summary"])
        return "".join(parts)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), str):
            return value["content"]
        if isinstance(value.get("summary"), str):
            return value["summary"]
    return ""


def _extract_reasoning_details(details: Any) -> str:
    if not isinstance(details, list):
        return ""
    parts: list[str] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary")
        if isinstance(summary, str) and summary:
            parts.append(summary)
        elif isinstance(summary, list):
            for subitem in summary:
                text = _extract_text(subitem)
                if text:
                    parts.append(text)
    return "\n".join(parts)


def _summarize_trace(events: list[dict[str, Any]], latency_ms: int) -> dict[str, Any]:
    reasoning_events = [event for event in events if event.get("type") == "reasoning_delta"]
    content_events = [event for event in events if event.get("type") == "content_delta"]

    def first_elapsed(items: list[dict[str, Any]]) -> int | None:
        return int(items[0]["elapsed_ms"]) if items else None

    def last_elapsed(items: list[dict[str, Any]]) -> int | None:
        return int(items[-1]["elapsed_ms"]) if items else None

    return {
        "latency_ms": latency_ms,
        "reasoning_events": len(reasoning_events),
        "content_events": len(content_events),
        "reasoning_chars": sum(int(event.get("chars", 0) or 0) for event in reasoning_events),
        "content_chars": sum(int(event.get("chars", 0) or 0) for event in content_events),
        "first_reasoning_ms": first_elapsed(reasoning_events),
        "last_reasoning_ms": last_elapsed(reasoning_events),
        "first_content_ms": first_elapsed(content_events),
        "last_content_ms": last_elapsed(content_events),
        "wait_before_reasoning_ms": sum(int(event.get("delta_ms", 0) or 0) for event in reasoning_events),
        "wait_before_content_ms": sum(int(event.get("delta_ms", 0) or 0) for event in content_events),
    }


def _reasoning_config(effort: str | None = None) -> dict[str, Any] | None:
    normalized = normalize_reasoning_effort(effort) if effort is not None else get_openrouter_reasoning_effort()
    if normalized == "none":
        return None
    return {
        "enabled": True,
        "exclude": False,
        "effort": normalized,
    }


def _non_stream_completion(
    *,
    request_payload: dict[str, Any],
    headers: dict[str, str],
    candidate_model: str,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    payload = dict(request_payload)
    payload["stream"] = False
    payload.pop("stream_options", None)
    if on_status is not None:
        on_status(f"retrying without streaming model={candidate_model}")
    start = time.perf_counter()
    response = _request_json(
        f"{OPENROUTER_BASE}/api/v1/chat/completions",
        headers=headers,
        body=payload,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    choices = response.get("choices") or []
    message = choices[0].get("message") if choices else {}
    content = _extract_text(message.get("content")) if isinstance(message, dict) else ""
    reasoning = ""
    if isinstance(message, dict):
        reasoning = _extract_text(message.get("reasoning")) or _extract_reasoning_details(message.get("reasoning_details"))
    events = [
        {"type": "request_sent", "elapsed_ms": 0},
        {"type": "non_stream_response", "elapsed_ms": latency_ms},
    ]
    if reasoning:
        events.append({"type": "reasoning_delta", "elapsed_ms": latency_ms, "delta_ms": latency_ms, "chars": len(reasoning)})
    if content:
        events.append({"type": "content_delta", "elapsed_ms": latency_ms, "delta_ms": latency_ms, "chars": len(content)})
    return {
        "content": content,
        "usage": response.get("usage") or {},
        "latency_ms": latency_ms,
        "provider": response.get("provider"),
        "model": response.get("model", candidate_model),
        "reasoning": reasoning,
        "reasoning_trace": {
            "requested": payload.get("reasoning"),
            "events": events,
            "summary": _summarize_trace(events, latency_ms),
        },
    }


def _request_json(url: str, headers: dict[str, str] | None = None, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def model_search(query: str) -> list[dict[str, Any]]:
    if query in _MODEL_SEARCH_CACHE:
        return _MODEL_SEARCH_CACHE[query]
    url = f"{OPENROUTER_BASE}/api/frontend/models/find?q={urllib.parse.quote(query)}"
    payload = _request_json(url)
    models = payload.get("data", {}).get("models", [])
    _MODEL_SEARCH_CACHE[query] = models
    return models


def all_models() -> list[dict[str, Any]]:
    return _request_json(f"{OPENROUTER_BASE}/api/v1/models").get("data", [])


def normalize_model_for_benchmark(model_id: str, reasoning_effort: str | None = None) -> str:
    if model_id == "demo/mock":
        return model_id
    if normalize_reasoning_effort(reasoning_effort) == "none":
        return _base_model_slug(model_id)
    if ":" in model_id:
        return model_id
    if model_id.startswith("deepseek/"):
        thinking_variant = f"{model_id}:thinking"
        search_hits = model_search(model_id.split("/")[-1])
        if any(item.get("slug") == thinking_variant for item in search_hits):
            return thinking_variant
    return model_id


def _base_model_slug(model_id: str) -> str:
    return model_id.split(":", 1)[0]


def model_supports_reasoning(model_id: str) -> bool:
    slug = _base_model_slug(model_id)
    cached = _MODEL_DETAILS_CACHE.get(slug)
    if cached is not None:
        return bool(cached.get("supports_reasoning", False))

    search_hits = model_search(slug.split("/")[-1])
    for item in search_hits:
        if item.get("slug") == slug:
            supports_reasoning = bool(item.get("supports_reasoning", False))
            _MODEL_DETAILS_CACHE[slug] = {"supports_reasoning": supports_reasoning}
            return supports_reasoning

    for item in all_models():
        if item.get("id") == slug:
            supported = item.get("supported_parameters") or []
            supports_reasoning = "reasoning" in supported or "include_reasoning" in supported
            _MODEL_DETAILS_CACHE[slug] = {"supports_reasoning": supports_reasoning}
            return supports_reasoning
    _MODEL_DETAILS_CACHE[slug] = {"supports_reasoning": False}
    return False


def fetch_model_metadata(model_id: str, reasoning_effort: str | None = None) -> dict[str, Any]:
    if model_id.startswith("cli2api/"):
        local = model_id[len("cli2api/") :]
        return {
            "model_id": model_id,
            "model_name": f"cli2api ({local})",
            "company_slug": "cli2api",
            "release_date": None,
            "supports_reasoning": True,
        }
    effective_model = normalize_model_for_benchmark(model_id, reasoning_effort)
    if effective_model == "demo/mock":
        return {
            "model_id": effective_model,
            "model_name": "Demo Mock",
            "company_slug": "demo",
            "release_date": None,
            "supports_reasoning": False,
        }
    slug = _base_model_slug(effective_model)
    short = model_search(slug.split("/")[-1])
    for item in short:
        if item.get("slug") == slug:
            model_name = item.get("name", effective_model)
            if effective_model.endswith(":thinking") and not model_name.endswith("(thinking)"):
                model_name = f"{model_name} (thinking)"
            return {
                "model_id": effective_model,
                "model_name": model_name,
                "company_slug": item.get("author", "unknown"),
                "release_date": item.get("created_at"),
                "supports_reasoning": bool(item.get("supports_reasoning", False)),
            }
    for item in all_models():
        if item.get("id") == slug:
            created = item.get("created")
            release_date = None
            if created:
                release_date = datetime.fromtimestamp(created, tz=UTC).isoformat()
            model_name = item.get("name", effective_model)
            if effective_model.endswith(":thinking") and not model_name.endswith("(thinking)"):
                model_name = f"{model_name} (thinking)"
            supported = item.get("supported_parameters") or []
            return {
                "model_id": effective_model,
                "model_name": model_name,
                "company_slug": slug.split("/", 1)[0],
                "release_date": release_date,
                "supports_reasoning": "reasoning" in supported or "include_reasoning" in supported,
            }
    return {
        "model_id": effective_model,
        "model_name": effective_model,
        "company_slug": slug.split("/", 1)[0] if "/" in slug else "unknown",
        "release_date": None,
        "supports_reasoning": model_supports_reasoning(effective_model),
    }


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    reasoning_effort: str | None = None,
    on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if model == "demo/mock":
        content = '{"tool":"play_move","arguments":{"placements":[{"row":7,"col":7,"letter":"A"}]}}'
        if on_stream_event is not None:
            for chunk in ('{"tool":"play_move","arguments":{"placements":[', '{"row":7,"col":7,"letter":"A"}]}}'):
                on_stream_event({"type": "content_delta", "text": chunk})
        return {
            "content": content,
            "usage": {"prompt_tokens": 12, "completion_tokens": 17, "total_tokens": 29},
            "latency_ms": 0,
            "provider": "mock",
            "model": model,
            "reasoning": "",
        }

    if model.startswith("cli2api/"):
        return _cli2api_chat_completion(
            model,
            messages,
            on_stream_event=on_stream_event,
            on_status=on_status,
        )

    api_key = get_openrouter_api_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to the root .env file.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Scrabble LLM Benchmark",
    }
    effective_model = normalize_model_for_benchmark(model, reasoning_effort)
    candidate_models = [effective_model]
    if effective_model.endswith(":thinking"):
        candidate_models.append(_base_model_slug(effective_model))

    last_error: Exception | None = None
    for candidate_model in candidate_models:
        reasoning_config = _reasoning_config(reasoning_effort)
        payload: dict[str, Any] = {
            "model": candidate_model,
            "temperature": 0,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if reasoning_config is not None:
            payload["reasoning"] = reasoning_config

        start = time.perf_counter()
        request = urllib.request.Request(
            f"{OPENROUTER_BASE}/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        trace_events: list[dict[str, Any]] = [{"type": "request_sent", "elapsed_ms": 0}]
        last_delta_at = start
        usage: dict[str, Any] = {}
        provider = None

        try:
            if on_status is not None:
                on_status(f"POST {OPENROUTER_BASE}/api/v1/chat/completions model={candidate_model}")
            with urllib.request.urlopen(request, timeout=300) as response:
                if on_status is not None:
                    on_status(f"stream opened model={candidate_model}")
                trace_events.append({"type": "stream_opened", "elapsed_ms": _elapsed_ms(start)})
                seen_bytes = False
                last_model_delta_at: float | None = None
                for raw_line in response:
                    now = time.perf_counter()
                    if not seen_bytes:
                        seen_bytes = True
                        last_model_delta_at = now
                        trace_events.append({"type": "first_stream_bytes", "elapsed_ms": _elapsed_ms(start)})
                        if on_status is not None:
                            on_status(f"first stream bytes model={candidate_model}")
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if (
                        last_model_delta_at is not None
                        and now - last_model_delta_at > STALL_WITHOUT_MODEL_TOKENS_SECONDS
                    ):
                        raise TimeoutError(
                            "OpenRouter stream stalled after opening and did not emit model tokens "
                            f"for {STALL_WITHOUT_MODEL_TOKENS_SECONDS} seconds."
                        )
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        message = chunk["error"].get("message") or "Streaming error from OpenRouter."
                        raise RuntimeError(str(message))
                    provider = chunk.get("provider", provider)
                    usage = chunk.get("usage") or usage
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    text = _extract_text(delta.get("content"))
                    if not text and not delta and isinstance(choices[0].get("message"), dict):
                        text = _extract_text(choices[0]["message"].get("content"))
                    if text:
                        last_model_delta_at = now
                        content_parts.append(text)
                        event = {
                            "type": "content_delta",
                            "elapsed_ms": _elapsed_ms(start),
                            "delta_ms": int((now - last_delta_at) * 1000),
                            "chars": len(text),
                        }
                        trace_events.append(event)
                        last_delta_at = now
                        if on_stream_event is not None:
                            on_stream_event({**event, "text": text})
                    reasoning = _extract_text(delta.get("reasoning")) or _extract_text(delta.get("reasoning_content"))
                    if not reasoning and not delta and isinstance(choices[0].get("message"), dict):
                        message = choices[0]["message"]
                        reasoning = (
                            _extract_text(message.get("reasoning"))
                            or _extract_text(message.get("reasoning_content"))
                            or _extract_reasoning_details(message.get("reasoning_details"))
                        )
                    if reasoning:
                        last_model_delta_at = now
                        reasoning_parts.append(reasoning)
                        event = {
                            "type": "reasoning_delta",
                            "elapsed_ms": _elapsed_ms(start),
                            "delta_ms": int((now - last_delta_at) * 1000),
                            "chars": len(reasoning),
                        }
                        trace_events.append(event)
                        last_delta_at = now
                        if on_stream_event is not None:
                            on_stream_event({**event, "text": reasoning})
                    reasoning_detail_text = _extract_reasoning_details(delta.get("reasoning_details"))
                    if reasoning_detail_text:
                        last_model_delta_at = now
                        reasoning_parts.append(reasoning_detail_text)
                        event = {
                            "type": "reasoning_delta",
                            "elapsed_ms": _elapsed_ms(start),
                            "delta_ms": int((now - last_delta_at) * 1000),
                            "chars": len(reasoning_detail_text),
                        }
                        trace_events.append(event)
                        last_delta_at = now
                        if on_stream_event is not None:
                            on_stream_event({**event, "text": reasoning_detail_text})
        except TimeoutError:
            return _non_stream_completion(
                request_payload=payload,
                headers=headers,
                candidate_model=candidate_model,
                on_status=on_status,
            )
        except KeyboardInterrupt:
            raise
        except HTTPError as exc:
            last_error = exc
            if exc.code == 404 and candidate_model != candidate_models[-1]:
                continue
            raise

        latency_ms = int((time.perf_counter() - start) * 1000)
        trace_events.append({"type": "stream_done", "elapsed_ms": latency_ms})
        return {
            "content": "".join(content_parts),
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": provider,
            "model": candidate_model,
            "reasoning": "".join(reasoning_parts),
            "reasoning_trace": {
                "requested": payload.get("reasoning"),
                "events": trace_events,
                "summary": _summarize_trace(trace_events, latency_ms),
            },
        }

    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenRouter request failed without a response.")


def search_cli2api_models(query: str) -> list[dict[str, Any]]:
    needle = query.strip().lower()
    if not needle:
        return []
    defaults = [
        {"slug": "cli2api/codex/gpt-5.6-terra", "name": "cli2api · Codex · gpt-5.6-terra"},
        {"slug": "cli2api/codex/default", "name": "cli2api · Codex · default"},
    ]
    results = []
    for row in defaults:
        hay = f"{row['slug']} {row['name']}".lower()
        if needle in hay or "cli2api" in needle or needle.startswith("cli2"):
            results.append({**row, "author": "cli2api", "source": "cli2api"})
    return results


def _cli2api_non_stream_completion(
    *,
    url: str,
    request_payload: dict[str, Any],
    headers: dict[str, str],
    api_model: str,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    payload = dict(request_payload)
    payload["stream"] = False
    if on_status is not None:
        on_status(f"retrying without streaming model={api_model}")
    start = time.perf_counter()
    response = _request_json(url, headers=headers, body=payload)
    latency_ms = int((time.perf_counter() - start) * 1000)
    choices = response.get("choices") or []
    message = choices[0].get("message") if choices else {}
    content = _extract_text(message.get("content")) if isinstance(message, dict) else ""
    reasoning = ""
    if isinstance(message, dict):
        reasoning = _extract_text(message.get("reasoning")) or _extract_text(message.get("reasoning_content"))
    events = [
        {"type": "request_sent", "elapsed_ms": 0},
        {"type": "non_stream_response", "elapsed_ms": latency_ms},
    ]
    if reasoning:
        events.append({"type": "reasoning_delta", "elapsed_ms": latency_ms, "delta_ms": latency_ms, "chars": len(reasoning)})
    if content:
        events.append({"type": "content_delta", "elapsed_ms": latency_ms, "delta_ms": latency_ms, "chars": len(content)})
    return {
        "content": content,
        "usage": response.get("usage") or {},
        "latency_ms": latency_ms,
        "provider": "cli2api",
        "model": response.get("model", api_model),
        "reasoning": reasoning,
        "reasoning_trace": {
            "requested": None,
            "events": events,
            "summary": _summarize_trace(events, latency_ms),
        },
    }


def _cli2api_chat_completion(
    model: str,
    messages: list[dict[str, str]],
    on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """OpenAI-compatible chat completions against a local cli2api gateway."""
    base = get_cli2api_base_url()
    token = get_cli2api_token() or "local"
    api_model = model[len("cli2api/") :] if model.startswith("cli2api/") else model
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Scrabble LLM Benchmark",
    }
    payload: dict[str, Any] = {
        "model": api_model,
        "temperature": 0,
        "messages": messages,
        "stream": True,
    }
    url = f"{base}/chat/completions"
    start = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    trace_events: list[dict[str, Any]] = [{"type": "request_sent", "elapsed_ms": 0}]
    last_delta_at = start
    usage: dict[str, Any] = {}

    try:
        if on_status is not None:
            on_status(f"POST {url} model={api_model}")
        with urllib.request.urlopen(request, timeout=300) as response:
            if on_status is not None:
                on_status(f"stream opened model={api_model}")
            trace_events.append({"type": "stream_opened", "elapsed_ms": _elapsed_ms(start)})
            seen_bytes = False
            last_model_delta_at: float | None = None
            for raw_line in response:
                now = time.perf_counter()
                if not seen_bytes:
                    seen_bytes = True
                    last_model_delta_at = now
                    trace_events.append({"type": "first_stream_bytes", "elapsed_ms": _elapsed_ms(start)})
                    if on_status is not None:
                        on_status(f"first stream bytes model={api_model}")
                if (
                    last_model_delta_at is not None
                    and now - last_model_delta_at > STALL_WITHOUT_MODEL_TOKENS_SECONDS
                ):
                    raise TimeoutError(
                        "cli2api stream stalled after opening and did not emit model tokens "
                        f"for {STALL_WITHOUT_MODEL_TOKENS_SECONDS} seconds."
                    )
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                if chunk.get("error"):
                    message = chunk["error"].get("message") if isinstance(chunk["error"], dict) else chunk["error"]
                    raise RuntimeError(str(message or "Streaming error from cli2api."))
                usage = chunk.get("usage") or usage
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = _extract_text(delta.get("content"))
                if text:
                    last_model_delta_at = now
                    content_parts.append(text)
                    event = {
                        "type": "content_delta",
                        "elapsed_ms": _elapsed_ms(start),
                        "delta_ms": int((now - last_delta_at) * 1000),
                        "chars": len(text),
                    }
                    trace_events.append(event)
                    last_delta_at = now
                    if on_stream_event is not None:
                        on_stream_event({**event, "text": text})
                reasoning = _extract_text(delta.get("reasoning")) or _extract_text(delta.get("reasoning_content"))
                if reasoning:
                    last_model_delta_at = now
                    reasoning_parts.append(reasoning)
                    event = {
                        "type": "reasoning_delta",
                        "elapsed_ms": _elapsed_ms(start),
                        "delta_ms": int((now - last_delta_at) * 1000),
                        "chars": len(reasoning),
                    }
                    trace_events.append(event)
                    last_delta_at = now
                    if on_stream_event is not None:
                        on_stream_event({**event, "text": reasoning})
    except TimeoutError:
        return _cli2api_non_stream_completion(
            url=url,
            request_payload=payload,
            headers=headers,
            api_model=api_model,
            on_status=on_status,
        )
    except KeyboardInterrupt:
        raise
    except (HTTPError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        if content_parts or reasoning_parts:
            raise
        if on_status is not None:
            on_status(f"cli2api stream failed ({exc}); retrying without streaming")
        return _cli2api_non_stream_completion(
            url=url,
            request_payload=payload,
            headers=headers,
            api_model=api_model,
            on_status=on_status,
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    trace_events.append({"type": "stream_done", "elapsed_ms": latency_ms})
    return {
        "content": "".join(content_parts),
        "usage": usage,
        "latency_ms": latency_ms,
        "provider": "cli2api",
        "model": api_model,
        "reasoning": "".join(reasoning_parts),
        "reasoning_trace": {
            "requested": None,
            "events": trace_events,
            "summary": _summarize_trace(trace_events, latency_ms),
        },
    }
