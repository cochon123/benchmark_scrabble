"use client";

import { useDeferredValue, useEffect, useRef, useState } from "react";

import { RunEvent } from "@/lib/types";
import { AutoScrollPre } from "@/components/AutoScrollPre";
import {
  detailCardClass,
  fieldClass,
  inputClass,
  mutedClass,
  panelClass,
  preClass,
  primaryButtonClass,
  streamCardClass,
  streamGridClass,
  suggestionItemClass,
  suggestionListClass,
  titleClass,
} from "@/lib/ui";

type SearchResult = {
  slug: string;
  name: string;
  author: string;
  created_at?: string;
};

export function NewRunConsole() {
  const [query, setQuery] = useState("openai/gpt-4o-mini");
  const deferredQuery = useDeferredValue(query);
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const [selectedModel, setSelectedModel] = useState("openai/gpt-4o-mini");
  const [reasoningEffort, setReasoningEffort] = useState("medium");
  const [preset, setPreset] = useState("smoke");
  const [boards, setBoards] = useState("5");
  const [concurrency, setConcurrency] = useState("4");
  const [runId, setRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState<string | null>(null);
  const [currentTarget, setCurrentTarget] = useState<string | null>(null);
  const [streamedContent, setStreamedContent] = useState("");
  const [streamedReasoning, setStreamedReasoning] = useState("");
  const statusRef = useRef(status);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(() => {
    const trimmed = deferredQuery.trim();
    if (!trimmed) {
      return;
    }

    const controller = new AbortController();
    fetch(`/api/models/search?q=${encodeURIComponent(trimmed)}`, {
      signal: controller.signal,
      cache: "no-store",
    })
      .then((response) => response.json())
      .then((payload: SearchResult[]) => setSuggestions(payload))
      .catch(() => setSuggestions([]));

    return () => controller.abort();
  }, [deferredQuery]);

  useEffect(() => {
    if (!runId) {
      return;
    }
    let cancelled = false;
    let source: EventSource | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled) {
        return;
      }
      source = new EventSource(`/api/runs/${runId}/stream`);
      source.onmessage = (event) => {
        let payload: RunEvent;
        try {
          payload = JSON.parse(event.data) as RunEvent;
        } catch {
          return;
        }
        if (payload.type === "attempt_started") {
          setCurrentTarget(`${payload.position_id} · attempt ${payload.attempt_index}`);
          setStreamedContent("");
          setStreamedReasoning("");
          setEvents((current) => [...current, payload]);
        }
        if (payload.type === "output_delta") {
          if (payload.channel === "content") {
            setStreamedContent((current) => current + payload.text);
          } else {
            setStreamedReasoning((current) => current + payload.text);
          }
          return;
        }
        if (payload.type === "stream_error") {
          setError(payload.error);
          setEvents((current) => [...current, payload]);
          return;
        }
        if (payload.type === "run_completed") {
          setStatus("completed");
          setEvents((current) => [...current, payload]);
          source?.close();
          return;
        }
        if (payload.type === "run_failed") {
          setStatus("failed");
          setError(payload.error);
          setEvents((current) => [...current, payload]);
          source?.close();
          return;
        }
        setEvents((current) => [...current, payload]);
      };
      source.onerror = () => {
        source?.close();
        if (!cancelled && statusRef.current !== "completed" && statusRef.current !== "failed") {
          reconnectTimer = setTimeout(connect, 1000);
        }
      };
    };

    connect();

    return () => {
      cancelled = true;
      source?.close();
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
    };
  }, [runId]);

  async function startRun() {
    setStatus("starting");
    setRunId(null);
    setEvents([]);
    setError(null);
    setCurrentTarget(null);
    setStreamedContent("");
    setStreamedReasoning("");

    const response = await fetch("/api/runs/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: selectedModel,
        reasoningEffort,
        preset,
        boards: preset === "custom" ? Number(boards) : undefined,
        concurrency: Number(concurrency),
      }),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({ error: "Run start failed." }));
      setStatus("failed");
      setError(payload.error ?? "Run start failed.");
      return;
    }

    const payload = await response.json();
    setRunId(payload.id);
    setStatus("running");
  }

  return (
    <div className="grid gap-[18px]">
      <div className={panelClass}>
        <h2 className={titleClass}>Launch Benchmark</h2>
        <div className="my-[18px] grid gap-[14px] [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
          <label className={fieldClass}>
            <span>Model</span>
            <input
              className={inputClass}
              value={query}
              onChange={(event) => {
                const nextValue = event.target.value;
                setQuery(nextValue);
                setSelectedModel(nextValue);
                if (!nextValue.trim()) {
                  setSuggestions([]);
                }
              }}
              placeholder="openai/gpt-4o-mini"
            />
            {!!suggestions.length && (
              <div className={suggestionListClass}>
                {suggestions.slice(0, 8).map((item) => (
                  <button
                    key={item.slug}
                    type="button"
                    className={suggestionItemClass}
                    onClick={() => {
                      setQuery(item.slug);
                      setSelectedModel(item.slug);
                      setSuggestions([]);
                    }}
                  >
                    <strong>{item.name}</strong>
                    <span>{item.slug}</span>
                  </button>
                ))}
              </div>
            )}
          </label>

          <label className={fieldClass}>
            <span>Reasoning</span>
            <select className={inputClass} value={reasoningEffort} onChange={(event) => setReasoningEffort(event.target.value)}>
              <option value="none">None</option>
              <option value="minimal">Minimal</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="xhigh">Xhigh</option>
            </select>
          </label>

          <label className={fieldClass}>
            <span>Preset</span>
            <select
              className={inputClass}
              value={preset}
              onChange={(event) => {
                setPreset(event.target.value);
                if (event.target.value === "smoke") {
                  setBoards("5");
                }
                if (event.target.value === "full") {
                  setBoards("100");
                }
              }}
            >
              <option value="smoke">Smoke</option>
              <option value="full">Full</option>
              <option value="custom">Custom</option>
            </select>
          </label>

          <label className={fieldClass}>
            <span>Boards</span>
            <input
              className={inputClass}
              value={boards}
              onChange={(event) => setBoards(event.target.value)}
              disabled={preset !== "custom"}
            />
          </label>

          <label className={fieldClass}>
            <span>Parallelism</span>
            <input
              className={inputClass}
              type="number"
              min="1"
              max="32"
              value={concurrency}
              onChange={(event) => setConcurrency(event.target.value)}
            />
          </label>
        </div>
        <button type="button" className={primaryButtonClass} onClick={() => void startRun()}>
          Start Run
        </button>
        <p className={`${mutedClass} mt-3`}>
          Status: <strong>{status}</strong> {runId ? `· run ${runId}` : ""}
        </p>
        {error ? <p className="mt-2.5 font-bold text-[#a02222]">{error}</p> : null}
      </div>

      <div className={panelClass}>
        <h2 className={titleClass}>Live Progress</h2>
        <div className={`${streamGridClass} mt-4`}>
          <div className={streamCardClass}>
            <strong>Current response</strong>
            <p className={mutedClass}>{currentTarget ?? "Waiting for model output."}</p>
            <AutoScrollPre className={preClass}>
              {streamedContent || "No content streamed yet."}
            </AutoScrollPre>
          </div>
          <div className={streamCardClass}>
            <strong>Current reasoning</strong>
            <p className={mutedClass}>Enabled by default when the model supports OpenRouter reasoning.</p>
            <AutoScrollPre className={preClass}>
              {streamedReasoning || "No reasoning stream yet."}
            </AutoScrollPre>
          </div>
        </div>
        <div className="grid gap-3">
          {events.length === 0 ? (
            <p className={mutedClass}>No events yet.</p>
          ) : (
            events.slice(-20).map((event, index) => (
              <div key={`${event.type}-${index}`} className={detailCardClass}>
                <strong>{event.type}</strong>
                <pre className="mt-2.5 overflow-auto whitespace-pre-wrap break-words [font-family:var(--font-geist-mono)] text-[0.84rem]">
                  {JSON.stringify(event, null, 2)}
                </pre>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
