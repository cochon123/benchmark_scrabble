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
  secondaryButtonClass,
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
  source?: string;
};

type BatchStartResponse = {
  runs: Array<{ id: string; model: string }>;
  errors: Array<{ model: string; error: string }>;
};

export function NewRunConsole() {
  const [query, setQuery] = useState("openai/gpt-4o-mini");
  const deferredQuery = useDeferredValue(query);
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [reasoningEffort, setReasoningEffort] = useState("medium");
  const [preset, setPreset] = useState("smoke");
  const [boards, setBoards] = useState("5");
  const [concurrency, setConcurrency] = useState("4");
  const [runId, setRunId] = useState<string | null>(null);
  const [launchedRuns, setLaunchedRuns] = useState<Array<{ id: string; model: string }>>([]);
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

  function addModel(model: string) {
    const normalized = model.trim();
    if (!normalized) {
      return;
    }
    setSelectedModels((current) => (current.includes(normalized) ? current : [...current, normalized]));
    setQuery("");
    setSuggestions([]);
  }

  async function startRun() {
    const models = selectedModels.length > 0 ? selectedModels : query.trim() ? [query.trim()] : [];
    if (models.length === 0) {
      setError("Add at least one model before starting the benchmark.");
      return;
    }
    setStatus("starting");
    setRunId(null);
    setLaunchedRuns([]);
    setEvents([]);
    setError(null);
    setCurrentTarget(null);
    setStreamedContent("");
    setStreamedReasoning("");

    const response = await fetch("/api/runs/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        models,
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

    const payload = (await response.json()) as BatchStartResponse;
    setLaunchedRuns(payload.runs);
    setRunId(payload.runs.length === 1 ? payload.runs[0].id : null);
    if (payload.errors.length > 0) {
      setError(payload.errors.map((item) => `${item.model}: ${item.error}`).join("\n"));
    }
    setStatus(payload.runs.length === 1 ? "running" : `running ${payload.runs.length} models`);
  }

  return (
    <div className="grid gap-[18px]">
      <div className={panelClass}>
        <h2 className={titleClass}>Launch Benchmark</h2>
        <div className="my-[18px] grid gap-[14px] [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
          <div className={fieldClass}>
            <span>Model</span>
            <div className="flex gap-2">
              <input
                className={inputClass}
                value={query}
                onChange={(event) => {
                  const nextValue = event.target.value;
                  setQuery(nextValue);
                  if (!nextValue.trim()) {
                    setSuggestions([]);
                  }
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    addModel(query);
                  }
                }}
                placeholder="openai/gpt-4o-mini or cli/codex/gpt-5.6-terra"
              />
              <button type="button" className={`${secondaryButtonClass} shrink-0`} onClick={() => addModel(query)}>
                Add
              </button>
            </div>
            {!!suggestions.length && (
              <div className={suggestionListClass}>
                {suggestions.slice(0, 10).map((item) => (
                  <button
                    key={item.slug}
                    type="button"
                    className={suggestionItemClass}
                    onClick={() => {
                      addModel(item.slug);
                    }}
                  >
                    <strong>
                      {item.source === "cli"
                        ? "CLI · "
                        : item.source === "cli2api"
                          ? "cli2api · "
                          : ""}
                      {item.name}
                    </strong>
                    <span>{item.slug}</span>
                  </button>
                ))}
              </div>
            )}
            {selectedModels.length > 0 ? (
              <div className="flex flex-wrap gap-2 pt-1">
                {selectedModels.map((model) => (
                  <button
                    key={model}
                    type="button"
                    className="rounded-full border border-[color:var(--border)] bg-[color:var(--surface-soft)] px-3 py-1.5 text-left text-[0.78rem] font-semibold"
                    title="Remove model"
                    onClick={() => setSelectedModels((current) => current.filter((item) => item !== model))}
                  >
                    {model} ×
                  </button>
                ))}
              </div>
            ) : null}
          </div>

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
        <button type="button" className={primaryButtonClass} onClick={() => void startRun()} disabled={status === "starting"}>
          Start {selectedModels.length > 1 ? `${selectedModels.length} Runs` : "Run"}
        </button>
        <p className={`${mutedClass} mt-3`}>
          Status: <strong>{status}</strong> {runId ? `· run ${runId}` : ""}
        </p>
        {error ? <p className="mt-2.5 font-bold text-[#a02222]">{error}</p> : null}
        {launchedRuns.length > 0 ? (
          <div className="mt-3 flex flex-wrap gap-2">
            {launchedRuns.map((run) => (
              <a key={run.id} href={`/runs/live/${run.id}`} className={secondaryButtonClass}>
                Watch {run.model}
              </a>
            ))}
          </div>
        ) : null}
      </div>

      <div className={panelClass}>
        <h2 className={titleClass}>Live Progress</h2>
        <div className={`${streamGridClass} mt-4`}>
          <div className={streamCardClass}>
            <strong>Current response</strong>
            <p className={mutedClass}>
              {runId ? currentTarget ?? "Waiting for model output." : launchedRuns.length > 1 ? "Open a run above to watch its stream." : "Waiting for model output."}
            </p>
            <AutoScrollPre className={preClass}>
              {streamedContent || "No content streamed yet."}
            </AutoScrollPre>
          </div>
          <div className={streamCardClass}>
            <strong>Current reasoning</strong>
            <p className={mutedClass}>
              Live reasoning from API models, CLI activity (Codex/OpenCode), or gateway streams when available.
            </p>
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
