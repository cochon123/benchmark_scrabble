"use client";

import { useEffect, useRef, useState } from "react";

import { LeaderboardRow, RunEvent } from "@/lib/types";
import { formatPercent } from "@/lib/format";
import {
  detailCardClass,
  mutedClass,
  panelClass,
  panelHeaderClass,
  preClass,
  primaryButtonClass,
  secondaryButtonClass,
  streamCardClass,
  streamGridClass,
  titleClass,
} from "@/lib/ui";
import { ModelBadge } from "@/components/ModelBadge";
import { AutoScrollPre } from "@/components/AutoScrollPre";

type BoardProgress = {
  positionId: string;
  attemptIndex: number;
  status: "streaming" | "done" | "invalid";
  moveScore: number;
  optimalScore: number;
};

export function RunLiveView({ run }: { run: LeaderboardRow }) {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [currentTarget, setCurrentTarget] = useState<string | null>(null);
  const [streamedContent, setStreamedContent] = useState("");
  const [streamedReasoning, setStreamedReasoning] = useState("");
  const [streamTiming, setStreamTiming] = useState({ reasoningChars: 0, contentChars: 0, lastReasoningMs: null as number | null, lastContentMs: null as number | null });
  const [boards, setBoards] = useState<BoardProgress[]>([]);
  const [runStatus, setRunStatus] = useState(run.status);
  const [rawPoints, setRawPoints] = useState(run.raw_points);
  const [optimalPoints, setOptimalPoints] = useState(run.optimal_raw_points);
  const [error, setError] = useState<string | null>(null);
  const statusRef = useRef(runStatus);

  useEffect(() => {
    statusRef.current = runStatus;
  }, [runStatus]);

  useEffect(() => {
    let cancelled = false;
    let source: EventSource | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled) return;
      source = new EventSource(`/api/runs/${run.run_id}/stream`);
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
          setStreamTiming({ reasoningChars: 0, contentChars: 0, lastReasoningMs: null, lastContentMs: null });
          setBoards((prev) => {
            const existing = prev.find((b) => b.positionId === payload.position_id);
            if (existing) {
              return prev.map((b) =>
                b.positionId === payload.position_id
                  ? { ...b, attemptIndex: payload.attempt_index, status: "streaming" as const }
                  : b,
              );
            }
            return [
              ...prev,
              {
                positionId: payload.position_id,
                attemptIndex: payload.attempt_index,
                status: "streaming" as const,
                moveScore: 0,
                optimalScore: 0,
              },
            ];
          });
          setEvents((prev) => [...prev, payload]);
        }
        if (payload.type === "output_delta") {
          if (payload.channel === "content") {
            setStreamedContent((prev) => prev + payload.text);
            setStreamTiming((prev) => ({
              ...prev,
              contentChars: prev.contentChars + (payload.chars ?? payload.text.length),
              lastContentMs: payload.elapsed_ms ?? prev.lastContentMs,
            }));
          } else {
            setStreamedReasoning((prev) => prev + payload.text);
            setStreamTiming((prev) => ({
              ...prev,
              reasoningChars: prev.reasoningChars + (payload.chars ?? payload.text.length),
              lastReasoningMs: payload.elapsed_ms ?? prev.lastReasoningMs,
            }));
          }
          return;
        }
        if (payload.type === "stream_error") {
          setError(payload.error);
          setEvents((prev) => [...prev, payload]);
          return;
        }
        if (payload.type === "board_result") {
          setBoards((prev) =>
            prev.map((b) =>
              b.positionId === payload.position_id
                ? {
                    ...b,
                    status: "done" as const,
                    moveScore: payload.move_score,
                    optimalScore: payload.optimal_score,
                  }
                : b,
            ),
          );
          setRawPoints((prev) => prev + payload.move_score);
          setOptimalPoints((prev) => prev + payload.optimal_score);
          setEvents((prev) => [...prev, payload]);
          return;
        }
        if (payload.type === "attempt_invalid") {
          setBoards((prev) =>
            prev.map((b) =>
              b.positionId === payload.position_id
                ? { ...b, status: "invalid" as const }
                : b,
            ),
          );
          setEvents((prev) => [...prev, payload]);
          return;
        }
        if (payload.type === "run_completed") {
          setRunStatus("completed");
          setEvents((prev) => [...prev, payload]);
          source?.close();
          return;
        }
        if (payload.type === "run_failed") {
          setRunStatus("failed");
          setError(payload.error);
          setEvents((prev) => [...prev, payload]);
          source?.close();
          return;
        }
        setEvents((prev) => [...prev, payload]);
      };
      source.onerror = () => {
        source?.close();
        if (!cancelled && statusRef.current !== "completed" && statusRef.current !== "failed") {
          reconnectTimer = setTimeout(connect, 2000);
        }
      };
    };

    if (run.status === "queued" || run.status === "running") {
      connect();
    }

    return () => {
      cancelled = true;
      source?.close();
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };
  }, [run.run_id, run.status]);

  const completedBoards = boards.filter((b) => b.status === "done").length;
  const progressPct = run.board_count > 0 ? (completedBoards / run.board_count) * 100 : 0;
  const scorePct = optimalPoints > 0 ? (100.0 * rawPoints) / optimalPoints : run.score_pct;

  async function cancelRun() {
    await fetch(`/api/runs/${run.run_id}/cancel`, { method: "POST" });
    setRunStatus("cancelled");
  }

  return (
    <div className="grid gap-[18px]">
      <section className={panelClass}>
        <div className={panelHeaderClass}>
          <div>
            <ModelBadge companySlug={run.company_slug} modelName={run.model_name} reasoningEffort={run.reasoning_effort} />
            <p className={mutedClass}>
              {run.model_id} · {runStatus} · {run.board_count} boards
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            {(runStatus === "queued" || runStatus === "running") && (
              <button type="button" className={secondaryButtonClass} onClick={() => void cancelRun()}>
                Cancel
              </button>
            )}
          </div>
        </div>
        <div className="mt-3 grid gap-2">
          <div className="flex items-center justify-between">
            <span className={mutedClass}>
              Progress: {completedBoards}/{run.board_count} boards
            </span>
            <strong>{formatPercent(scorePct)}</strong>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-full bg-[color:var(--surface-soft)]">
            <div
              className="h-full rounded-full bg-[color:var(--accent)] transition-all duration-300"
              style={{ width: `${Math.min(progressPct, 100)}%` }}
            />
          </div>
          <p className={mutedClass}>
            {rawPoints}/{optimalPoints} points
          </p>
        </div>
        {error && <p className="mt-2.5 font-bold text-[#a02222]">{error}</p>}
      </section>

      <section className={panelClass}>
        <h2 className={titleClass}>Board Results</h2>
        <div className="mt-4 grid gap-2 max-h-[320px] overflow-y-auto">
          {boards.length === 0 ? (
            <p className={mutedClass}>Waiting for first board...</p>
          ) : (
            boards.map((board) => (
              <div
                key={board.positionId}
                className="flex items-center justify-between rounded-xl border border-[color:var(--line)] bg-[color:var(--surface-subtle)] px-4 py-2.5"
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`inline-block h-2.5 w-2.5 rounded-full ${
                      board.status === "done"
                        ? "bg-green-500"
                        : board.status === "invalid"
                          ? "bg-red-400"
                          : "bg-amber-400 animate-pulse"
                    }`}
                  />
                  <span className="font-medium">{board.positionId}</span>
                  {board.attemptIndex > 0 && (
                    <span className={mutedClass}>attempt {board.attemptIndex}</span>
                  )}
                </div>
                <div className="flex items-center gap-4">
                  {board.status === "done" && (
                    <>
                      <span className={mutedClass}>
                        {board.moveScore}/{board.optimalScore}
                      </span>
                      <strong>{formatPercent(board.optimalScore > 0 ? (100 * board.moveScore) / board.optimalScore : 0)}</strong>
                    </>
                  )}
                  {board.status === "invalid" && <span className="text-red-400">invalid</span>}
                  {board.status === "streaming" && <span className={mutedClass}>streaming...</span>}
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      <section className={panelClass}>
        <h2 className={titleClass}>Live Stream</h2>
        <div className={`${streamGridClass} mt-4`}>
          <div className={streamCardClass}>
            <strong>Current response</strong>
            <p className={mutedClass}>
              {currentTarget ?? "Waiting for model output."}
              {streamTiming.lastContentMs !== null ? ` · ${streamTiming.contentChars} chars by ${(streamTiming.lastContentMs / 1000).toFixed(1)}s` : ""}
            </p>
            <AutoScrollPre className={preClass}>{streamedContent || "No content streamed yet."}</AutoScrollPre>
          </div>
          <div className={streamCardClass}>
            <strong>Current reasoning</strong>
            <p className={mutedClass}>
              Reasoning stream when supported by the model.
              {streamTiming.lastReasoningMs !== null ? ` ${streamTiming.reasoningChars} chars by ${(streamTiming.lastReasoningMs / 1000).toFixed(1)}s.` : ""}
            </p>
            <AutoScrollPre className={preClass}>{streamedReasoning || "No reasoning stream yet."}</AutoScrollPre>
          </div>
        </div>
        <div className="mt-4 grid gap-3">
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
      </section>

      {runStatus === "completed" && (
        <a href={`/runs/${run.run_id}`} className={primaryButtonClass + " text-center"}>
          View Full Results
        </a>
      )}
    </div>
  );
}
