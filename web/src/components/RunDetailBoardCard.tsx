"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { BoardView } from "@/components/BoardView";
import { mutedClass, preClass, primaryButtonClass, secondaryButtonClass, titleClass } from "@/lib/ui";
import { Position, RunBoardResult } from "@/lib/types";

function formatMs(value?: number | null) {
  if (value === undefined || value === null) {
    return "n/a";
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}s`;
  }
  return `${value}ms`;
}

function traceSummary(attempt: RunBoardResult["attempt_trace"][number]) {
  const summary = attempt.reasoning_trace?.summary;
  if (!summary) {
    return "No timing trace.";
  }
  const requested = attempt.reasoning_trace?.requested?.effort;
  return [
    `requested ${requested ? `${requested} reasoning` : "reasoning"}`,
    `latency ${formatMs(summary.latency_ms)}`,
    `first reasoning ${formatMs(summary.first_reasoning_ms)}`,
    `first content ${formatMs(summary.first_content_ms)}`,
    `reasoning chars ${summary.reasoning_chars ?? 0}`,
    `content chars ${summary.content_chars ?? 0}`,
    `wait before reasoning ${formatMs(summary.wait_before_reasoning_ms)}`,
    `wait before content ${formatMs(summary.wait_before_content_ms)}`,
  ].join(" · ");
}

export function RunDetailBoardCard({
  position,
  result,
  runId,
  runStatus,
}: {
  position: Position;
  result: RunBoardResult;
  runId: string;
  runStatus: string;
}) {
  const router = useRouter();
  const [compareToBest, setCompareToBest] = useState(false);
  const [isStopping, setIsStopping] = useState(false);

  async function stopRun() {
    setIsStopping(true);
    try {
      const response = await fetch(`/api/runs/${runId}/cancel`, { method: "POST" });
      if (!response.ok) {
        throw new Error("Unable to stop run.");
      }
      router.refresh();
    } finally {
      setIsStopping(false);
    }
  }

  return (
    <section className="flex w-full max-w-[420px] flex-col gap-4 rounded-[24px] border border-[color:var(--line)] bg-[color:var(--surface)] p-5 shadow-[var(--shadow)] backdrop-blur-[10px]">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-3">
        <div>
          <h2 className={titleClass}>{result.position_id}</h2>
          <p className={mutedClass}>
            model {result.move_score} / optimal {result.optimal_score} · retry {result.retry_used ? "yes" : "no"}
          </p>
        </div>
        <button type="button" className={secondaryButtonClass} onClick={() => setCompareToBest((value) => !value)}>
          {compareToBest ? "Hide best" : "Show best"}
        </button>
      </div>
      {runStatus === "running" ? (
        <button type="button" className={primaryButtonClass} onClick={() => void stopRun()} disabled={isStopping}>
          {isStopping ? "Stopping..." : "Stop run"}
        </button>
      ) : null}
      <BoardView
        board={position.board}
        rack={position.rack}
        highlightPlacements={compareToBest ? position.canonical_optimal_move.placements : result.parsed_move?.arguments.placements ?? []}
        highlightTone={compareToBest ? "optimal" : "model"}
      />
      <details className="rounded-[14px] border border-[color:var(--line)] bg-[color:var(--surface-subtle)] p-3">
        <summary className="cursor-pointer font-bold">Attempt trace</summary>
        <div className="mt-3 grid gap-3">
          {result.attempt_trace.map((attempt) => (
            <div key={attempt.attempt} className="grid gap-2 border-t border-[color:var(--line)] pt-3 first:border-t-0 first:pt-0">
              <div>
                <strong>
                  attempt {attempt.attempt} · {attempt.status}
                </strong>
                {attempt.error ? <p className="text-[#a02222]">{attempt.error}</p> : null}
                <p className={mutedClass}>{traceSummary(attempt)}</p>
              </div>
              {attempt.reasoning ? (
                <details>
                  <summary className="cursor-pointer text-sm font-bold">Reasoning text</summary>
                  <pre className={preClass}>{attempt.reasoning}</pre>
                </details>
              ) : (
                <p className={mutedClass}>No reasoning text captured.</p>
              )}
              {attempt.reasoning_trace?.events?.length ? (
                <details>
                  <summary className="cursor-pointer text-sm font-bold">Timing events</summary>
                  <pre className={preClass}>{JSON.stringify(attempt.reasoning_trace.events, null, 2)}</pre>
                </details>
              ) : null}
            </div>
          ))}
        </div>
      </details>
    </section>
  );
}
