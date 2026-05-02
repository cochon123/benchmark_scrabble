"use client";

import { LeaderboardRow, RunDetail } from "@/lib/types";
import { RunLiveView } from "@/components/RunLiveView";

export function RunLive({ run }: { run: RunDetail }) {
  const leaderboardRow: LeaderboardRow = {
    run_id: run.id,
    company_slug: run.company_slug,
    model_id: run.model_id,
    model_name: run.model_name,
    release_date: run.release_date,
    score_pct: run.score_pct,
    raw_points: run.raw_points,
    optimal_raw_points: run.optimal_raw_points,
    avg_total_tokens: run.avg_total_tokens,
    min_total_tokens: run.min_total_tokens,
    max_total_tokens: run.max_total_tokens,
    status: run.status,
    mode: run.mode,
    board_count: run.board_count,
    started_at: run.started_at,
  };
  return <RunLiveView run={leaderboardRow} />;
}
