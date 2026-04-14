import fs from "node:fs";
import path from "node:path";

import Database from "better-sqlite3";

import type { LeaderboardRow, Position, RunDetail, RunBoardResult } from "@/lib/types";
import { repoRoot } from "@/lib/server";

const dbPath = path.join(repoRoot, "db", "benchmark.sqlite");
const datasetPath = path.join(repoRoot, "data", "dataset", "benchmark_positions.json");

let dbSingleton: Database.Database | null = null;
let datasetCache: Position[] | null = null;

function getDb() {
  if (!fs.existsSync(dbPath)) {
    return null;
  }
  if (!dbSingleton) {
    dbSingleton = new Database(dbPath, { readonly: true });
  }
  return dbSingleton;
}

export function getDataset(): Position[] {
  if (datasetCache) {
    return datasetCache;
  }
  if (!fs.existsSync(datasetPath)) {
    return [];
  }
  datasetCache = JSON.parse(fs.readFileSync(datasetPath, "utf-8")) as Position[];
  return datasetCache;
}

export function getLeaderboard(): LeaderboardRow[] {
  const db = getDb();
  if (!db) {
    return [];
  }
  const rows = db
    .prepare(
      `
      SELECT id AS run_id, company_slug, model_id, model_name, release_date,
             score_pct, raw_points, optimal_raw_points,
             avg_total_tokens, min_total_tokens, max_total_tokens,
             status, mode, board_count, started_at
      FROM runs
      ORDER BY score_pct DESC, started_at DESC
      `,
    )
    .all() as LeaderboardRow[];
  return rows;
}

export function getRun(runId: string): RunDetail | null {
  const db = getDb();
  if (!db) {
    return null;
  }
  const run = db.prepare("SELECT * FROM runs WHERE id = ?").get(runId) as RunDetail | undefined;
  if (!run) {
    return null;
  }
  const boardResults = db
    .prepare("SELECT * FROM board_results WHERE run_id = ? ORDER BY id ASC")
    .all(runId) as Array<RunBoardResult & { parsed_move: string | null; attempt_trace: string }>;

  return {
    ...run,
    board_results: boardResults.map((result) => ({
      ...result,
      parsed_move: result.parsed_move ? JSON.parse(result.parsed_move) : null,
      attempt_trace: JSON.parse(result.attempt_trace),
    })),
  };
}

