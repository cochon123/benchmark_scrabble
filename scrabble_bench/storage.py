from __future__ import annotations

import csv
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .config import DB_PATH, EXPORT_DIR, LOG_DIR, ensure_directories


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_directories()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def init_db() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY,
              model_id TEXT NOT NULL,
              model_name TEXT NOT NULL,
              company_slug TEXT NOT NULL,
              release_date TEXT,
              reasoning_effort TEXT NOT NULL DEFAULT 'medium',
              mode TEXT NOT NULL,
              board_count INTEGER NOT NULL,
              status TEXT NOT NULL,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              raw_points INTEGER NOT NULL DEFAULT 0,
              optimal_raw_points INTEGER NOT NULL DEFAULT 0,
              score_pct REAL NOT NULL DEFAULT 0,
              avg_total_tokens REAL NOT NULL DEFAULT 0,
              min_total_tokens INTEGER NOT NULL DEFAULT 0,
              max_total_tokens INTEGER NOT NULL DEFAULT 0,
              error TEXT
            );

            CREATE TABLE IF NOT EXISTS board_results (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              position_id TEXT NOT NULL,
              attempt_index INTEGER NOT NULL,
              raw_response TEXT NOT NULL,
              parsed_move TEXT,
              validation_error TEXT,
              attempt_trace TEXT NOT NULL,
              retry_used INTEGER NOT NULL,
              move_score INTEGER NOT NULL,
              optimal_score INTEGER NOT NULL,
              is_optimal INTEGER NOT NULL,
              prompt_tokens INTEGER NOT NULL DEFAULT 0,
              completion_tokens INTEGER NOT NULL DEFAULT 0,
              total_tokens INTEGER NOT NULL DEFAULT 0,
              latency_ms INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL
            );
            """
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
        if "reasoning_effort" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN reasoning_effort TEXT NOT NULL DEFAULT 'high'")
        connection.commit()


def active_run() -> dict[str, Any] | None:
    init_db()
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM runs WHERE status IN ('queued', 'running') ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def create_run(
    *,
    model_id: str,
    model_name: str,
    company_slug: str,
    release_date: str | None,
    reasoning_effort: str,
    mode: str,
    board_count: int,
) -> dict[str, Any]:
    init_db()
    if active_run():
        raise RuntimeError("Another benchmark run is already active.")
    run_id = uuid.uuid4().hex
    row = {
        "id": run_id,
        "model_id": model_id,
        "model_name": model_name,
        "company_slug": company_slug,
        "release_date": release_date,
        "reasoning_effort": reasoning_effort,
        "mode": mode,
        "board_count": board_count,
        "status": "queued",
        "started_at": utc_now(),
    }
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO runs (
              id, model_id, model_name, company_slug, release_date, reasoning_effort,
              mode, board_count, status, started_at
            )
            VALUES (
              :id, :model_id, :model_name, :company_slug, :release_date, :reasoning_effort,
              :mode, :board_count, :status, :started_at
            )
            """,
            row,
        )
        connection.commit()
    return row


def update_run_status(run_id: str, status: str, error: str | None = None) -> None:
    init_db()
    fields = {"status": status, "id": run_id}
    query = "UPDATE runs SET status = :status"
    if status in {"completed", "failed", "cancelled"}:
        query += ", finished_at = :finished_at"
        fields["finished_at"] = utc_now()
    if error is not None:
        query += ", error = :error"
        fields["error"] = error
    query += " WHERE id = :id"
    with connect() as connection:
        connection.execute(query, fields)
        connection.commit()


def should_stop(run_id: str) -> bool:
    init_db()
    with connect() as connection:
        row = connection.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
    return bool(row and row["status"] == "cancelled")


def append_log(run_id: str, payload: dict[str, Any]) -> None:
    ensure_directories()
    log_path = LOG_DIR / f"run_{run_id}.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def record_board_result(run_id: str, result: dict[str, Any]) -> None:
    init_db()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO board_results (
              run_id, position_id, attempt_index, raw_response, parsed_move, validation_error,
              attempt_trace, retry_used, move_score, optimal_score, is_optimal,
              prompt_tokens, completion_tokens, total_tokens, latency_ms, created_at
            )
            VALUES (
              :run_id, :position_id, :attempt_index, :raw_response, :parsed_move, :validation_error,
              :attempt_trace, :retry_used, :move_score, :optimal_score, :is_optimal,
              :prompt_tokens, :completion_tokens, :total_tokens, :latency_ms, :created_at
            )
            """,
            {
                "run_id": run_id,
                "position_id": result["position_id"],
                "attempt_index": result["attempt_index"],
                "raw_response": result["raw_response"],
                "parsed_move": json.dumps(result.get("parsed_move")) if result.get("parsed_move") else None,
                "validation_error": result.get("validation_error"),
                "attempt_trace": json.dumps(result["attempt_trace"]),
                "retry_used": int(result["retry_used"]),
                "move_score": result["move_score"],
                "optimal_score": result["optimal_score"],
                "is_optimal": int(result["is_optimal"]),
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "latency_ms": result["latency_ms"],
                "created_at": utc_now(),
            },
        )
        connection.commit()
    refresh_run_aggregate(run_id)


def refresh_run_aggregate(run_id: str) -> None:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT move_score, optimal_score, total_tokens
            FROM board_results
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
        if not rows:
            return
        raw_points = sum(row["move_score"] for row in rows)
        optimal_points = sum(row["optimal_score"] for row in rows)
        token_values = [row["total_tokens"] for row in rows]
        score_pct = (100.0 * raw_points / optimal_points) if optimal_points else 0.0
        connection.execute(
            """
            UPDATE runs
            SET raw_points = ?, optimal_raw_points = ?, score_pct = ?,
                avg_total_tokens = ?, min_total_tokens = ?, max_total_tokens = ?
            WHERE id = ?
            """,
            (
                raw_points,
                optimal_points,
                score_pct,
                sum(token_values) / len(token_values),
                min(token_values),
                max(token_values),
                run_id,
            ),
        )
        connection.commit()


def list_runs() -> list[dict[str, Any]]:
    init_db()
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM runs ORDER BY started_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_run(run_id: str) -> dict[str, Any] | None:
    init_db()
    with connect() as connection:
        run = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            return None
        results = connection.execute(
            "SELECT * FROM board_results WHERE run_id = ? ORDER BY id ASC", (run_id,)
        ).fetchall()
    payload = dict(run)
    payload["board_results"] = []
    for row in results:
        item = dict(row)
        item["parsed_move"] = json.loads(item["parsed_move"]) if item["parsed_move"] else None
        item["attempt_trace"] = json.loads(item["attempt_trace"])
        payload["board_results"].append(item)
    return payload


def leaderboard() -> list[dict[str, Any]]:
    init_db()
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id AS run_id, company_slug, model_id, model_name, release_date,
                   score_pct, raw_points, optimal_raw_points,
                   avg_total_tokens, min_total_tokens, max_total_tokens,
                   status, mode, board_count, started_at, reasoning_effort
            FROM runs
            ORDER BY score_pct DESC, started_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def export_all_csv(path: Path) -> Path:
    rows = leaderboard()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "company",
                "model",
                "score_pct",
                "raw_points",
                "optimal_raw_points",
                "release_date",
                "avg_total_tokens",
                "min_total_tokens",
                "max_total_tokens",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "company": row["company_slug"],
                    "model": row["model_name"],
                    "score_pct": row["score_pct"],
                    "raw_points": row["raw_points"],
                    "optimal_raw_points": row["optimal_raw_points"],
                    "release_date": row["release_date"],
                    "avg_total_tokens": row["avg_total_tokens"],
                    "min_total_tokens": row["min_total_tokens"],
                    "max_total_tokens": row["max_total_tokens"],
                }
            )
    return path


def export_run_csv(run_id: str, path: Path) -> Path:
    payload = get_run(run_id)
    if payload is None:
        raise RuntimeError(f"Unknown run id: {run_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "position_id",
                "move_score",
                "optimal_score",
                "is_optimal",
                "retry_used",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "latency_ms",
                "validation_error",
            ],
        )
        writer.writeheader()
        for result in payload["board_results"]:
            writer.writerow(
                {
                    "position_id": result["position_id"],
                    "move_score": result["move_score"],
                    "optimal_score": result["optimal_score"],
                    "is_optimal": result["is_optimal"],
                    "retry_used": result["retry_used"],
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                    "latency_ms": result["latency_ms"],
                    "validation_error": result["validation_error"],
                }
            )
    return path


def default_export_path(kind: str, run_id: str | None = None) -> Path:
    ensure_directories()
    if kind == "all":
        return EXPORT_DIR / "benchmark_results.csv"
    if kind == "run" and run_id:
        return EXPORT_DIR / f"run_{run_id}.csv"
    raise ValueError("Unknown export kind")
