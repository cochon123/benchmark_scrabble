export type Placement = {
  row: number;
  col: number;
  letter: string;
  is_blank?: boolean;
};

export type Move = {
  placements: Placement[];
  score: number;
  words: string[];
};

export type BoardCell = {
  row: number;
  col: number;
  letter: string;
  is_blank?: boolean;
  is_existing?: boolean;
};

export type Position = {
  id: string;
  band_ply: number;
  source_game_id: string;
  source_seed: number;
  board: BoardCell[];
  rack: string;
  player_to_move: number;
  bag_count: number;
  tiles_played: number;
  optimal_score: number;
  optimal_moves: Move[];
  canonical_optimal_move: Move;
};

export type LeaderboardRow = {
  run_id: string;
  company_slug: string;
  model_id: string;
  model_name: string;
  reasoning_effort: string;
  release_date: string | null;
  score_pct: number;
  raw_points: number;
  optimal_raw_points: number;
  avg_total_tokens: number;
  min_total_tokens: number;
  max_total_tokens: number;
  status: string;
  mode: string;
  board_count: number;
  started_at: string;
};

export type RunBoardResult = {
  id: number;
  position_id: string;
  attempt_index: number;
  raw_response: string;
  parsed_move: { tool: string; arguments: { placements: Placement[] } } | null;
  validation_error: string | null;
  attempt_trace: Array<{
    attempt: number;
    raw_response: string;
    reasoning?: string;
    reasoning_trace?: {
      requested?: { enabled?: boolean; exclude?: boolean; effort?: string };
      events?: Array<{ type: string; elapsed_ms?: number; delta_ms?: number; chars?: number }>;
      summary?: {
        latency_ms?: number;
        reasoning_events?: number;
        content_events?: number;
        reasoning_chars?: number;
        content_chars?: number;
        first_reasoning_ms?: number | null;
        last_reasoning_ms?: number | null;
        first_content_ms?: number | null;
        last_content_ms?: number | null;
        wait_before_reasoning_ms?: number;
        wait_before_content_ms?: number;
      };
    };
    status: string;
    error?: string;
  }>;
  retry_used: number;
  move_score: number;
  optimal_score: number;
  is_optimal: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  latency_ms: number;
  created_at: string;
};

export type RunDetail = {
  id: string;
  model_id: string;
  model_name: string;
  reasoning_effort: string;
  company_slug: string;
  release_date: string | null;
  mode: string;
  board_count: number;
  status: string;
  started_at: string;
  finished_at?: string | null;
  raw_points: number;
  optimal_raw_points: number;
  score_pct: number;
  avg_total_tokens: number;
  min_total_tokens: number;
  max_total_tokens: number;
  error?: string | null;
  board_results: RunBoardResult[];
};

export type RunEvent =
  | { type: "run_started"; run_id: string; model: string; reasoning_effort?: string; board_count: number }
  | {
      type: "attempt_started";
      run_id: string;
      index: number;
      board_count: number;
      position_id: string;
      attempt_index: number;
      model: string;
      reasoning_effort?: string;
    }
  | {
      type: "output_delta";
      run_id: string;
      index: number;
      board_count: number;
      position_id: string;
      attempt_index: number;
      channel: "content" | "reasoning";
      text: string;
      elapsed_ms?: number;
      delta_ms?: number;
      chars?: number;
    }
  | {
      type: "attempt_invalid";
      run_id: string;
      index: number;
      board_count: number;
      position_id: string;
      attempt_index: number;
      error: string;
    }
  | {
      type: "board_result";
      run_id: string;
      index: number;
      board_count: number;
      position_id: string;
      move_score: number;
      optimal_score: number;
      retry_used: boolean;
      validation_error: string | null;
      total_tokens: number;
      latency_ms: number;
    }
  | { type: "run_completed"; run_id: string }
  | { type: "run_failed"; run_id: string; error: string }
  | { type: "stream_error"; error: string };
