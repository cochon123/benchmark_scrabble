import { badgeTheme } from "@/lib/badges";
import { formatDate, formatNumber, formatPercent } from "@/lib/format";
import { LeaderboardRow } from "@/lib/types";
import { mutedClass, panelClass, panelHeaderClass, runListClass, secondaryButtonClass, titleClass } from "@/lib/ui";

function numericTicks(max: number, count = 5) {
  return Array.from({ length: count + 1 }, (_, index) => {
    const value = (max * index) / count;
    return { value, ratio: max > 0 ? value / max : 0 };
  });
}

function dateTicks(min: number, max: number, count = 4) {
  return Array.from({ length: count + 1 }, (_, index) => {
    const time = min + ((max - min) * index) / count;
    return { time, ratio: max > min ? (time - min) / (max - min) : 0 };
  });
}

export function LeaderboardBars({ rows }: { rows: LeaderboardRow[] }) {
  const scoreTicks = [0, 25, 50, 75, 100];

  return (
    <div className={panelClass}>
      <div className={panelHeaderClass}>
        <h2 className={titleClass}>Leaderboard</h2>
        <a className={secondaryButtonClass} href="/api/export/all">
          Export CSV
        </a>
      </div>
      <div className="mb-3 grid grid-cols-5 gap-2 pl-1 text-[0.72rem] font-medium text-[color:var(--muted)]">
        {scoreTicks.map((tick) => (
          <span key={tick} className={tick === scoreTicks.at(-1) ? "text-right" : ""}>
            {tick}%
          </span>
        ))}
      </div>
      <div className={runListClass}>
        {rows.map((row) => (
          <div
            key={row.run_id}
            className="grid items-center gap-3 min-[900px]:grid-cols-[minmax(0,280px)_auto_minmax(0,1fr)]"
          >
            <strong className="text-[0.95rem]">{row.model_name}</strong>
            <span
              className="grid h-6 w-6 place-items-center rounded-full border border-[color:var(--line)] text-[0.8rem] font-bold text-[color:var(--muted)]"
              title={`${row.mode} · ${row.status}`}
              aria-label={`${row.mode} · ${row.status}`}
            >
              ?
            </span>
            <div className="overflow-hidden rounded-full bg-[rgba(38,64,78,0.09)]">
              <div
                className="min-w-max rounded-full bg-[color:var(--accent)] px-[14px] py-2 font-bold text-white"
                style={{ width: `${Math.max(row.score_pct, 2)}%` }}
              >
                {formatPercent(row.score_pct)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function TokenRangeChart({ rows }: { rows: LeaderboardRow[] }) {
  const width = 860;
  const height = Math.max(180, rows.length * 36 + 40);
  const maxToken = Math.max(...rows.map((row) => row.max_total_tokens || 1), 1);
  const axisLeft = 210;
  const axisWidth = 610;
  const ticks = numericTicks(maxToken);

  return (
    <div className={panelClass}>
      <h2 className={titleClass}>Token Range</h2>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img">
        {ticks.map((tick) => {
          const x = axisLeft + tick.ratio * axisWidth;
          return (
            <g key={tick.value}>
              <line x1={x} y1="18" x2={x} y2={height - 12} stroke="rgba(22, 32, 42, 0.08)" strokeWidth="1" />
              <text x={x} y="14" textAnchor="middle" className="fill-[color:var(--muted)] text-[11px]">
                {formatNumber(Math.round(tick.value))}
              </text>
            </g>
          );
        })}
        {rows.map((row, index) => {
          const y = 30 + index * 34;
          const minX = axisLeft + (row.min_total_tokens / maxToken) * axisWidth;
          const avgX = axisLeft + (row.avg_total_tokens / maxToken) * axisWidth;
          const maxX = axisLeft + (row.max_total_tokens / maxToken) * axisWidth;
          return (
            <g key={row.run_id}>
              <text x="10" y={y + 4} className="fill-[color:var(--ink)] text-[12px]">
                {row.model_name}
              </text>
              <line x1={minX} y1={y} x2={maxX} y2={y} stroke="#4b6b8f" strokeWidth="4" />
              <circle cx={avgX} cy={y} r="6" fill="#17324d" />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function ScatterChart({ rows }: { rows: LeaderboardRow[] }) {
  const width = 820;
  const height = 340;
  const left = 80;
  const right = 760;
  const top = 40;
  const bottom = 280;
  const maxTokens = Math.max(...rows.map((row) => row.avg_total_tokens || 1), 1);
  const xTicks = numericTicks(maxTokens);
  const yTicks = [0, 25, 50, 75, 100];

  return (
    <div className={panelClass}>
      <h2 className={titleClass}>Average Tokens vs Score</h2>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img">
        {xTicks.map((tick) => {
          const x = left + tick.ratio * (right - left);
          return (
            <g key={tick.value}>
              <line x1={x} y1={top} x2={x} y2={bottom} stroke="rgba(22, 32, 42, 0.08)" strokeWidth="1" />
              <text x={x} y={bottom + 22} textAnchor="middle" className="fill-[color:var(--muted)] text-[11px]">
                {formatNumber(Math.round(tick.value))}
              </text>
            </g>
          );
        })}
        {yTicks.map((tick) => {
          const y = bottom - (tick / 100) * (bottom - top);
          return (
            <g key={tick}>
              <line x1={left} y1={y} x2={right} y2={y} stroke="rgba(22, 32, 42, 0.08)" strokeWidth="1" />
              <text x={left - 10} y={y + 4} textAnchor="end" className="fill-[color:var(--muted)] text-[11px]">
                {tick}%
              </text>
            </g>
          );
        })}
        <line x1={left} y1={bottom} x2={right} y2={bottom} stroke="#7a8e9b" />
        <line x1={left} y1={top} x2={left} y2={bottom} stroke="#7a8e9b" />
        <text x={(left + right) / 2} y={height - 12} textAnchor="middle" className="fill-[color:var(--muted)] text-[12px]">
          Average tokens
        </text>
        <text
          x="18"
          y={(top + bottom) / 2}
          textAnchor="middle"
          transform={`rotate(-90 18 ${(top + bottom) / 2})`}
          className="fill-[color:var(--muted)] text-[12px]"
        >
          Score
        </text>
        {rows.map((row) => {
          const theme = badgeTheme(row.company_slug);
          const x = left + (row.avg_total_tokens / maxTokens) * 650;
          const y = bottom - (row.score_pct / 100) * 220;
          const tooltip = `${row.model_name} | ${formatPercent(row.score_pct)} | avg ${row.avg_total_tokens.toFixed(1)} tokens`;
          return (
            <g key={row.run_id} aria-label={tooltip}>
              <circle cx={x} cy={y} r="16" fill="#ffffff" stroke={theme.bg} strokeWidth="2.5" />
              {theme.logoSrc ? (
                <foreignObject x={x - 10} y={y - 10} width="20" height="20">
                  <div className="flex h-full w-full items-center justify-center">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={theme.logoSrc} alt="" className="block h-auto max-h-full max-w-full w-auto object-contain" />
                  </div>
                </foreignObject>
              ) : (
                <text x={x} y={y + 4} textAnchor="middle" fill={theme.bg} className="text-[10px] font-bold">
                  {theme.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function TimelineChart({ rows }: { rows: LeaderboardRow[] }) {
  const datedRows = rows.filter((row) => row.release_date);
  if (datedRows.length === 0) {
    return (
      <div className={panelClass}>
        <h2 className={titleClass}>Release Timeline</h2>
        <p className={mutedClass}>No release-date metadata available yet.</p>
      </div>
    );
  }
  const width = 820;
  const height = 320;
  const left = 80;
  const right = 760;
  const top = 50;
  const bottom = 260;
  const times = datedRows.map((row) => new Date(row.release_date as string).getTime());
  const min = Math.min(...times);
  const max = Math.max(...times);
  const xTicks = dateTicks(min, max);
  const yTicks = [0, 25, 50, 75, 100];

  return (
    <div className={panelClass}>
      <h2 className={titleClass}>Release Timeline</h2>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img">
        {xTicks.map((tick) => {
          const x = left + tick.ratio * (right - left);
          return (
            <g key={tick.time}>
              <line x1={x} y1={top} x2={x} y2={bottom} stroke="rgba(22, 32, 42, 0.08)" strokeWidth="1" />
              <text x={x} y={bottom + 22} textAnchor="middle" className="fill-[color:var(--muted)] text-[11px]">
                {formatDate(new Date(tick.time).toISOString())}
              </text>
            </g>
          );
        })}
        {yTicks.map((tick) => {
          const y = bottom - (tick / 100) * (bottom - top);
          return (
            <g key={tick}>
              <line x1={left} y1={y} x2={right} y2={y} stroke="rgba(22, 32, 42, 0.08)" strokeWidth="1" />
              <text x={left - 10} y={y + 4} textAnchor="end" className="fill-[color:var(--muted)] text-[11px]">
                {tick}%
              </text>
            </g>
          );
        })}
        <line x1={left} y1={bottom} x2={right} y2={bottom} stroke="#7a8e9b" />
        <line x1={left} y1={top} x2={left} y2={bottom} stroke="#7a8e9b" />
        <text x={(left + right) / 2} y={height - 12} textAnchor="middle" className="fill-[color:var(--muted)] text-[12px]">
          Release date
        </text>
        <text
          x="18"
          y={(top + bottom) / 2}
          textAnchor="middle"
          transform={`rotate(-90 18 ${(top + bottom) / 2})`}
          className="fill-[color:var(--muted)] text-[12px]"
        >
          Score
        </text>
        {datedRows.map((row) => {
          const theme = badgeTheme(row.company_slug);
          const time = new Date(row.release_date as string).getTime();
          const x = left + ((time - min) / Math.max(max - min, 1)) * 650;
          const y = bottom - (row.score_pct / 100) * 190;
          const tooltip = `${row.model_name} | ${formatDate(row.release_date)} | ${formatPercent(row.score_pct)}`;
          return (
            <g key={row.run_id} aria-label={tooltip}>
              <rect x={x - 14} y={y - 14} width="28" height="28" rx="8" fill="#ffffff" stroke={theme.bg} strokeWidth="2" />
              {theme.logoSrc ? (
                <foreignObject x={x - 10} y={y - 10} width="20" height="20">
                  <div className="flex h-full w-full items-center justify-center">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={theme.logoSrc} alt="" className="block h-auto max-h-full max-w-full w-auto object-contain" />
                  </div>
                </foreignObject>
              ) : (
                <text x={x} y={y + 4} textAnchor="middle" fill={theme.bg} className="text-[10px] font-bold">
                  {theme.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
