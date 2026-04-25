"use client";

import { useState } from "react";

import { badgeTheme } from "@/lib/badges";
import { formatDate, formatNumber, formatPercent } from "@/lib/format";
import { LeaderboardRow } from "@/lib/types";
import { mutedClass, panelClass, panelHeaderClass, runListClass, secondaryButtonClass, titleClass } from "@/lib/ui";

type TooltipState = {
  x: number;
  y: number;
  title: string;
  lines: string[];
};

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

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function tooltipPlacement(
  x: number,
  y: number,
  width: number,
  height: number,
  tooltipWidth: number,
  tooltipHeight: number,
) {
  const padding = 12;
  const preferredX = x + 16;
  const preferredY = y - tooltipHeight - 16;
  const nextX = preferredX + tooltipWidth > width - padding ? x - tooltipWidth - 16 : preferredX;
  const nextY = preferredY < padding ? y + 16 : preferredY;
  return {
    x: clamp(nextX, padding, width - tooltipWidth - padding),
    y: clamp(nextY, padding, height - tooltipHeight - padding),
  };
}

function ChartTooltip({
  tooltip,
  width,
  height,
}: {
  tooltip: TooltipState | null;
  width: number;
  height: number;
}) {
  if (!tooltip) {
    return null;
  }

  const boxWidth = 238;
  const lineCount = tooltip.lines.length;
  const boxHeight = 54 + lineCount * 16;
  const position = tooltipPlacement(tooltip.x, tooltip.y, width, height, boxWidth, boxHeight);

  return (
    <foreignObject x={position.x} y={position.y} width={boxWidth} height={boxHeight} pointerEvents="none">
      <div className="h-full w-full">
        <div className="flex h-full flex-col gap-1 rounded-2xl border border-[color:var(--line)] bg-[color:var(--surface-strong)]/95 px-3 py-2 text-[color:var(--ink)] shadow-[var(--shadow)] backdrop-blur-sm">
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-[color:var(--accent)]" />
            <p className="text-[0.86rem] font-semibold leading-5">{tooltip.title}</p>
          </div>
          <div className="grid gap-0.5">
            {tooltip.lines.map((line) => (
              <p key={line} className="text-[0.78rem] leading-5 text-[color:var(--muted)]">
                {line}
              </p>
            ))}
          </div>
        </div>
      </div>
    </foreignObject>
  );
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
            <div className="overflow-hidden rounded-full bg-[color:var(--chart-grid)]">
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
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
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
              <line x1={x} y1="18" x2={x} y2={height - 12} stroke="var(--chart-grid)" strokeWidth="1" />
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
          const lines = [
            `Score ${formatPercent(row.score_pct)}`,
            `Tokens min ${formatNumber(row.min_total_tokens)} · avg ${formatNumber(row.avg_total_tokens)} · max ${formatNumber(row.max_total_tokens)}`,
            row.release_date ? `Released ${formatDate(row.release_date)}` : null,
          ].filter((line): line is string => line !== null);
          return (
            <g
              key={row.run_id}
              onPointerEnter={() => setTooltip({ x: avgX, y, title: row.model_name, lines })}
              onPointerLeave={() => setTooltip(null)}
            >
              <text x="10" y={y + 4} className="fill-[color:var(--ink)] text-[12px]">
                {row.model_name}
              </text>
              <line x1={minX} y1={y} x2={maxX} y2={y} stroke="var(--chart-series)" strokeWidth="4" />
              <circle cx={avgX} cy={y} r="6" fill="var(--chart-point)" />
            </g>
          );
        })}
        <ChartTooltip tooltip={tooltip} width={width} height={height} />
      </svg>
    </div>
  );
}

export function ScatterChart({ rows }: { rows: LeaderboardRow[] }) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
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
              <line x1={x} y1={top} x2={x} y2={bottom} stroke="var(--chart-grid)" strokeWidth="1" />
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
              <line x1={left} y1={y} x2={right} y2={y} stroke="var(--chart-grid)" strokeWidth="1" />
              <text x={left - 10} y={y + 4} textAnchor="end" className="fill-[color:var(--muted)] text-[11px]">
                {tick}%
              </text>
            </g>
          );
        })}
        <line x1={left} y1={bottom} x2={right} y2={bottom} stroke="var(--chart-axis)" />
        <line x1={left} y1={top} x2={left} y2={bottom} stroke="var(--chart-axis)" />
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
          const lines = [
            `Score ${formatPercent(row.score_pct)}`,
            `Avg tokens ${formatNumber(row.avg_total_tokens)}`,
            row.release_date ? `Released ${formatDate(row.release_date)}` : `Company ${row.company_slug}`,
          ];
          return (
            <g
              key={row.run_id}
              aria-label={`${row.model_name} | ${formatPercent(row.score_pct)} | avg ${row.avg_total_tokens.toFixed(1)} tokens`}
              onPointerEnter={() => setTooltip({ x, y, title: row.model_name, lines })}
              onPointerLeave={() => setTooltip(null)}
            >
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
        <ChartTooltip tooltip={tooltip} width={width} height={height} />
      </svg>
    </div>
  );
}

export function TimelineChart({ rows }: { rows: LeaderboardRow[] }) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
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
        {xTicks.map((tick, i) => {
          const x = left + tick.ratio * (right - left);
          return (
            <g key={i}>
              <line x1={x} y1={top} x2={x} y2={bottom} stroke="var(--chart-grid)" strokeWidth="1" />
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
              <line x1={left} y1={y} x2={right} y2={y} stroke="var(--chart-grid)" strokeWidth="1" />
              <text x={left - 10} y={y + 4} textAnchor="end" className="fill-[color:var(--muted)] text-[11px]">
                {tick}%
              </text>
            </g>
          );
        })}
        <line x1={left} y1={bottom} x2={right} y2={bottom} stroke="var(--chart-axis)" />
        <line x1={left} y1={top} x2={left} y2={bottom} stroke="var(--chart-axis)" />
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
          const lines = [
            `Release ${formatDate(row.release_date)}`,
            `Score ${formatPercent(row.score_pct)}`,
            `Avg tokens ${formatNumber(row.avg_total_tokens)}`,
          ];
          return (
            <g
              key={row.run_id}
              aria-label={`${row.model_name} | ${formatDate(row.release_date)} | ${formatPercent(row.score_pct)}`}
              onPointerEnter={() => setTooltip({ x, y, title: row.model_name, lines })}
              onPointerLeave={() => setTooltip(null)}
            >
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
        <ChartTooltip tooltip={tooltip} width={width} height={height} />
      </svg>
    </div>
  );
}
