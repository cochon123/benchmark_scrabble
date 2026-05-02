import Link from "next/link";

import { LeaderboardBars, ScatterChart, TimelineChart, TokenRangeChart } from "@/components/Charts";
import { formatPercent } from "@/lib/format";
import { getLeaderboard } from "@/lib/store";
import { LeaderboardRow } from "@/lib/types";
import {
  mutedClass,
  pageClass,
  panelClass,
  primaryButtonClass,
  secondaryButtonClass,
  statCardClass,
  statsGridClass,
} from "@/lib/ui";

export const dynamic = "force-dynamic";

export default function BenchmarkPage() {
  const rows: LeaderboardRow[] = getLeaderboard();
  const completed = rows.filter((row) => row.status === "completed");

  return (
    <div className={pageClass}>
      <section className={`${panelClass} p-[30px]`}>
        <p className="mb-2.5 text-[0.84rem] font-bold uppercase tracking-[0.14em] text-[color:var(--accent)]">
          Benchmark
        </p>
        <h2 className="max-w-[760px] text-[clamp(1.8rem,3vw,3rem)] leading-none tracking-[-0.04em]">
          Rank models by how close they get to the perfect immediate Scrabble move.
        </h2>
        <p className={`mt-[14px] max-w-[720px] leading-[1.6] ${mutedClass}`}>
          Each run sums model points across fixed benchmark boards and divides by the exact solver total.
          The benchmark ignores exchange strategy and leave value on purpose.
        </p>
        <div className="mt-[18px] flex flex-wrap gap-3">
          <Link href="/runs/new" className={primaryButtonClass}>
            Launch a run
          </Link>
          <Link href="/dataset" className={secondaryButtonClass}>
            Inspect dataset
          </Link>
        </div>
      </section>

      <section className={statsGridClass}>
        <div className={statCardClass}>
          <span className={mutedClass}>Total runs</span>
          <strong className="mt-2 block text-[2rem] leading-none">{rows.length}</strong>
        </div>
        <div className={statCardClass}>
          <span className={mutedClass}>Completed runs</span>
          <strong className="mt-2 block text-[2rem] leading-none">{completed.length}</strong>
        </div>
        <div className={statCardClass}>
          <span className={mutedClass}>Best score</span>
          <strong className="mt-2 block text-[2rem] leading-none">
            {completed[0] ? formatPercent(completed[0].score_pct) : "N/A"}
          </strong>
        </div>
      </section>

      <LeaderboardBars rows={rows} />

      <TokenRangeChart rows={completed} />
      <ScatterChart rows={completed} />
      <TimelineChart rows={completed} />
    </div>
  );
}
