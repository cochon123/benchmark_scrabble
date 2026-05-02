import { notFound } from "next/navigation";

import { RunDetailBoardCard } from "@/components/RunDetailBoardCard";
import { ModelBadge } from "@/components/ModelBadge";
import { formatDate, formatPercent } from "@/lib/format";
import { getDataset, getRun } from "@/lib/store";
import { Position, RunDetail } from "@/lib/types";
import {
  mutedClass,
  pageClass,
  panelClass,
  panelHeaderClass,
  primaryButtonClass,
  secondaryButtonClass,
} from "@/lib/ui";

export const dynamic = "force-dynamic";

export default async function RunDetailPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  const run: RunDetail | null = getRun(runId);
  if (!run) {
    notFound();
  }
  const dataset: Position[] = getDataset();
  const positionMap = new Map(dataset.map((position) => [position.id, position]));
  const isActive = run.status === "queued" || run.status === "running";

  return (
    <div className={pageClass}>
      <section className={panelClass}>
        <div className={panelHeaderClass}>
          <div>
            <ModelBadge companySlug={run.company_slug} modelName={run.model_name} reasoningEffort={run.reasoning_effort} />
            <p className={mutedClass}>
              {run.model_id} · {run.status} · released {formatDate(run.release_date)}
            </p>
          </div>
          <div className="mt-[18px] flex flex-wrap gap-3 sm:mt-0">
            {isActive && (
              <a className={primaryButtonClass} href={`/runs/live/${run.id}`}>
                Watch Live
              </a>
            )}
            <a className={secondaryButtonClass} href={`/api/export/run/${run.id}`}>
              Export this run
            </a>
            <a className={secondaryButtonClass} href="/api/export/all">
              Export all
            </a>
          </div>
        </div>
        <p className={mutedClass}>
          Score {formatPercent(run.score_pct)} · {run.raw_points}/{run.optimal_raw_points} points · avg tokens{" "}
          {run.avg_total_tokens.toFixed(1)}
        </p>
      </section>

      <section className="grid items-start gap-4 [grid-template-columns:repeat(auto-fit,minmax(360px,1fr))] max-[840px]:grid-cols-2 max-[640px]:grid-cols-1">
        {run.board_results.map((result) => {
        const position = positionMap.get(result.position_id);
        if (!position) {
          return null;
        }
        return <RunDetailBoardCard key={result.id} position={position} result={result} runId={run.id} runStatus={run.status} />;
      })}
      </section>
    </div>
  );
}
