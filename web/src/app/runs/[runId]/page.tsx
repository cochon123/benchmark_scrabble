import Link from "next/link";
import { notFound } from "next/navigation";

import { BoardView } from "@/components/BoardView";
import { ModelBadge } from "@/components/ModelBadge";
import { formatDate, formatPercent } from "@/lib/format";
import { getDataset, getRun } from "@/lib/store";
import { Position, RunDetail } from "@/lib/types";
import {
  detailCardClass,
  mutedClass,
  pageClass,
  panelClass,
  panelHeaderClass,
  secondaryButtonClass,
  titleClass,
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

  return (
    <div className={pageClass}>
      <section className={panelClass}>
        <div className={panelHeaderClass}>
          <div>
            <ModelBadge companySlug={run.company_slug} modelName={run.model_name} />
            <p className={mutedClass}>
              {run.model_id} · {run.status} · released {formatDate(run.release_date)}
            </p>
          </div>
          <div className="mt-[18px] flex flex-wrap gap-3 sm:mt-0">
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

      {run.board_results.map((result) => {
        const position = positionMap.get(result.position_id);
        if (!position) {
          return null;
        }
        return (
          <section key={result.id} className={`${panelClass} flex flex-col gap-4`}>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-3">
              <div>
                <h2 className={titleClass}>{result.position_id}</h2>
                <p className={mutedClass}>
                  model {result.move_score} / optimal {result.optimal_score} · retry {result.retry_used ? "yes" : "no"}
                </p>
              </div>
              <Link href="/dataset" className={secondaryButtonClass}>
                Compare with dataset
              </Link>
            </div>
            <BoardView
              board={position.board}
              rack={position.rack}
              highlightPlacements={result.parsed_move?.arguments.placements ?? []}
              highlightTone="model"
            />
            <details className={detailCardClass}>
              <summary>Raw response</summary>
              <pre className="mt-2.5 overflow-auto whitespace-pre-wrap break-words [font-family:var(--font-geist-mono)] text-[0.84rem]">
                {result.raw_response}
              </pre>
            </details>
            <details className={detailCardClass}>
              <summary>Attempt trace</summary>
              <pre className="mt-2.5 overflow-auto whitespace-pre-wrap break-words [font-family:var(--font-geist-mono)] text-[0.84rem]">
                {JSON.stringify(result.attempt_trace, null, 2)}
              </pre>
            </details>
          </section>
        );
      })}
    </div>
  );
}
