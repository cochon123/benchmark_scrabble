import Link from "next/link";

import { ActiveRunsList } from "@/components/ActiveRunsList";
import { ModelBadge } from "@/components/ModelBadge";
import { formatDate, formatPercent } from "@/lib/format";
import { getLeaderboard } from "@/lib/store";
import { LeaderboardRow } from "@/lib/types";
import {
  mutedClass,
  pageClass,
  panelClass,
  panelHeaderClass,
  primaryButtonClass,
  runListClass,
  runListItemClass,
  titleClass,
} from "@/lib/ui";

export const dynamic = "force-dynamic";

export default function RunsPage() {
  const runs: LeaderboardRow[] = getLeaderboard();

  return (
    <div className={pageClass}>
      <section className={panelClass}>
        <div className={panelHeaderClass}>
          <h2 className={titleClass}>Model Runs</h2>
          <Link href="/runs/new" className={primaryButtonClass}>
            New Run
          </Link>
        </div>
      </section>
      <ActiveRunsList />
      <section className={panelClass}>
        <div className={runListClass}>
          {runs.map((run) => (
            <Link key={run.run_id} href={`/runs/${run.run_id}`} className={runListItemClass}>
              <div>
                <ModelBadge companySlug={run.company_slug} modelName={run.model_name} reasoningEffort={run.reasoning_effort} />
                <p className={mutedClass}>
                  {run.mode} · {run.status} · {formatDate(run.release_date)}
                </p>
              </div>
              <strong>{formatPercent(run.score_pct)}</strong>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
