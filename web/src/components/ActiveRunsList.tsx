"use client";

import { useCallback, useEffect, useState } from "react";

import { LeaderboardRow } from "@/lib/types";
import { ModelBadge } from "@/components/ModelBadge";
import { formatDate } from "@/lib/format";
import {
  mutedClass,
  panelClass,
  primaryButtonClass,
  runListClass,
  runListItemClass,
  titleClass,
} from "@/lib/ui";

export function ActiveRunsList() {
  const [runs, setRuns] = useState<LeaderboardRow[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchActive = useCallback(async () => {
    try {
      const response = await fetch("/api/runs/active", { cache: "no-store" });
      const data = (await response.json()) as LeaderboardRow[];
      setRuns(data);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchActive();
    const interval = setInterval(fetchActive, 3000);
    return () => clearInterval(interval);
  }, [fetchActive]);

  if (loading) {
    return null;
  }

  if (runs.length === 0) {
    return null;
  }

  return (
    <section className={panelClass}>
      <h2 className={titleClass}>Active Runs</h2>
      <div className={`${runListClass} mt-3`}>
        {runs.map((run) => (
          <a
            key={run.run_id}
            href={`/runs/live/${run.run_id}`}
            className={`${runListItemClass} group cursor-pointer hover:bg-[color:var(--surface-soft)] -mx-2 rounded-xl px-2 transition-colors`}
          >
            <div>
              <ModelBadge companySlug={run.company_slug} modelName={run.model_name} reasoningEffort={run.reasoning_effort} />
              <p className={mutedClass}>
                {run.mode} · <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
                  {run.status}
                </span> · {run.board_count} boards · {formatDate(run.release_date)}
              </p>
            </div>
            <span className={`${primaryButtonClass} !py-2 !text-[0.84rem]`}>
              Watch
            </span>
          </a>
        ))}
      </div>
    </section>
  );
}
