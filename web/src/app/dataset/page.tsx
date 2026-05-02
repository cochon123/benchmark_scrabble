import Link from "next/link";

import { BoardView } from "@/components/BoardView";
import { getDataset } from "@/lib/store";
import { Position } from "@/lib/types";
import { mutedClass, pageClass, panelClass, titleClass } from "@/lib/ui";

export const dynamic = "force-dynamic";

const PAGE_SIZE = 12;

function clampPage(value: string | undefined, totalPages: number) {
  const parsed = Number.parseInt(value ?? "1", 10);
  if (!Number.isFinite(parsed) || parsed < 1) {
    return 1;
  }
  return Math.min(parsed, totalPages);
}

export default async function DatasetPage({
  searchParams,
}: {
  searchParams: Promise<{ page?: string }>;
}) {
  const dataset: Position[] = getDataset();
  const totalPages = Math.max(1, Math.ceil(dataset.length / PAGE_SIZE));
  const params = await searchParams;
  const page = clampPage(params.page, totalPages);
  const start = (page - 1) * PAGE_SIZE;
  const visiblePositions = dataset.slice(start, start + PAGE_SIZE);
  const firstPosition = dataset.length === 0 ? 0 : start + 1;
  const lastPosition = Math.min(start + visiblePositions.length, dataset.length);

  function pageHref(targetPage: number) {
    return targetPage === 1 ? "/dataset" : `/dataset?page=${targetPage}`;
  }

  return (
    <div className={pageClass}>
      <section className={panelClass}>
        <h2 className={titleClass}>Dataset</h2>
        <p className={mutedClass}>
          Existing tiles are grey. Each position shows the canonical optimal move in green.
        </p>
        <p className={`${mutedClass} mt-4 text-sm`}>
          Showing {firstPosition}-{lastPosition} of {dataset.length} positions.
        </p>
      </section>

      <section className="grid items-start gap-4 [grid-template-columns:repeat(auto-fit,minmax(360px,1fr))] max-[840px]:grid-cols-2 max-[640px]:grid-cols-1">
        {visiblePositions.map((position) => (
          <section key={position.id} className={`${panelClass} flex flex-col gap-4`}>
            <div className="flex flex-col gap-1 sm:flex-row sm:items-start sm:justify-between sm:gap-3">
              <h2 className={titleClass}>{position.id}</h2>
              <span className={`${mutedClass} sm:pt-0.5`}>
                {position.band_ply} plies · optimal {position.optimal_score}
              </span>
            </div>
            <BoardView
              board={position.board}
              rack={position.rack}
              highlightPlacements={position.canonical_optimal_move.placements}
              highlightTone="optimal"
            />
            <p className={`${mutedClass} pt-1`}>
              rack {position.rack} · tied moves {position.optimal_moves.length}
            </p>
          </section>
        ))}
      </section>

      {totalPages > 1 ? (
        <nav className="flex flex-wrap items-center justify-center gap-2" aria-label="Dataset pages">
          {Array.from({ length: totalPages }, (_, index) => {
            const pageNumber = index + 1;
            const isCurrent = pageNumber === page;
            return (
              <Link
                key={pageNumber}
                href={pageHref(pageNumber)}
                aria-current={isCurrent ? "page" : undefined}
                className={`inline-flex min-w-[2.75rem] items-center justify-center rounded-full border px-4 py-2 text-sm font-semibold ${
                  isCurrent
                    ? "border-[color:var(--accent-dark)] bg-[color:var(--accent-dark)] text-white"
                    : "border-[color:var(--line)] bg-[color:var(--surface-subtle)] text-[color:var(--ink)]"
                }`}
              >
                {pageNumber}
              </Link>
            );
          })}
        </nav>
      ) : null}
    </div>
  );
}
