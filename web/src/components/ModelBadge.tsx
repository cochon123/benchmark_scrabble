import { badgeTheme } from "@/lib/badges";

export function ModelBadge({
  companySlug,
  modelName,
  reasoningEffort,
}: {
  companySlug: string;
  modelName: string;
  reasoningEffort?: string | null;
}) {
  const theme = badgeTheme(companySlug);
  const displayName = modelName.replace(/^[^:]+:\s*/, "");
  return (
    <span className="inline-flex min-w-0 items-center gap-2.5 font-bold">
      {theme.logoSrc ? (
        <span
          className="inline-flex h-[42px] w-[42px] shrink-0 items-center justify-center overflow-hidden rounded-[14px] border border-[color:var(--line)] bg-white p-0"
          aria-hidden
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={theme.logoSrc} alt="" className="block h-auto max-h-[70%] max-w-[70%] w-auto object-contain" />
        </span>
      ) : (
        <span
          className="inline-flex h-[42px] min-w-[42px] shrink-0 items-center justify-center rounded-[14px] px-2.5 text-[0.85rem]"
          style={{ backgroundColor: theme.bg, color: theme.fg }}
          aria-hidden
        >
          {theme.label}
        </span>
      )}
      <span className="line-clamp-2 overflow-hidden break-words text-ellipsis">{displayName}</span>
      {reasoningEffort ? (
        <span className="rounded-full border border-[color:var(--line)] bg-[color:var(--surface-soft)] px-2 py-0.5 text-[0.72rem] uppercase tracking-[0.08em] text-[color:var(--muted)]">
          {reasoningEffort}
        </span>
      ) : null}
    </span>
  );
}
