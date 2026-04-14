import { badgeTheme } from "@/lib/badges";

export function ModelBadge({
  companySlug,
  modelName,
}: {
  companySlug: string;
  modelName: string;
}) {
  const theme = badgeTheme(companySlug);
  return (
    <span className="inline-flex items-center gap-2.5 font-bold">
      {theme.logoSrc ? (
        <span
          className="inline-flex h-[42px] min-w-[42px] items-center justify-center overflow-hidden rounded-[14px] border border-[rgba(22,32,42,0.08)] bg-white p-0"
          style={{ backgroundColor: theme.bg, color: theme.fg }}
          aria-hidden
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={theme.logoSrc} alt="" className="block h-auto max-h-[70%] max-w-[70%] w-auto object-contain" />
        </span>
      ) : (
        <span
          className="inline-flex h-[42px] min-w-[42px] items-center justify-center rounded-[14px] px-2.5 text-[0.85rem]"
          style={{ backgroundColor: theme.bg, color: theme.fg }}
          aria-hidden
        >
          {theme.label}
        </span>
      )}
      <span>{modelName}</span>
    </span>
  );
}
