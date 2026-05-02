export const pageClass = "grid gap-[18px]";

export const panelClass =
  "rounded-[24px] border border-[color:var(--line)] bg-[color:var(--surface)] p-5 shadow-[var(--shadow)] backdrop-blur-[10px]";

export const panelHeaderClass = "mb-[14px] flex flex-wrap items-start justify-between gap-3";

export const titleClass = "text-[1.25rem] leading-[1.1] tracking-[-0.03em]";

export const mutedClass = "text-[color:var(--muted)]";

export const pillButtonClass =
  "inline-flex items-center justify-center gap-2 rounded-full border px-[1.1rem] py-3 font-bold transition-all duration-150 ease-out hover:-translate-y-[1px] hover:shadow-[0_8px_18px_rgba(22,32,42,0.08)] active:translate-y-0 active:shadow-[0_3px_10px_rgba(22,32,42,0.08)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[color:var(--background)]";

export const secondaryButtonClass =
  `${pillButtonClass} border-[color:var(--line)] bg-[color:var(--surface-subtle)] text-[color:var(--ink)] hover:bg-[color:var(--surface-strong)] active:bg-[color:var(--surface-soft)]`;

export const primaryButtonClass =
  `${pillButtonClass} cursor-pointer border-[color:var(--accent-dark)] bg-[color:var(--accent-dark)] !text-[color:var(--accent-contrast)] hover:bg-[color:var(--accent)] active:bg-[color:var(--accent-dark)]`;

export const statsGridClass = "grid gap-[14px] [grid-template-columns:repeat(auto-fit,minmax(180px,1fr))]";

export const statCardClass = `${panelClass} p-5`;

export const runListClass = "grid gap-3";

export const runListItemClass =
  "flex items-center justify-between gap-3 border-t border-[color:var(--line)] py-[14px] first:border-t-0";

export const fieldClass = "grid gap-2";

export const inputClass =
  "w-full rounded-2xl border border-[color:var(--line)] bg-[color:var(--surface-input)] px-4 py-[0.9rem] text-[inherit] outline-none";

export const suggestionListClass = "mt-1.5 grid gap-1.5";

export const suggestionItemClass =
  "rounded-[14px] border border-[color:var(--line)] bg-[color:var(--surface-soft)] px-3 py-2.5 text-left";

export const streamGridClass = "mb-4 grid gap-3 min-[841px]:grid-cols-2";

export const streamCardClass = "rounded-[18px] border border-[color:var(--line)] bg-[color:var(--surface-soft)] p-4";

export const preClass =
  "mt-2.5 max-h-80 overflow-auto rounded-[14px] border border-[color:var(--line)] bg-[color:var(--surface-pre)] px-[14px] py-3 [font-family:var(--font-geist-mono)] text-[0.84rem] leading-[1.55] whitespace-pre-wrap break-words";

export const detailCardClass = "rounded-2xl border border-[color:var(--line)] bg-[color:var(--surface-subtle)] p-[14px]";

export const shellClass = "mx-auto w-[calc(100%-32px)] max-w-[1380px] pt-7 pb-12 max-[840px]:pt-[18px]";

export const shellHeaderClass = "flex flex-wrap items-end justify-between gap-5 px-1 pt-2 pb-5";

export const eyebrowClass =
  "mb-2.5 text-[0.84rem] font-bold uppercase tracking-[0.14em] text-[color:var(--accent)]";

export const shellTitleClass = "max-w-[720px] text-[clamp(2rem,4vw,3.6rem)] leading-[0.98] tracking-[-0.04em]";

export const navRowClass = "flex flex-wrap gap-2.5";
