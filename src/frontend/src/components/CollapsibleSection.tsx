import { type ReactNode, useId, useState } from "react";

export type CollapsibleSectionProps = {
  title: string;
  subtitle?: string;
  badge?: string;
  defaultOpen?: boolean;
  children: ReactNode;
  /** e.g. first section open by default for timeline */
  variant?: "default" | "muted";
};

export function CollapsibleSection({
  title,
  subtitle,
  badge,
  defaultOpen = false,
  children,
  variant = "default",
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const panelId = useId();

  const border =
    variant === "muted"
      ? "border-slate-800/80 bg-slate-900/40"
      : "border-slate-800 bg-slate-900/60";

  return (
    <section className={`rounded-2xl border ${border} overflow-hidden`}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start justify-between gap-3 px-5 py-4 text-left transition hover:bg-slate-800/30"
      >
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-2">
            <span className="text-lg font-medium text-slate-200">{title}</span>
            {badge ? (
              <span className="rounded-full border border-slate-600 bg-slate-800/80 px-2 py-0.5 text-xs font-medium text-slate-400">
                {badge}
              </span>
            ) : null}
          </span>
          {subtitle ? <p className="mt-1 text-sm text-slate-500">{subtitle}</p> : null}
        </span>
        <span
          className={`mt-0.5 shrink-0 text-slate-500 transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden
        >
          <ChevronIcon />
        </span>
      </button>
      {open ? (
        <div id={panelId} className="border-t border-slate-800">
          <div className="p-5 pt-4">{children}</div>
        </div>
      ) : null}
    </section>
  );
}

function ChevronIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
