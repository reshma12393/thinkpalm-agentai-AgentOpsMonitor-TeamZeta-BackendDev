import { useMemo, useState } from "react";
import type { AutomlDebateResponse, AutomlModelRow } from "../api";

export type ModelComparisonTableProps = {
  models: AutomlModelRow[];
  metricsByModel: AutomlDebateResponse["metrics"];
  /** Highlights the judge-selected model row */
  winnerModelKey?: string;
  /** Omit outer card when nested inside another section (e.g. collapsible). */
  embedded?: boolean;
};

/**
 * Metric names that are better when larger (classification-style scores).
 * Anything outside this set is treated as lower-is-better (e.g. mae, rmse, log_loss).
 */
const HIGHER_IS_BETTER = new Set([
  "accuracy",
  "balanced_accuracy",
  "f1",
  "f1_macro",
  "f1_micro",
  "f1_weighted",
  "precision",
  "precision_macro",
  "precision_weighted",
  "recall",
  "recall_macro",
  "recall_weighted",
  "roc_auc",
  "roc_auc_ovr",
  "roc_auc_ovo",
  "pr_auc",
  "r2",
]);

function coerceNumber(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) return Number(v);
  return null;
}

function isHigherBetter(name: string): boolean {
  return HIGHER_IS_BETTER.has(name.toLowerCase());
}

function formatCell(v: number | string | undefined | null): string {
  if (v == null) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "—";
    const abs = Math.abs(v);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return v.toExponential(3);
    return v.toFixed(4);
  }
  return String(v);
}

export function ModelComparisonTable({
  models,
  metricsByModel,
  winnerModelKey,
  embedded = false,
}: ModelComparisonTableProps) {
  const [copied, setCopied] = useState(false);

  /** Union of all metric names across rows, preserving first-seen order. */
  const metricColumns = useMemo(() => {
    const ordered: string[] = [];
    const seen = new Set<string>();
    for (const row of models) {
      for (const k of Object.keys(row.metrics ?? {})) {
        if (!seen.has(k)) {
          seen.add(k);
          ordered.push(k);
        }
      }
    }
    for (const bag of Object.values(metricsByModel ?? {})) {
      if (bag && typeof bag === "object") {
        for (const k of Object.keys(bag as Record<string, unknown>)) {
          if (!seen.has(k)) {
            seen.add(k);
            ordered.push(k);
          }
        }
      }
    }
    return ordered;
  }, [models, metricsByModel]);

  /** Per-metric best value → highlight. */
  const bestByMetric = useMemo(() => {
    const result: Record<string, { key: string; value: number } | null> = {};
    for (const m of metricColumns) {
      const higher = isHigherBetter(m);
      let best: { key: string; value: number } | null = null;
      for (const row of models) {
        const raw = (row.metrics ?? {})[m];
        const num = coerceNumber(raw);
        if (num == null) continue;
        if (!best) {
          best = { key: row.model_key, value: num };
        } else if (higher ? num > best.value : num < best.value) {
          best = { key: row.model_key, value: num };
        }
      }
      result[m] = best;
    }
    return result;
  }, [models, metricColumns]);

  const downloadJson = () => {
    try {
      const blob = new Blob([JSON.stringify(metricsByModel ?? {}, null, 2)], {
        type: "application/json;charset=utf-8",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "automl_metrics.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch {
      /* no-op: clipboard still available */
    }
  };

  const copyJson = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(metricsByModel ?? {}, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  const inner = (
    <>
      {!embedded ? (
        <>
          <h2 className="text-lg font-medium text-slate-200">Model comparison</h2>
          <p className="mt-1 text-sm text-slate-500">Holdout metrics — judge&apos;s pick is highlighted.</p>
        </>
      ) : null}
      <div className={`overflow-x-auto rounded-xl border border-slate-800/80 bg-slate-950/40 ${embedded ? "" : "mt-4"}`}>
        <table className="w-full min-w-[640px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-900/60 text-xs uppercase tracking-wider text-slate-500">
              <th className="px-4 py-3 font-semibold">Model</th>
              <th className="px-4 py-3 font-semibold">Type</th>
              <th className="px-4 py-3 font-semibold">Agent</th>
              {metricColumns.map((m) => (
                <th key={m} className="px-4 py-3 text-right font-semibold" title={isHigherBetter(m) ? "Higher is better" : "Lower is better"}>
                  <span className="inline-flex items-center justify-end gap-1">
                    <span>{m}</span>
                    <span className="text-[10px] text-slate-600" aria-hidden>
                      {isHigherBetter(m) ? "↑" : "↓"}
                    </span>
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.length === 0 ? (
              <tr>
                <td colSpan={3 + metricColumns.length} className="px-4 py-6 text-center text-sm text-slate-500">
                  No model runs to compare.
                </td>
              </tr>
            ) : (
              models.map((row) => {
                const isWinner = Boolean(winnerModelKey && row.model_key === winnerModelKey);
                const metrics = row.metrics ?? {};
                return (
                  <tr
                    key={row.model_key}
                    className={`border-b border-slate-800/70 transition last:border-0 ${
                      isWinner ? "bg-emerald-950/35 ring-1 ring-inset ring-emerald-500/45" : "hover:bg-slate-800/25"
                    }`}
                  >
                    <td className="px-4 py-3">
                      <span className="inline-flex flex-wrap items-center gap-2">
                        <span className={`font-mono font-semibold ${isWinner ? "text-emerald-200" : "text-emerald-300/95"}`}>
                          {row.model_key}
                        </span>
                        {isWinner ? (
                          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/25 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-emerald-100 ring-1 ring-emerald-400/40">
                            <TrophyIcon className="h-3 w-3" />
                            Winner
                          </span>
                        ) : null}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate-300">{row.model_type || "—"}</td>
                    <td className="px-4 py-3 text-slate-400">{row.agent || "—"}</td>
                    {metricColumns.map((m) => {
                      const raw = metrics[m];
                      const num = coerceNumber(raw);
                      const best = bestByMetric[m];
                      const isBest = best != null && best.key === row.model_key && num != null;
                      return (
                        <td
                          key={m}
                          className={`px-4 py-3 text-right font-mono tabular-nums ${
                            isBest ? "font-semibold text-amber-200" : "text-slate-300"
                          }`}
                          title={isBest ? "Best on this metric" : undefined}
                        >
                          {formatCell(raw ?? null)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500/55" aria-hidden /> Winner row
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-amber-400/70" aria-hidden /> Best per metric
          </span>
          <span>↑ higher-is-better · ↓ lower-is-better</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={downloadJson}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900/80 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-emerald-500/50 hover:bg-emerald-500/10 hover:text-emerald-200 focus:outline-none focus:ring-2 focus:ring-emerald-500/35"
            title="Download raw metrics map as JSON"
          >
            <DownloadIcon className="h-3.5 w-3.5" />
            Download JSON
          </button>
          <button
            type="button"
            onClick={() => void copyJson()}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900/80 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:border-slate-500 hover:bg-slate-800/80 focus:outline-none focus:ring-2 focus:ring-slate-500/40"
            title="Copy raw metrics JSON to clipboard"
          >
            <CopyIcon className="h-3.5 w-3.5" />
            {copied ? "Copied" : "Copy JSON"}
          </button>
        </div>
      </div>
    </>
  );

  if (embedded) {
    return <div className="space-y-4">{inner}</div>;
  }

  return <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">{inner}</section>;
}

function TrophyIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M8 21h8m-4-4v4m-5-9a4 4 0 004 4h2a4 4 0 004-4V4H7v8zm0 0H5a2 2 0 01-2-2V7a1 1 0 011-1h3m10 6h2a2 2 0 002-2V7a1 1 0 00-1-1h-3" />
    </svg>
  );
}

function DownloadIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v12m0 0l-4-4m4 4l4-4M4 20h16" />
    </svg>
  );
}

function CopyIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
    </svg>
  );
}
