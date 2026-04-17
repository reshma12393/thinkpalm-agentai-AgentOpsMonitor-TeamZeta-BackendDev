import type { AutomlDebateResponse, AutomlModelRow } from "../api";
import { formatMetrics } from "../lib/formatters";

export type ModelComparisonTableProps = {
  models: AutomlModelRow[];
  metricsByModel: AutomlDebateResponse["metrics"];
  /** Highlights the judge-selected model row */
  winnerModelKey?: string;
  /** Omit outer card when nested inside another section (e.g. collapsible). */
  embedded?: boolean;
};

export function ModelComparisonTable({ models, metricsByModel, winnerModelKey, embedded = false }: ModelComparisonTableProps) {
  const inner = (
    <>
      {!embedded ? (
        <>
          <h2 className="text-lg font-medium text-slate-200">Model comparison</h2>
          <p className="mt-1 text-sm text-slate-500">The judge&apos;s pick is highlighted.</p>
        </>
      ) : null}
      <div className={`overflow-x-auto ${embedded ? "" : "mt-4"}`}>
        <table className="w-full min-w-[640px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-slate-700 text-xs uppercase tracking-wider text-slate-500">
              <th className="pb-3 pr-4 font-semibold">Model</th>
              <th className="pb-3 pr-4 font-semibold">Type</th>
              <th className="pb-3 font-semibold">Holdout metrics</th>
            </tr>
          </thead>
          <tbody>
            {models.map((row) => {
              const isWinner = Boolean(winnerModelKey && row.model_key === winnerModelKey);
              return (
                <tr
                  key={row.model_key}
                  className={`border-b border-slate-800/80 transition last:border-0 ${
                    isWinner
                      ? "bg-emerald-950/35 ring-1 ring-inset ring-emerald-500/45"
                      : "hover:bg-slate-800/20"
                  }`}
                >
                  <td className="py-3 pr-4">
                    <span className="inline-flex flex-wrap items-center gap-2">
                      <span className={`font-mono ${isWinner ? "text-emerald-300" : "text-emerald-400"}`}>
                        {row.model_key}
                      </span>
                      {isWinner ? (
                        <span className="rounded-full bg-emerald-500/25 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-emerald-200 ring-1 ring-emerald-400/40">
                          Winner
                        </span>
                      ) : null}
                    </span>
                  </td>
                  <td className="py-3 pr-4 text-slate-300">{row.model_type}</td>
                  <td className="py-3 font-mono text-xs text-slate-400">{formatMetrics(row.metrics ?? undefined)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-6">
        <h3 className="text-sm font-medium text-slate-400">Raw metrics map (API)</h3>
        <div className="mt-2 max-h-48 overflow-auto rounded-xl border border-slate-800 bg-slate-950/50 p-3">
          <pre className="whitespace-pre-wrap font-mono text-xs text-slate-500">
            {JSON.stringify(metricsByModel, null, 2)}
          </pre>
        </div>
      </div>
    </>
  );

  if (embedded) {
    return <div className="space-y-4">{inner}</div>;
  }

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
      {inner}
    </section>
  );
}
