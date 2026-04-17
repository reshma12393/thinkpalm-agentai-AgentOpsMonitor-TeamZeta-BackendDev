import type { AutomlDebateResponse } from "../api";
import { CollapsibleSection } from "./CollapsibleSection";

type Deterministic = Record<string, unknown>;

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (Number.isFinite(v)) return Math.abs(v) >= 1000 ? v.toLocaleString() : String(v);
    return String(v);
  }
  if (typeof v === "boolean") return v ? "Yes" : "No";
  return String(v);
}

function pctFromFraction(v: unknown): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(2)}%`;
}

export type EdaSummaryTablesProps = {
  eda: AutomlDebateResponse["eda"];
};

/**
 * Renders deterministic EDA as readable tables; safe when `deterministic` is missing.
 */
export function EdaSummaryTables({ eda }: EdaSummaryTablesProps) {
  const det = eda?.deterministic as Deterministic | undefined;
  if (!det || typeof det !== "object") {
    return (
      <p className="text-sm text-slate-500">
        No deterministic EDA block in the response. Expand <strong>Full structured EDA (JSON)</strong> below if raw
        data is present.
      </p>
    );
  }

  const tp = (det.target_profile as Deterministic) || {};
  const mv = (det.missing_values as Deterministic) || {};
  const imb = det.class_imbalance as Deterministic | null | undefined;
  const corr = (det.correlations as Deterministic) || {};
  const featureTypes = (det.feature_types as Record<string, Deterministic>) || {};
  const byCol = (mv.by_column as Record<string, number>) || {};
  const pairs = (corr.top_feature_pairs as Array<Record<string, unknown>>) || [];

  const overviewRows: { label: string; value: string }[] = [
    { label: "Rows", value: fmt(det.n_rows) },
    { label: "Features", value: fmt(det.n_features) },
    { label: "Target column", value: fmt(det.target_column) },
    { label: "Task (inferred)", value: fmt(tp.task_hint) },
  ];
  if (tp.n_classes != null) overviewRows.push({ label: "Classes", value: fmt(tp.n_classes) });
  overviewRows.push(
    { label: "Target missing (count)", value: fmt(tp.missing_count) },
    { label: "Target missing (fraction)", value: pctFromFraction(tp.missing_fraction) },
    { label: "Rows with any feature missing", value: fmt(mv.rows_with_any_feature_missing) },
    { label: "Fraction of rows with any missing feature", value: pctFromFraction(mv.fraction_rows_any_missing) },
  );
  if (imb && typeof imb === "object" && imb.imbalance_ratio != null) {
    const r = imb.imbalance_ratio;
    overviewRows.push({
      label: "Class imbalance ratio",
      value: typeof r === "number" && !Number.isFinite(r) ? "∞" : fmt(r),
    });
  }

  const featureRows = Object.keys(featureTypes)
    .sort((a, b) => a.localeCompare(b))
    .map((name) => {
      const ft = featureTypes[name] || {};
      const miss = byCol[name];
      return {
        name,
        role: fmt(ft.role),
        dtype: fmt(ft.dtype),
        nUnique: fmt(ft.n_unique),
        missing: typeof miss === "number" ? pctFromFraction(miss) : "—",
      };
    });

  return (
    <div className="space-y-3">
      <CollapsibleSection title="Dataset overview" defaultOpen variant="muted">
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full min-w-[320px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-slate-700 bg-slate-950/50 text-xs uppercase tracking-wider text-slate-500">
                <th className="px-4 py-2.5 font-semibold">Metric</th>
                <th className="px-4 py-2.5 font-semibold">Value</th>
              </tr>
            </thead>
            <tbody>
              {overviewRows.map((row) => (
                <tr key={row.label} className="border-b border-slate-800/80 last:border-0">
                  <td className="px-4 py-2.5 text-slate-400">{row.label}</td>
                  <td className="px-4 py-2.5 font-mono text-slate-200">{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CollapsibleSection>

      {featureRows.length > 0 ? (
        <CollapsibleSection title="Features" subtitle={`${featureRows.length} columns`} defaultOpen variant="muted">
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full min-w-[560px] border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-slate-700 bg-slate-950/50 text-xs uppercase tracking-wider text-slate-500">
                  <th className="px-3 py-2.5 font-semibold">Feature</th>
                  <th className="px-3 py-2.5 font-semibold">Role</th>
                  <th className="px-3 py-2.5 font-semibold">Dtype</th>
                  <th className="px-3 py-2.5 font-semibold">Unique</th>
                  <th className="px-3 py-2.5 font-semibold">Missing</th>
                </tr>
              </thead>
              <tbody>
                {featureRows.map((row) => (
                  <tr key={row.name} className="border-b border-slate-800/80 last:border-0 hover:bg-slate-800/15">
                    <td className="px-3 py-2 font-mono text-xs text-emerald-400/95">{row.name}</td>
                    <td className="px-3 py-2 text-slate-300">{row.role}</td>
                    <td className="px-3 py-2 font-mono text-xs text-slate-500">{row.dtype}</td>
                    <td className="px-3 py-2 font-mono text-xs text-slate-400">{row.nUnique}</td>
                    <td className="px-3 py-2 font-mono text-xs text-slate-400">{row.missing}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CollapsibleSection>
      ) : null}

      {pairs.length > 0 ? (
        <CollapsibleSection
          title="Feature–feature correlations"
          subtitle="Pearson · strongest pairs"
          defaultOpen
          variant="muted"
        >
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full min-w-[400px] border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-slate-700 bg-slate-950/50 text-xs uppercase tracking-wider text-slate-500">
                  <th className="px-3 py-2.5 font-semibold">Feature A</th>
                  <th className="px-3 py-2.5 font-semibold">Feature B</th>
                  <th className="px-3 py-2.5 font-semibold">r</th>
                </tr>
              </thead>
              <tbody>
                {pairs.slice(0, 10).map((p, i) => (
                  <tr key={`${p.feature_a}-${p.feature_b}-${i}`} className="border-b border-slate-800/80 last:border-0">
                    <td className="px-3 py-2 font-mono text-xs text-slate-300">{fmt(p.feature_a)}</td>
                    <td className="px-3 py-2 font-mono text-xs text-slate-300">{fmt(p.feature_b)}</td>
                    <td className="px-3 py-2 font-mono text-xs text-sky-400/90">{fmt(p.pearson)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CollapsibleSection>
      ) : null}
    </div>
  );
}
