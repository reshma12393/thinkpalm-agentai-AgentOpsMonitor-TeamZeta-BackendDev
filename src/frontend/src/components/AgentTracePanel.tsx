import { useMemo } from "react";
import { JsonDownloadPanel } from "./JsonDownloadPanel";

/**
 * Collapsible “Agent trace” content: pipeline node order + downloadable JSON snapshot of graph state.
 */

export type AgentTracePanelProps = {
  trace: Record<string, unknown>;
};

export function AgentTracePanel({ trace }: AgentTracePanelProps) {
  const nodes = useMemo(
    () => (Array.isArray(trace.pipeline_nodes) ? (trace.pipeline_nodes as unknown[]).map((n) => String(n)) : []),
    [trace],
  );
  const runId = typeof trace.run_id === "string" && trace.run_id ? trace.run_id : "run";

  if (Object.keys(trace).length === 0) {
    return <p className="text-sm text-slate-500">No agent trace payload was returned for this run.</p>;
  }

  return (
    <div className="space-y-5">
      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Pipeline graph (node order)</p>
        <div className="flex flex-wrap items-center gap-y-2 text-xs leading-relaxed">
          {nodes.length === 0 ? (
            <span className="text-slate-500">—</span>
          ) : (
            nodes.map((n, i) => (
              <span key={`${n}-${i}`} className="inline-flex items-center gap-1">
                {i > 0 ? (
                  <span className="px-0.5 text-slate-600" aria-hidden>
                    →
                  </span>
                ) : null}
                <code className="rounded border border-emerald-500/30 bg-emerald-950/35 px-2 py-1 font-mono text-[0.7rem] text-emerald-100/95">
                  {n}
                </code>
              </span>
            ))
          )}
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Parallel workers <code className="rounded bg-slate-800/80 px-1">model_rf</code>,{" "}
          <code className="rounded bg-slate-800/80 px-1">model_xgb</code>,{" "}
          <code className="rounded bg-slate-800/80 px-1">model_lr</code> run after memory retrieval; evaluation joins
          their outputs.
        </p>
      </div>

      <JsonDownloadPanel
        data={trace}
        label="Graph state (JSON)"
        subtitle="Full LangGraph state snapshot"
        fileName={`agent_trace_${runId}`}
      />
    </div>
  );
}
