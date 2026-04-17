import type { AutomlDebateResponse } from "../api";
import { edaQuickSummary } from "../lib/formatters";
import { AgentReasoningTimeline } from "./AgentReasoningTimeline";
import { AgentTracePanel } from "./AgentTracePanel";
import { CollapsibleSection } from "./CollapsibleSection";
import { JsonDownloadPanel } from "./JsonDownloadPanel";
import { EdaSummaryTables } from "./EdaSummaryTables";
import { ModelComparisonTable } from "./ModelComparisonTable";
import { WinnerCard } from "./WinnerCard";

export type ResultsDashboardProps = {
  result: AutomlDebateResponse;
};

/**
 * Composes collapsible sections + winner card + agent reasoning timeline + agent trace.
 */
export function ResultsDashboard({ result }: ResultsDashboardProps) {
  const logs = result.reasoning_logs ?? [];
  const trace = result.agent_trace ?? {};
  const nodeCount = Array.isArray(trace.pipeline_nodes) ? trace.pipeline_nodes.length : 0;
  const conf = result.judge_confidence;

  return (
    <div className="mt-10 grid gap-6">
      <WinnerCard
        winner={result.winner}
        winnerAgent={result.winner_agent}
        judgeReason={result.judge_reason}
        judgeConfidence={typeof conf === "number" ? conf : undefined}
      />

      <CollapsibleSection title="EDA" subtitle={edaQuickSummary(result.eda)} badge="Tables" defaultOpen>
        <div className="space-y-4">
          <EdaSummaryTables eda={result.eda} />
          <CollapsibleSection
            title="Full structured EDA (JSON)"
            subtitle="Complete payload — debugging, copy/export."
            badge="Raw"
            defaultOpen={false}
            variant="muted"
          >
            <JsonDownloadPanel
              data={result.eda}
              label="Full structured EDA (JSON)"
              subtitle="Complete EDA payload"
              fileName="eda_structured"
            />
          </CollapsibleSection>
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        title="Model comparison"
        subtitle="Holdout metrics — the judge’s pick is highlighted."
        badge={`${result.models.length} models`}
        defaultOpen
      >
        <ModelComparisonTable
          embedded
          models={result.models}
          metricsByModel={result.metrics}
          winnerModelKey={result.winner}
        />
      </CollapsibleSection>

      <CollapsibleSection title="Debate summary" badge="Transcript" defaultOpen={false}>
        <div className="max-h-[28rem] overflow-auto rounded-xl border border-slate-800 bg-slate-950/40 p-4">
          <p className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-slate-300">{result.debate || "—"}</p>
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        title="Agent reasoning timeline"
        subtitle="Pipeline steps grouped by agent — expand each group to read logs."
        badge={`${logs.length} entries`}
        defaultOpen
        variant="muted"
      >
        <AgentReasoningTimeline logs={logs} />
      </CollapsibleSection>

      <CollapsibleSection
        title="Agent trace"
        subtitle="LangGraph pipeline order and merged state (metrics, judge, memory, proposals)."
        badge={nodeCount ? `${nodeCount} nodes` : "State"}
        defaultOpen={false}
        variant="muted"
      >
        <AgentTracePanel trace={trace} />
      </CollapsibleSection>
    </div>
  );
}
