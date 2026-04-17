import type { AutomlDebateResponse } from "../api";
import { edaQuickSummary } from "../lib/formatters";
import { AgentReasoningTimeline } from "./AgentReasoningTimeline";
import { CollapsibleSection } from "./CollapsibleSection";
import { ModelComparisonTable } from "./ModelComparisonTable";
import { WinnerCard } from "./WinnerCard";

export type ResultsDashboardProps = {
  result: AutomlDebateResponse;
};

/**
 * Composes collapsible sections + winner card + agent reasoning timeline.
 */
export function ResultsDashboard({ result }: ResultsDashboardProps) {
  const logs = result.reasoning_logs ?? [];
  const conf = result.judge_confidence;

  return (
    <div className="mt-10 grid gap-6">
      <WinnerCard
        winner={result.winner}
        winnerAgent={result.winner_agent}
        judgeReason={result.judge_reason}
        judgeConfidence={typeof conf === "number" ? conf : undefined}
      />

      <CollapsibleSection title="EDA summary" subtitle={edaQuickSummary(result.eda)} badge="Structured" defaultOpen>
        <div className="max-h-80 overflow-auto rounded-xl border border-slate-800 bg-slate-950/50 p-4">
          <pre className="whitespace-pre-wrap break-words font-mono text-xs text-slate-400">
            {JSON.stringify(result.eda, null, 2)}
          </pre>
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
    </div>
  );
}
