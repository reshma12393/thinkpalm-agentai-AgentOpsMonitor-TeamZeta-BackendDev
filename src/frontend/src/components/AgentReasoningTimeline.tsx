import { useState } from "react";
import type { ReasoningLogEntry } from "../api";
import { formatAgentLabel } from "../lib/agentLabels";

export type AgentReasoningTimelineProps = {
  logs: ReasoningLogEntry[];
};

type AgentGroup = {
  agent: string;
  entries: ReasoningLogEntry[];
};

function groupByAgentInOrder(logs: ReasoningLogEntry[]): AgentGroup[] {
  const order: string[] = [];
  const map = new Map<string, ReasoningLogEntry[]>();
  for (const log of logs) {
    const a = log.agent || "unknown";
    if (!map.has(a)) {
      order.push(a);
      map.set(a, []);
    }
    map.get(a)!.push(log);
  }
  return order.map((agent) => ({ agent, entries: map.get(agent)! }));
}

export function AgentReasoningTimeline({ logs }: AgentReasoningTimelineProps) {
  const groups = groupByAgentInOrder(logs);
  const [openAgents, setOpenAgents] = useState<Record<string, boolean>>(() => {
    const init: Record<string, boolean> = {};
    for (const g of groups) {
      init[g.agent] = true;
    }
    return init;
  });

  if (!logs.length) {
    return <p className="text-sm text-slate-500">No reasoning logs were returned for this run.</p>;
  }

  const toggle = (agent: string) => {
    setOpenAgents((prev) => ({ ...prev, [agent]: !prev[agent] }));
  };

  return (
    <ul className="space-y-0">
      {groups.map((group, gi) => {
        const expanded = openAgents[group.agent] !== false;
        const isLast = gi === groups.length - 1;
        return (
          <li key={group.agent} className="flex gap-0">
            <div className="flex w-10 shrink-0 flex-col items-center">
              <div
                className="z-10 mt-2 h-3.5 w-3.5 rounded-full border-2 border-emerald-400/90 bg-slate-950 shadow-[0_0_14px_rgba(52,211,153,0.35)]"
                aria-hidden
              />
              {!isLast ? <div className="w-px flex-1 min-h-[1.5rem] bg-gradient-to-b from-emerald-600/30 to-slate-700" /> : null}
            </div>
            <div className={`min-w-0 flex-1 pb-8 ${isLast ? "pb-0" : ""}`}>
              <div className="rounded-xl border border-slate-700/80 bg-slate-950/50">
                <button
                  type="button"
                  onClick={() => toggle(group.agent)}
                  className="flex w-full items-center justify-between gap-2 rounded-xl px-3 py-2.5 text-left text-sm font-semibold text-slate-200 hover:bg-slate-800/40"
                >
                  <span>
                    {formatAgentLabel(group.agent)}
                    <span className="ml-2 font-normal text-slate-500">
                      ({group.entries.length} step{group.entries.length === 1 ? "" : "s"})
                    </span>
                  </span>
                  <span className={`shrink-0 text-slate-500 transition ${expanded ? "rotate-180" : ""}`}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M6 9l6 6 6-6" />
                    </svg>
                  </span>
                </button>
                {expanded ? (
                  <ol className="space-y-4 border-t border-slate-800 px-3 py-3">
                    {group.entries.map((entry, idx) => (
                      <li key={`${group.agent}-${idx}-${entry.step}`}>
                        <p className="text-xs font-semibold uppercase tracking-wide text-emerald-500/90">{entry.step}</p>
                        <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-slate-900/80 p-3 font-mono text-xs leading-relaxed text-slate-400">
                          {entry.content}
                        </pre>
                      </li>
                    ))}
                  </ol>
                ) : null}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
