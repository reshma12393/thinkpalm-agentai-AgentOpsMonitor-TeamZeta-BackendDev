export type DebatePanelProps = {
  debate: string;
  title?: string;
};

export function DebatePanel({ debate, title = "Debate summary" }: DebatePanelProps) {
  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
      <h2 className="text-lg font-medium text-slate-200">{title}</h2>
      <div className="mt-4 max-h-[28rem] overflow-auto rounded-xl border border-slate-800 bg-slate-950/40 p-4">
        <p className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-slate-300">{debate || "—"}</p>
      </div>
    </section>
  );
}
