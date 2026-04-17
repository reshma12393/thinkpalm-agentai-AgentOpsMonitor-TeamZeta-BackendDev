export type WinnerCardProps = {
  winner: string;
  subtitle?: string;
  winnerAgent?: string;
  judgeReason?: string;
  judgeConfidence?: number;
};

export function WinnerCard({
  winner,
  subtitle = "Selected by the judge agent from holdout metrics and debate.",
  winnerAgent,
  judgeReason,
  judgeConfidence,
}: WinnerCardProps) {
  const conf =
    typeof judgeConfidence === "number" && !Number.isNaN(judgeConfidence)
      ? judgeConfidence.toFixed(2)
      : null;

  return (
    <section className="relative overflow-hidden rounded-2xl border-2 border-emerald-500/40 bg-gradient-to-br from-emerald-950/60 via-slate-900/90 to-slate-950 p-6 shadow-lg shadow-emerald-900/20 ring-1 ring-emerald-400/20">
      <div className="pointer-events-none absolute -right-8 -top-8 h-32 w-32 rounded-full bg-emerald-500/10 blur-2xl" />
      <p className="text-sm font-medium uppercase tracking-wide text-emerald-400/90">Final winner</p>
      <p className="mt-2 font-mono text-4xl font-bold tracking-tight text-white">{winner}</p>
      {winnerAgent ? (
        <p className="mt-1 text-xs text-emerald-200/70">
          Agent: <span className="font-mono text-emerald-100">{winnerAgent}</span>
        </p>
      ) : null}
      {conf !== null ? (
        <p className="mt-1 text-sm text-slate-400">
          Judge confidence: <span className="font-mono text-slate-200">{conf}</span> (0–1)
        </p>
      ) : null}
      <p className="mt-3 text-sm text-slate-400">{subtitle}</p>
      {judgeReason ? (
        <div className="mt-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-emerald-300/90">Reasoning</h3>
          <div className="mt-1.5 rounded-xl border border-emerald-800/50 bg-slate-950/50 p-3 text-sm leading-relaxed text-slate-300">
            {judgeReason}
          </div>
        </div>
      ) : null}
    </section>
  );
}
