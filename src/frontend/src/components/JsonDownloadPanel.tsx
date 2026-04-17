import { useMemo, useState } from "react";

/**
 * Reusable control bar for arbitrary JSON payloads:
 *   View JSON toggle · Download JSON · Copy JSON
 */

export type JsonDownloadPanelProps = {
  /** Any JSON-serializable value (objects, arrays, primitives). */
  data: unknown;
  /** Small label shown above the bar (e.g. "Graph state (JSON)"). */
  label: string;
  /** Small secondary line (optional). Size suffix is appended automatically. */
  subtitle?: string;
  /** File name for the download (without extension). `.json` is appended. */
  fileName: string;
  /** Initial open state for the inline viewer. Defaults to false. */
  defaultViewing?: boolean;
  /** DOM id for the viewer block (for aria-controls). */
  viewerId?: string;
};

export function JsonDownloadPanel({
  data,
  label,
  subtitle,
  fileName,
  defaultViewing = false,
  viewerId,
}: JsonDownloadPanelProps) {
  const [copied, setCopied] = useState(false);
  const [viewing, setViewing] = useState(defaultViewing);

  const jsonText = useMemo(() => {
    try {
      return JSON.stringify(data ?? {}, null, 2);
    } catch {
      return String(data ?? "");
    }
  }, [data]);

  const sizeKb = useMemo(() => (new Blob([jsonText]).size / 1024).toFixed(1), [jsonText]);
  const autoId = viewerId ?? `json-viewer-${Math.abs(hash(label + fileName))}`;

  const downloadJson = () => {
    try {
      const blob = new Blob([jsonText], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const safe = fileName.replace(/[^\w.-]+/g, "_") || "data";
      a.download = `${safe}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch {
      /* no-op */
    }
  };

  const copyJson = async () => {
    try {
      await navigator.clipboard.writeText(jsonText);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800/90 bg-slate-950/60 px-4 py-3">
        <div className="min-w-0">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
          <p className="mt-1 text-xs text-slate-500">
            {subtitle ? <span>{subtitle} · </span> : null}
            <span className="font-mono text-slate-400">{sizeKb} KB</span>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setViewing((v) => !v)}
            aria-expanded={viewing}
            aria-controls={autoId}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900/80 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-slate-500 hover:bg-slate-800/80 focus:outline-none focus:ring-2 focus:ring-slate-500/40"
            title={viewing ? "Hide JSON" : "Show JSON"}
          >
            <EyeIcon className="h-3.5 w-3.5" />
            {viewing ? "Hide JSON" : "View JSON"}
          </button>
          <button
            type="button"
            onClick={downloadJson}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900/80 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-emerald-500/50 hover:bg-emerald-500/10 hover:text-emerald-200 focus:outline-none focus:ring-2 focus:ring-emerald-500/35"
            title={`Download as ${fileName}.json`}
          >
            <DownloadIcon className="h-3.5 w-3.5" />
            Download JSON
          </button>
          <button
            type="button"
            onClick={() => void copyJson()}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900/80 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:border-slate-500 hover:bg-slate-800/80 focus:outline-none focus:ring-2 focus:ring-slate-500/40"
            title="Copy JSON to clipboard"
          >
            <CopyIcon className="h-3.5 w-3.5" />
            {copied ? "Copied" : "Copy JSON"}
          </button>
        </div>
      </div>

      {viewing ? (
        <div id={autoId} className="max-h-[32rem] overflow-auto rounded-xl border border-slate-800/90 bg-slate-950/60 p-4">
          <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-slate-400">{jsonText}</pre>
        </div>
      ) : null}
    </div>
  );
}

/** Tiny, stable (non-cryptographic) string hash for deterministic viewer ids. */
function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h << 5) - h + s.charCodeAt(i);
    h |= 0;
  }
  return h;
}

function EyeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.5 12S5.5 5.5 12 5.5 21.5 12 21.5 12 18.5 18.5 12 18.5 2.5 12 2.5 12z" />
      <circle cx="12" cy="12" r="3" />
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
