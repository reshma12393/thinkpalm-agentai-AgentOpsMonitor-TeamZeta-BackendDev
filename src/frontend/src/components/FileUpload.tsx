import { useState } from "react";
import { AndroidRobotIcon } from "./AndroidRobotIcon";
import { Spinner } from "./Spinner";

export type FileUploadProps = {
  file: File | null;
  targetColumn: string;
  /** Column names from POST /api/v1/dataset/csv-columns (empty before fetch or on failure). */
  targetOptions: string[];
  targetOptionsLoading: boolean;
  targetOptionsError: string | null;
  loading: boolean;
  error: string | null;
  onFileChange: (file: File | null) => void;
  onTargetChange: (value: string) => void;
  onSubmit: () => void;
  submitLabel?: string;
  loadingLabel?: string;
};

/**
 * Controlled form: CSV file + target column + submit. Parent owns all state.
 */
export function FileUpload({
  file,
  targetColumn,
  targetOptions,
  targetOptionsLoading,
  targetOptionsError,
  loading,
  error,
  onFileChange,
  onTargetChange,
  onSubmit,
  submitLabel = "Run",
  loadingLabel = "Running pipeline…",
}: FileUploadProps) {
  const [targetHintOpen, setTargetHintOpen] = useState(false);
  const selectDisabled = loading || !file || targetOptionsLoading || targetOptions.length === 0;
  const selectId = "target-column-select";

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 shadow-xl shadow-black/40 backdrop-blur">
      <h2 className="text-lg font-medium text-slate-200">Dataset</h2>
      <div className="mt-4 grid gap-5 sm:grid-cols-2">
        <label className="grid gap-2 text-sm text-slate-400">
          CSV file
          <input
            type="file"
            accept=".csv,text/csv"
            disabled={loading}
            onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
            className="cursor-pointer rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-slate-200 file:mr-3 file:rounded-md file:border-0 file:bg-emerald-600/80 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-white hover:file:bg-emerald-500"
          />
        </label>
        <div className="grid gap-2 text-sm text-slate-400">
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor={selectId} className="text-slate-400">
              Target column
            </label>
            <button
              type="button"
              className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-slate-600 bg-slate-800/90 text-amber-400/95 transition hover:border-amber-500/45 hover:bg-amber-500/10 hover:text-amber-200 focus:outline-none focus:ring-2 focus:ring-amber-500/35"
              onClick={() => setTargetHintOpen((v) => !v)}
              aria-expanded={targetHintOpen}
              aria-controls="target-column-hint"
              aria-label={targetHintOpen ? "Hide target column warning" : "Show target column warning"}
              title="Target column: click to show or hide the warning"
            >
              <WarningTriangleIcon className="h-4 w-4" aria-hidden />
            </button>
          </div>
          <p
            id="target-column-hint"
            className={
              targetHintOpen
                ? "flex gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-2.5 py-2 text-xs leading-snug text-amber-100/95"
                : "hidden"
            }
            role="region"
          >
            <span className="select-none" aria-hidden>
              ⚠️
            </span>
            <span>Select the correct target column—this is the variable the model will try to predict.</span>
          </p>
          <select
            id={selectId}
            aria-describedby={targetHintOpen ? "target-column-hint" : undefined}
            value={targetOptions.includes(targetColumn) ? targetColumn : ""}
            disabled={selectDisabled}
            onChange={(e) => onTargetChange(e.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-slate-100 focus:border-emerald-500/50 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {!file ? (
              <option value="">Choose a CSV file first</option>
            ) : targetOptionsLoading ? (
              <option value="">Reading columns…</option>
            ) : targetOptions.length === 0 ? (
              <option value="">No columns loaded</option>
            ) : (
              <>
                <option value="" disabled>
                  Select target column
                </option>
                {targetOptions.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </>
            )}
          </select>
          {targetOptionsError ? (
            <p className="text-xs text-rose-300/90" role="alert">
              {targetOptionsError}
            </p>
          ) : null}
        </div>
      </div>
      <div className="mt-6 flex flex-wrap items-center gap-4">
        <button
          type="button"
          disabled={loading}
          onClick={() => void onSubmit()}
          className="inline-flex items-center justify-center rounded-xl bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-emerald-900/30 transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <Spinner />
              {loadingLabel}
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <AndroidRobotIcon className="h-5 w-5 shrink-0 text-white" />
              {submitLabel}
            </span>
          )}
        </button>
        {loading ? <span className="text-sm text-slate-500">This may take a minute while models train.</span> : null}
        {file ? (
          <span className="text-sm text-slate-500">
            Selected: <span className="font-mono text-slate-300">{file.name}</span>
          </span>
        ) : null}
      </div>
      {error ? (
        <div
          className="mt-5 rounded-xl border border-rose-500/40 bg-rose-950/40 px-4 py-3 text-sm text-rose-100"
          role="alert"
        >
          <strong className="font-semibold">Error: </strong>
          {error}
        </div>
      ) : null}
    </section>
  );
}

function WarningTriangleIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      stroke="currentColor"
      strokeWidth={2}
      aria-hidden="true"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
      />
    </svg>
  );
}
