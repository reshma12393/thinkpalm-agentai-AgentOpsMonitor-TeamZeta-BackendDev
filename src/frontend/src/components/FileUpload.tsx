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
  submitLabel = "Run AutoML Arena",
  loadingLabel = "Running pipeline…",
}: FileUploadProps) {
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
          <label htmlFor={selectId}>Target column</label>
          <select
            id={selectId}
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
            submitLabel
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
