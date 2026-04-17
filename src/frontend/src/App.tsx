/**
 * Root application shell. Maps to an "App.js" entry in plain React projects; this repo uses TypeScript (App.tsx).
 */
import { useCallback, useEffect, useState } from "react";
import type { AutomlDebateResponse } from "./api";
import { healthCheck, postAutomlDebate, postCsvColumns } from "./api";
import { FileUpload } from "./components/FileUpload";
import { ResultsDashboard } from "./components/ResultsDashboard";

function guessDefaultTargetColumn(columns: string[]): string {
  const pref =
    columns.find((c) => /^(target|label)$/i.test(c)) ??
    columns.find((c) => /^species$/i.test(c)) ??
    columns[0] ??
    "";
  return pref;
}

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [targetOptions, setTargetOptions] = useState<string[]>([]);
  const [targetOptionsLoading, setTargetOptionsLoading] = useState(false);
  const [targetOptionsError, setTargetOptionsError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AutomlDebateResponse | null>(null);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);

  const onFileChange = useCallback((f: File | null) => {
    setFile(f);
    setTargetOptions([]);
    setTargetOptionsError(null);
    setTargetColumn("");
    if (!f) {
      setTargetOptionsLoading(false);
      return;
    }
    setTargetOptionsLoading(true);
    void postCsvColumns(f)
      .then((cols) => {
        setTargetOptions(cols);
        setTargetColumn(guessDefaultTargetColumn(cols));
      })
      .catch((e) => {
        setTargetOptionsError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        setTargetOptionsLoading(false);
      });
  }, []);

  const runPipeline = useCallback(async () => {
    setError(null);
    setResult(null);
    if (!file) {
      setError("Please choose a CSV file.");
      return;
    }
    if (targetOptionsError) {
      setError("Could not read CSV columns. Fix the file or try another upload, then pick a target column.");
      return;
    }
    if (targetOptionsLoading) {
      setError("Still reading CSV columns; wait a moment.");
      return;
    }
    if (targetOptions.length > 0 && !targetOptions.includes(targetColumn)) {
      setError("Pick a target column from the dropdown.");
      return;
    }
    if (!targetColumn.trim()) {
      setError("Select the target column.");
      return;
    }
    setLoading(true);
    try {
      const data = await postAutomlDebate(file, targetColumn.trim());
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [file, targetColumn, targetOptions, targetOptionsLoading]);

  useEffect(() => {
    void healthCheck().then(setBackendOk);
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 pb-20">
      <header className="mb-10">
        <h1 className="text-3xl font-semibold tracking-tight text-white">AutoML Arena</h1>
        <p className="mt-1 text-lg font-medium tracking-tight text-slate-300">
          A Multi-Agent Debate System for Model Selection
        </p>
        <p className="mt-4 max-w-2xl text-base leading-relaxed text-slate-300">
          AutoML Arena is a multi-agent system where users upload datasets, competing ML models debate using real
          performance metrics, and an intelligent judge selects the most robust model.
        </p>
        <p className="mt-4 max-w-2xl text-slate-400">
          Upload a CSV dataset, select the target column, and launch the pipeline:{" "}
          <strong>
            Prepare → EDA → Memory Recall → Model Training (RF, XGBoost, Linear in parallel) → Evaluation → Debate → Judge
          </strong>
          .
        </p>
        {backendOk === false ? (
          <p className="mt-3 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
            Backend not reachable at <code className="rounded bg-slate-800 px-1">/health</code>. Start the FastAPI
            server (e.g. port 8000) and keep the Vite proxy.
          </p>
        ) : null}
      </header>

      <FileUpload
        file={file}
        targetColumn={targetColumn}
        targetOptions={targetOptions}
        targetOptionsLoading={targetOptionsLoading}
        targetOptionsError={targetOptionsError}
        loading={loading}
        error={error}
        onFileChange={onFileChange}
        onTargetChange={setTargetColumn}
        onSubmit={runPipeline}
      />

      {result ? <ResultsDashboard result={result} /> : null}
    </div>
  );
}
