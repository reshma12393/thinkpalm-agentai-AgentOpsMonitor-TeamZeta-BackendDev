import axios from "axios";

const client = axios.create({
  baseURL: "",
  timeout: 0,
  headers: { Accept: "application/json" },
});

function formatAxiosError(err: unknown): string {
  if (!axios.isAxiosError(err)) {
    return err instanceof Error ? err.message : String(err);
  }
  const ax = err as import("axios").AxiosError<{ detail?: unknown }>;
  const d = ax.response?.data?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d.map((x: { msg?: string }) => x.msg ?? JSON.stringify(x)).join("; ");
  }
  if (d != null && typeof d === "object") {
    return JSON.stringify(d);
  }
  return ax.message || `HTTP ${ax.response?.status ?? "error"}`;
}

export type ReasoningLogEntry = {
  agent: string;
  step: string;
  content: string;
  metadata?: Record<string, unknown>;
};

export type AutomlModelRow = {
  model_key: string;
  agent: string;
  model_type: string;
  artifact_path: string;
  params: Record<string, unknown>;
  metrics?: Record<string, number | string> | null;
  proposal?: Record<string, unknown> | null;
  evaluation?: Record<string, unknown> | null;
};

/** Response from POST /automl-debate */
export type AutomlDebateResponse = {
  eda: Record<string, unknown>;
  models: AutomlModelRow[];
  debate: string;
  winner: string;
  winner_agent?: string;
  judge_reason?: string;
  judge_confidence?: number;
  metrics: Record<string, Record<string, number | string | unknown>>;
  reasoning_logs?: ReasoningLogEntry[];
  /** LangGraph pipeline state snapshot (node order, metrics, judge, memory, …). */
  agent_trace?: Record<string, unknown>;
};

export type CsvColumnsResponse = {
  columns: string[];
};

/** POST /api/v1/dataset/csv-columns — header row only, same file can be submitted again for the full run. */
export async function postCsvColumns(file: File): Promise<string[]> {
  const fd = new FormData();
  fd.append("file", file);
  try {
    const { data } = await client.post<CsvColumnsResponse>("/api/v1/dataset/csv-columns", fd);
    return data.columns ?? [];
  } catch (e) {
    throw new Error(formatAxiosError(e));
  }
}

export async function postAutomlDebate(file: File, targetColumn: string): Promise<AutomlDebateResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("target_column", targetColumn.trim());
  try {
    const { data } = await client.post<AutomlDebateResponse>("/automl-debate", fd);
    return data;
  } catch (e) {
    throw new Error(formatAxiosError(e));
  }
}

export type ChatMessage = {
  role: "user" | "assistant" | "system";
  content: string;
};

export type ChatResponse = {
  reply: string;
  model: string;
};

export async function postChat(
  messages: ChatMessage[],
  runContext: Record<string, unknown> | null,
): Promise<ChatResponse> {
  try {
    const { data } = await client.post<ChatResponse>("/api/v1/chat", {
      messages,
      run_context: runContext,
    });
    return data;
  } catch (e) {
    throw new Error(formatAxiosError(e));
  }
}

export async function healthCheck(): Promise<boolean> {
  try {
    const { data } = await client.get<{ status?: string }>("/health");
    return data?.status === "ok";
  } catch {
    return false;
  }
}
