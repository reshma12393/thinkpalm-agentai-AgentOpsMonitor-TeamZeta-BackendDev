import { useEffect, useMemo, useRef, useState } from "react";
import type { AutomlDebateResponse, ChatMessage } from "../api";
import { postChat } from "../api";

export type ChatAssistantProps = {
  /** Latest successful run; used as compact context for run-specific questions. */
  result: AutomlDebateResponse | null;
};

type Msg = ChatMessage & { id: string };

const STORAGE_KEY = "automlarena.chat.history.v1";

function uid(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function buildRunContext(r: AutomlDebateResponse | null): Record<string, unknown> | null {
  if (!r) return null;
  const det =
    (r.eda && typeof r.eda === "object" ? (r.eda as Record<string, unknown>).deterministic : null) ??
    null;
  return {
    task_type:
      det && typeof det === "object"
        ? ((det as Record<string, unknown>).target_profile as Record<string, unknown> | undefined)?.task_hint
        : undefined,
    target_column:
      det && typeof det === "object"
        ? ((det as Record<string, unknown>).target_profile as Record<string, unknown> | undefined)?.target_column
        : undefined,
    eda: r.eda,
    models: r.models,
    metrics: r.metrics,
    winner: r.winner,
    winner_agent: r.winner_agent,
    judge_reason: r.judge_reason,
    judge_confidence: r.judge_confidence,
    debate: r.debate,
  };
}

const INITIAL_GREETING: Msg = {
  id: "greet",
  role: "assistant",
  content:
    "Hi — I'm the AutoML Arena advisor. I give short summaries on general ML picks " +
    "(e.g. \"which algorithm fits a small imbalanced tabular dataset?\") and on your current run " +
    "(e.g. \"why did the judge pick this model?\"). When a run is loaded I automatically see its EDA, metrics, " +
    "debate and judge reasoning.",
};

export function ChatAssistant({ result }: ChatAssistantProps) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Msg[]>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as Msg[];
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch {
      /* ignore corrupt state */
    }
    return [INITIAL_GREETING];
  });
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const runContext = useMemo(() => buildRunContext(result), [result]);
  const hasRun = Boolean(result);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
      /* storage quota / private mode */
    }
  }, [messages]);

  useEffect(() => {
    if (open && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [open, messages, sending]);

  const resetChat = () => {
    setMessages([INITIAL_GREETING]);
    setError(null);
  };

  const send = async () => {
    const trimmed = input.trim();
    if (!trimmed || sending) return;
    setError(null);
    const userMsg: Msg = { id: uid(), role: "user", content: trimmed };
    const next = [...messages, userMsg];
    setMessages(next);
    setInput("");
    setSending(true);
    try {
      const history: ChatMessage[] = next.map(({ role, content }) => ({ role, content }));
      const { reply } = await postChat(history, runContext);
      setMessages((curr) => [...curr, { id: uid(), role: "assistant", content: reply || "(empty response)" }]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setMessages((curr) => [
        ...curr,
        {
          id: uid(),
          role: "assistant",
          content: `Sorry — I couldn't reach the assistant (${msg}). Check the backend and retry.`,
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  };

  return (
    <>
      <button
        type="button"
        aria-label={open ? "Close ML advisor chat" : "Open ML advisor chat"}
        aria-expanded={open}
        aria-controls="ml-advisor-panel"
        onClick={() => setOpen((v) => !v)}
        className="fixed bottom-5 right-5 z-40 inline-flex h-14 w-14 items-center justify-center rounded-full bg-emerald-600 text-white shadow-xl shadow-emerald-900/40 ring-1 ring-emerald-400/40 transition hover:bg-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-400"
        title={open ? "Close chat" : "Ask the ML advisor"}
      >
        {open ? <CloseIcon className="h-6 w-6" /> : <ChatIcon className="h-6 w-6" />}
      </button>

      {open ? (
        <div
          id="ml-advisor-panel"
          role="dialog"
          aria-label="ML advisor chat"
          className="fixed bottom-24 right-5 z-40 flex h-[32rem] w-[22rem] max-w-[calc(100vw-2.5rem)] flex-col overflow-hidden rounded-2xl border border-slate-700 bg-slate-950/95 shadow-2xl shadow-black/50 backdrop-blur"
        >
          <div className="flex items-center justify-between gap-2 border-b border-slate-800 bg-slate-900/80 px-4 py-3">
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-slate-100">ML Advisor</p>
              <p className="truncate text-[11px] text-slate-500">
                {hasRun ? "Connected to this run" : "General guidance (no run loaded)"}
              </p>
            </div>
            <button
              type="button"
              onClick={resetChat}
              className="rounded-md border border-slate-700 px-2 py-1 text-[11px] text-slate-300 hover:border-slate-500 hover:bg-slate-800/70"
              title="Clear chat history"
            >
              Reset
            </button>
          </div>

          <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-3 py-3">
            {messages.map((m) => (
              <MessageBubble key={m.id} role={m.role} content={m.content} />
            ))}
            {sending ? (
              <div className="flex items-center gap-2 px-1 text-xs text-slate-500">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                Thinking…
              </div>
            ) : null}
            {error ? (
              <p className="rounded-md border border-rose-500/40 bg-rose-950/40 px-2 py-1 text-[11px] text-rose-200">
                {error}
              </p>
            ) : null}
          </div>

          <div className="border-t border-slate-800 bg-slate-900/70 p-2">
            <div className="flex items-end gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                rows={2}
                placeholder={
                  hasRun
                    ? "Ask about this run or ML in general…"
                    : "Ask about algorithms, dataset shapes, evaluation…"
                }
                className="min-h-[2.5rem] flex-1 resize-none rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:border-emerald-500/60 focus:outline-none focus:ring-2 focus:ring-emerald-500/25"
              />
              <button
                type="button"
                disabled={sending || !input.trim()}
                onClick={() => void send()}
                className="inline-flex h-10 shrink-0 items-center gap-1.5 rounded-lg bg-emerald-600 px-3 text-sm font-medium text-white shadow-sm transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
                title="Send (Enter)"
              >
                <SendIcon className="h-4 w-4" />
                Send
              </button>
            </div>
            <p className="mt-1 px-1 text-[10px] text-slate-600">
              Enter to send · Shift+Enter for newline · Clears only in this browser
            </p>
          </div>
        </div>
      ) : null}
    </>
  );
}

function MessageBubble({ role, content }: { role: ChatMessage["role"]; content: string }) {
  const isUser = role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap break-words ${
          isUser
            ? "bg-emerald-600/90 text-white"
            : "border border-slate-800 bg-slate-900/70 text-slate-200"
        }`}
      >
        {content}
      </div>
    </div>
  );
}

function ChatIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M21 12c0 4.418-4.03 8-9 8a9.94 9.94 0 01-4.22-.924L3 20l1.07-3.8A7.97 7.97 0 013 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  );
}

function CloseIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}

function SendIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 12l14-7-4 14-3-6-7-1z" />
    </svg>
  );
}
