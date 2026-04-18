from typing import Any

from pydantic import BaseModel, Field


class CsvColumnsResponse(BaseModel):
    """Column names from the header row of an uploaded CSV (preview parse)."""

    columns: list[str] = Field(description="Ordered header names as read by pandas.")


class ReasoningLogEntry(BaseModel):
    agent: str
    step: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRunSummary(BaseModel):
    model_key: str = ""
    agent: str
    model_type: str
    artifact_path: str
    params: dict[str, Any] = Field(default_factory=dict)
    proposal: dict[str, Any] | None = None
    metrics: dict[str, float | str] | None = None
    evaluation: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None


class MetricsRow(BaseModel):
    model_key: str
    agent: str
    metrics: dict[str, float | str]


class JudgeDecision(BaseModel):
    winner: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0, description="Model certainty about the choice, 0–1.")
    winner_agent: str = ""


class DebateRunResult(BaseModel):
    run_id: str
    status: str
    target_column: str
    task_type: str
    eda_summary: str
    eda_structured: dict[str, Any] | None = None
    evaluation_report: dict[str, Any] | None = None
    model_runs: list[ModelRunSummary]
    metrics_comparison: list[MetricsRow]
    debate_transcript: str
    debate_analysis: dict[str, Any] | None = None
    judge: JudgeDecision
    reasoning_logs: list[ReasoningLogEntry]
    error: str | None = None


class ChatMessage(BaseModel):
    role: str = Field(description="One of: user, assistant, system.")
    content: str = Field(description="Message body (markdown allowed).")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="Ordered chat history; the final message should be the latest user turn.",
    )
    run_context: dict[str, Any] | None = Field(
        default=None,
        description="Optional snapshot of the current AutoML run (eda, models, metrics, judge_reason, debate, …).",
    )


class ChatResponse(BaseModel):
    reply: str
    model: str = Field(default="", description="Identifier of the LLM that produced the reply.")


class DebateRunStatus(BaseModel):
    run_id: str
    status: str
    message: str | None = None
    result: DebateRunResult | None = None


class AutomlDebateResponse(BaseModel):
    """Synchronous POST /automl-debate payload: full structured run result."""

    eda: dict[str, Any] = Field(description="Structured EDA JSON (schema_version, deterministic, llm_reasoning, …).")
    models: list[dict[str, Any]] = Field(
        description="Each trained candidate: model_key, agent, model_type, metrics, proposal, artifact_path, …"
    )
    debate: str = Field(description="Debate agent transcript (metric-grounded comparison + optional LLM narrative).")
    winner: str = Field(description="Judge-selected winning model key (e.g. rf, xgb, lr).")
    winner_agent: str = Field(default="", description="Pipeline agent id for the winning model.")
    judge_reason: str = Field(default="", description="Judge rationale text.")
    judge_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Judge confidence 0–1.")
    metrics: dict[str, Any] = Field(
        description="Holdout metrics keyed by model_key (values are metric name → number or string)."
    )
    reasoning_logs: list[ReasoningLogEntry] = Field(
        default_factory=list,
        description="Ordered agent reasoning steps for timeline UI.",
    )
    agent_trace: dict[str, Any] = Field(
        default_factory=dict,
        description="LangGraph pipeline state snapshot (nodes, metrics, judge, memory, …) for Agent trace UI.",
    )
