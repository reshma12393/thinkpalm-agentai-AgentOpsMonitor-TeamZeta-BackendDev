from __future__ import annotations

import asyncio
import uuid
from typing import Any

from app.agents.judge_agent import normalize_judge_payload
from app.schemas import DebateRunResult, DebateRunStatus, JudgeDecision, MetricsRow, ModelRunSummary, ReasoningLogEntry


class RunStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._runs: dict[str, dict[str, Any]] = {}

    async def create(self, csv_path: str, target_column: str) -> str:
        run_id = str(uuid.uuid4())
        async with self._lock:
            self._runs[run_id] = {
                "status": "queued",
                "csv_path": csv_path,
                "target_column": target_column,
                "message": None,
                "result": None,
            }
        return run_id

    async def update(self, run_id: str, **kwargs: Any) -> None:
        async with self._lock:
            if run_id not in self._runs:
                return
            self._runs[run_id].update(kwargs)

    async def get(self, run_id: str) -> dict[str, Any] | None:
        async with self._lock:
            return self._runs.get(run_id)

    def to_status(self, run_id: str, data: dict[str, Any] | None) -> DebateRunStatus | None:
        if not data:
            return None
        result = data.get("result")
        return DebateRunStatus(
            run_id=run_id,
            status=data["status"],
            message=data.get("message"),
            result=result,
        )


def graph_result_to_api(run_id: str, final_state: dict[str, Any]) -> DebateRunResult:
    err = final_state.get("error")
    bundle = final_state.get("dataset_bundle") or {}
    model_runs_raw = final_state.get("model_runs") or {}
    metrics_raw = final_state.get("metrics") or {}
    judge_raw = final_state.get("judge_decision") or {}

    model_runs = [
        ModelRunSummary(
            model_key=key,
            agent=m["agent"],
            model_type=m["model_type"],
            artifact_path=m["artifact_path"],
            params=m.get("params") or {},
            proposal=m.get("proposal"),
            metrics=m.get("metrics"),
            evaluation=m.get("evaluation"),
            tool_calls=m.get("tool_calls"),
        )
        for key, m in model_runs_raw.items()
    ]

    metrics_rows = []
    for key, m in metrics_raw.items():
        info = model_runs_raw.get(key, {})
        metrics_rows.append(
            MetricsRow(
                model_key=key,
                agent=str(info.get("agent", key)),
                metrics={k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in m.items()},
            )
        )

    normalized = normalize_judge_payload(judge_raw if isinstance(judge_raw, dict) else {})
    judge = JudgeDecision(
        winner=str(normalized.get("winner", "none")),
        reason=str(normalized.get("reason", "")),
        confidence=float(normalized.get("confidence", 0.0)),
        winner_agent=str(normalized.get("winner_agent", "")),
    )

    logs = [ReasoningLogEntry(**x) for x in final_state.get("reasoning_logs") or []]

    eda_struct = final_state.get("eda_structured")
    if eda_struct is not None and not isinstance(eda_struct, dict):
        eda_struct = None

    ev_rep = final_state.get("evaluation_report")
    if ev_rep is not None and not isinstance(ev_rep, dict):
        ev_rep = None

    debate_analysis = final_state.get("debate_analysis")
    if debate_analysis is not None and not isinstance(debate_analysis, dict):
        debate_analysis = None

    return DebateRunResult(
        run_id=run_id,
        status="failed" if err else "completed",
        target_column=str(bundle.get("target_column", "")),
        task_type=str(final_state.get("task_type", "")),
        eda_summary=str(final_state.get("eda_summary", "")),
        eda_structured=eda_struct,
        evaluation_report=ev_rep,
        debate_analysis=debate_analysis,
        model_runs=model_runs,
        metrics_comparison=metrics_rows,
        debate_transcript=str(final_state.get("debate_transcript", "")),
        judge=judge,
        reasoning_logs=logs,
        error=err,
    )


run_store = RunStore()
