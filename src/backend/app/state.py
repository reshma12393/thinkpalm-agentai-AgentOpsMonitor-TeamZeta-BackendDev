from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _merge_dict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    out.update(b)
    return out


class DebateGraphState(TypedDict, total=False):
    run_id: str
    csv_path: str
    target_column: str
    task_type: str  # "classification" | "regression"

    # Dataset service output paths / ids
    dataset_bundle: dict[str, Any]

    eda_summary: str
    eda_structured: dict[str, Any]  # full EDA JSON (deterministic + llm_reasoning)

    # Chroma dataset-pattern memory (similar past runs → text priors for model agents)
    memory_context: str
    memory_hits: list[dict[str, Any]]

    # rf | xgb | lr -> proposal JSON from model agents
    model_proposals: Annotated[dict[str, dict[str, Any]], _merge_dict]

    # model_key -> info
    model_runs: Annotated[dict[str, dict[str, Any]], _merge_dict]

    # model_key -> metrics dict
    metrics: Annotated[dict[str, dict[str, float]], _merge_dict]

    # Evaluation agent: normalized metrics, ranking, overfitting, comparison table
    evaluation_report: dict[str, Any]

    debate_transcript: str
    debate_analysis: dict[str, Any]
    judge_decision: dict[str, Any]

    reasoning_logs: Annotated[list[dict[str, Any]], operator.add]

    error: str | None
