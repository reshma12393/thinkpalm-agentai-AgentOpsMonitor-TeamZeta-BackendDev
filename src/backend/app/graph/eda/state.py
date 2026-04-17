from __future__ import annotations

from typing import Any, TypedDict


class EDAAgentState(TypedDict, total=False):
    """State for the standalone EDA LangGraph."""

    csv_path: str
    target_column: str
    run_id: str

    deterministic: dict[str, Any]
    llm_reasoning: dict[str, Any]
    eda_report_json: dict[str, Any]

    error: str | None
