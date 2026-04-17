"""Serialize post-run LangGraph state into a JSON-safe snapshot for the Agent trace UI panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.graph.workflow import PIPELINE_NODE_ORDER


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return str(value)


def graph_state_to_agent_trace(final: dict[str, Any]) -> dict[str, Any]:
    """Compact pipeline graph state for collapsible 'Agent trace' (avoids duplicating full EDA JSON)."""
    bundle_raw = final.get("dataset_bundle") or {}
    bundle: dict[str, Any] = _json_safe(bundle_raw) if isinstance(bundle_raw, dict) else {}
    if isinstance(bundle, dict):
        p = bundle.get("csv_path")
        if isinstance(p, str) and p:
            bundle["csv_path"] = Path(p).name

    mem = final.get("memory_context") or ""
    mem_preview = mem if isinstance(mem, str) else str(mem)
    if len(mem_preview) > 4000:
        mem_preview = mem_preview[:4000] + "…"

    runs_raw = final.get("model_runs") or {}
    model_runs: dict[str, Any] = {}
    if isinstance(runs_raw, dict):
        for key, v in runs_raw.items():
            if not isinstance(v, dict):
                continue
            row = _json_safe({k: val for k, val in v.items() if k != "artifact_path"})
            if not isinstance(row, dict):
                continue
            ap = v.get("artifact_path")
            if isinstance(ap, str) and ap:
                row["artifact_basename"] = Path(ap).name
            model_runs[str(key)] = row

    eda = final.get("eda_structured")
    eda_meta: dict[str, Any] = {}
    if isinstance(eda, dict):
        eda_meta["keys"] = list(eda.keys())
        if "schema_version" in eda:
            eda_meta["schema_version"] = eda.get("schema_version")

    return {
        "pipeline_nodes": list(PIPELINE_NODE_ORDER),
        "run_id": final.get("run_id"),
        "task_type": final.get("task_type"),
        "target_column": bundle.get("target_column") if isinstance(bundle, dict) else final.get("target_column"),
        "error": final.get("error"),
        "dataset_bundle": bundle,
        "memory_context_preview": mem_preview,
        "memory_hits": _json_safe(final.get("memory_hits") or []),
        "model_proposals": _json_safe(final.get("model_proposals") or {}),
        "model_runs": model_runs,
        "metrics": _json_safe(final.get("metrics") or {}),
        "evaluation_report": _json_safe(final.get("evaluation_report") or {}),
        "debate_analysis": _json_safe(final.get("debate_analysis"))
        if final.get("debate_analysis") is not None
        else None,
        "judge_decision": _json_safe(final.get("judge_decision") or {}),
        "eda_structured_meta": eda_meta,
        "reasoning_log_count": len(final.get("reasoning_logs") or []),
    }
