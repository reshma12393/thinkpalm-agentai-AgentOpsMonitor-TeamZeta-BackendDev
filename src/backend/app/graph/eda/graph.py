from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.graph.eda.deterministic import compute_deterministic_eda
from app.graph.eda.state import EDAAgentState
from app.agents.llm_util import get_chat_model
from app.config import settings
from app.services.memory_service import MemoryService


def _heuristic_llm(deterministic: dict[str, Any]) -> dict[str, Any]:
    bullets: list[str] = []
    risks: list[str] = []
    hints: list[str] = []

    mv = deterministic.get("missing_values", {})
    frac = float(mv.get("fraction_rows_any_missing", 0) or 0)
    if frac > 0.05:
        risks.append(f"~{frac:.1%} of rows have at least one missing feature; impute or drop with care.")
    else:
        bullets.append("Feature missingness is low overall.")

    ci = deterministic.get("class_imbalance")
    if ci and isinstance(ci, dict):
        ratio = ci.get("imbalance_ratio")
        if ratio is not None and ratio != float("inf") and float(ratio) > 3:
            risks.append(f"Class imbalance ratio ~{float(ratio):.2f}; use macro-F1 / balanced metrics.")
            hints.append("Consider class_weight or stratified CV.")
        else:
            bullets.append("Class distribution is relatively balanced or moderately skewed.")

    tp = deterministic.get("target_profile", {})
    if tp.get("task_hint") == "regression":
        hints.append("Regression: check outliers on target and heavy-tailed errors.")

    cors = deterministic.get("correlations", {}).get("top_feature_pairs") or []
    if cors and abs(cors[0].get("pearson", 0)) > 0.9:
        risks.append("Very high feature-feature correlation suggests multicollinearity / redundancy.")

    if not bullets:
        bullets.append("Review correlations and missingness columns for downstream leakage.")
    if not hints:
        hints.append("Tree models are a strong baseline for mixed tabular features.")

    return {
        "summary_bullets": bullets[:5],
        "risks": risks[:4],
        "modeling_hints": hints[:4],
    }


def _parse_llm_json(text: str) -> dict[str, Any]:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {}


def node_load_and_deterministic(state: EDAAgentState) -> dict[str, Any]:
    path = Path(state["csv_path"])
    target = state["target_column"]
    if not path.is_file():
        return {"error": f"CSV not found: {path}"}
    try:
        df = pd.read_csv(path)
        det = compute_deterministic_eda(df, target)
    except Exception as e:
        return {"error": str(e)}
    return {"deterministic": det, "error": None}


def node_llm_reasoning(state: EDAAgentState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    det = state.get("deterministic") or {}
    llm = get_chat_model(temperature=0.15)
    if llm:
        mx = max(2000, settings.openrouter_max_llm_input_chars)
        payload = json.dumps(det, default=str, indent=2)[:mx]
        msg = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a senior ML data analyst. Given ONLY the JSON statistics below, "
                        "respond with a single JSON object (no markdown) with keys: "
                        "summary_bullets (array of up to 5 short strings), "
                        "risks (up to 4 strings: data quality, leakage, spurious correlation), "
                        "modeling_hints (up to 4 strings). "
                        "Be concise; no repetition of raw numbers unless needed."
                    )
                ),
                HumanMessage(content=payload),
            ]
        )
        parsed = _parse_llm_json(str(msg.content))
        if not parsed or not isinstance(parsed, dict):
            out = _heuristic_llm(det)
        else:
            out = {
                "summary_bullets": list(parsed.get("summary_bullets", []))[:5],
                "risks": list(parsed.get("risks", []))[:4],
                "modeling_hints": list(parsed.get("modeling_hints", []))[:4],
            }
    else:
        out = _heuristic_llm(det)
    return {"llm_reasoning": out}


def node_merge_json(state: EDAAgentState) -> dict[str, Any]:
    if state.get("error"):
        return {
            "eda_report_json": {
                "schema_version": "1.0",
                "error": state.get("error"),
                "deterministic": None,
                "llm_reasoning": None,
            }
        }
    det = state.get("deterministic") or {}
    llm = state.get("llm_reasoning") or {}
    report = {
        "schema_version": "1.0",
        "deterministic": det,
        "llm_reasoning": llm,
    }
    run_id = state.get("run_id")
    if run_id:
        try:
            mem = MemoryService()
            mem.add_texts(
                [json.dumps(report, default=str)[:8000]],
                metadatas=[{"run_id": run_id, "agent": "eda_agent", "kind": "eda_json"}],
            )
        except Exception:
            pass
    return {"eda_report_json": report}


def _route_after_load(state: EDAAgentState) -> str:
    if state.get("error"):
        return "merge"
    return "llm"


def build_eda_agent_graph() -> Any:
    """LangGraph: load CSV → deterministic EDA (pandas) → LLM JSON reasoning → merged structured report."""
    g = StateGraph(EDAAgentState)
    g.add_node("deterministic", node_load_and_deterministic)
    g.add_node("llm", node_llm_reasoning)
    g.add_node("merge", node_merge_json)

    g.add_edge(START, "deterministic")
    g.add_conditional_edges("deterministic", _route_after_load, {"llm": "llm", "merge": "merge"})
    g.add_edge("llm", "merge")
    g.add_edge("merge", END)
    return g.compile()
