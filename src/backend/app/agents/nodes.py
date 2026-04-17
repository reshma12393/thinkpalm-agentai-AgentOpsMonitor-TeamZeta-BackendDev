from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.eda.graph import build_eda_agent_graph
from app.agents.llm_util import get_chat_model
from app.agents.debate_agent import (
    build_debate_analysis,
    format_debate_transcript,
    merge_transcript_with_llm_narrative,
)
from app.agents.evaluation_agent import build_evaluation_report
from app.agents.judge_agent import build_judge_decision, normalize_judge_payload
from app.agents.model_agent_tools import run_proposal_with_train_eval_tools
from app.agents.model_proposals import (
    propose_logistic_regression_agent,
    propose_random_forest_agent,
    propose_xgboost_agent,
)
from app.config import settings
from app.services.dataset import build_dataset_bundle
from app.services.dataset_memory import DatasetPatternMemory, format_priors_for_model_agents
from app.services.memory_service import MemoryService
from app.state import DebateGraphState
from app.tools.ml_tools import build_training_toolkit


def _log(agent: str, step: str, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "reasoning_logs": [
            {
                "agent": agent,
                "step": step,
                "content": content,
                "metadata": metadata or {},
            }
        ]
    }


def _bundle(state: DebateGraphState) -> dict[str, Any]:
    b = state.get("dataset_bundle")
    if not b:
        raise RuntimeError("dataset_bundle missing")
    return b


def node_prepare_dataset(state: DebateGraphState) -> dict[str, Any]:
    run_id = state.get("run_id") or str(uuid.uuid4())
    run_dir = settings.runs_dir / run_id
    try:
        bundle = build_dataset_bundle(
            csv_path=Path(state["csv_path"]),
            target_column=state["target_column"],
            run_id=run_id,
            run_dir=run_dir,
        )
    except Exception as e:
        return {"error": str(e), **_log("system", "dataset", f"Dataset build failed: {e}")}

    d = bundle.to_state_dict()
    return {
        "run_id": run_id,
        "dataset_bundle": d,
        "task_type": bundle.task_type,
        **_log(
            "system",
            "dataset",
            f"Train/test split materialized at {run_dir}; task={bundle.task_type}, "
            f"features={len(bundle.feature_columns)} (num={len(bundle.numeric_features)}, "
            f"cat={len(bundle.categorical_features)}).",
            {"run_dir": str(run_dir)},
        ),
    }


def node_eda_agent(state: DebateGraphState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    run_id = state["run_id"]
    eda_g = build_eda_agent_graph()
    eda_out = eda_g.invoke(
        {
            "csv_path": state["csv_path"],
            "target_column": state["target_column"],
            "run_id": run_id,
        }
    )
    report = eda_out.get("eda_report_json") or {}
    text = json.dumps(report, indent=2, default=str)
    return {
        "eda_structured": report,
        "eda_summary": text[:50000],
        **_log("eda_agent", "eda_langgraph", text[:4000], {"chars": len(text)}),
    }


def node_memory_retrieve(state: DebateGraphState) -> dict[str, Any]:
    """
    Retrieve similar past dataset runs from Chroma (dataset characteristics + outcomes) and
    expose a short prior-knowledge string for model proposal agents.
    """
    if state.get("error"):
        return {}
    task = state.get("task_type") or "classification"
    eda = state.get("eda_structured") if isinstance(state.get("eda_structured"), dict) else None
    rid = str(state.get("run_id") or "")
    try:
        mem = DatasetPatternMemory()
        hits = mem.find_similar_dataset_patterns(eda, task, k=5, exclude_run_id=rid)
        ctx = format_priors_for_model_agents(hits)
        hit_summaries = [
            {
                "run_id": (h.metadata or {}).get("run_id"),
                "winner": (h.metadata or {}).get("winner"),
                "distance": h.distance,
            }
            for h in hits[:5]
        ]
    except Exception as e:
        ctx = ""
        hit_summaries = []
        return {
            "memory_context": "",
            "memory_hits": [],
            **_log(
                "memory_agent",
                "dataset_patterns",
                f"Retrieval skipped or failed: {e}",
                {"error": str(e)},
            ),
        }

    return {
        "memory_context": ctx,
        "memory_hits": hit_summaries,
        **_log(
            "memory_agent",
            "dataset_patterns",
            f"Retrieved {len(hit_summaries)} similar pattern(s). Context chars={len(ctx)}.",
            {"hits": hit_summaries},
        ),
    }


def node_model_agent_rf(state: DebateGraphState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    eda = state.get("eda_structured")
    prior = (state.get("memory_context") or "").strip() or None
    proposal = propose_random_forest_agent(eda, prior_memory=prior)
    b = _bundle(state)
    meta = run_proposal_with_train_eval_tools(
        b,
        proposal,
        model_key="rf",
        agent_name="model_agent_random_forest",
    )
    log_payload = json.dumps(
        {"proposal": proposal, "metrics": meta.get("metrics"), "evaluation": meta.get("evaluation")},
        default=str,
    )
    return {
        "model_proposals": {"rf": proposal},
        "model_runs": {"rf": meta},
        **_log("model_agent_random_forest", "propose_train_eval_tools", log_payload[:8000]),
    }


def node_model_agent_xgb(state: DebateGraphState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    eda = state.get("eda_structured")
    prior = (state.get("memory_context") or "").strip() or None
    proposal = propose_xgboost_agent(eda, prior_memory=prior)
    b = _bundle(state)
    meta = run_proposal_with_train_eval_tools(
        b,
        proposal,
        model_key="xgb",
        agent_name="model_agent_xgboost",
    )
    log_payload = json.dumps(
        {"proposal": proposal, "metrics": meta.get("metrics"), "evaluation": meta.get("evaluation")},
        default=str,
    )
    return {
        "model_proposals": {"xgb": proposal},
        "model_runs": {"xgb": meta},
        **_log("model_agent_xgboost", "propose_train_eval_tools", log_payload[:8000]),
    }


def node_model_agent_lr(state: DebateGraphState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    eda = state.get("eda_structured")
    prior = (state.get("memory_context") or "").strip() or None
    proposal = propose_logistic_regression_agent(eda, prior_memory=prior)
    b = _bundle(state)
    meta = run_proposal_with_train_eval_tools(
        b,
        proposal,
        model_key="lr",
        agent_name="model_agent_logistic_regression",
    )
    log_payload = json.dumps(
        {"proposal": proposal, "metrics": meta.get("metrics"), "evaluation": meta.get("evaluation")},
        default=str,
    )
    return {
        "model_proposals": {"lr": proposal},
        "model_runs": {"lr": meta},
        **_log("model_agent_logistic_regression", "propose_train_eval_tools", log_payload[:8000]),
    }


def node_evaluation_agent(state: DebateGraphState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    b = _bundle(state)
    runs = state.get("model_runs") or {}
    toolkit = build_training_toolkit(b)
    eval_tool = toolkit["evaluate_artifact"]
    metrics: dict[str, dict[str, float | str]] = {}
    lines: list[str] = []
    for key, info in runs.items():
        if info.get("metrics") is not None:
            m = info["metrics"]
            metrics[key] = m  # type: ignore[assignment]
            lines.append(f"{key} (from evaluate_model tool): {json.dumps(m)}")
            continue
        raw = eval_tool.invoke({"model_key": key, "artifact_path": info["artifact_path"]})
        payload = json.loads(str(raw))
        m = payload["metrics"]
        metrics[key] = m  # type: ignore[assignment]
        lines.append(f"{key}: {json.dumps(m)}")

    task = state.get("task_type") or "classification"
    evaluation_report = build_evaluation_report(runs, task, metrics)
    reasoning = (
        "Evaluation agent: collected metrics; normalized columns; ranking_score and overfitting gaps.\n"
        + "\n".join(lines)
        + "\n\nRanking:\n"
        + json.dumps(evaluation_report.get("ranking", []), indent=2)
    )
    return {
        "metrics": metrics,  # type: ignore[typeddict-item]
        "evaluation_report": evaluation_report,
        **_log("evaluation_agent", "evaluate", reasoning[:8000]),
    }


def node_debate_agent(state: DebateGraphState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    mem = MemoryService()
    run_id = state["run_id"]
    task = state["task_type"] or "classification"
    metrics = state.get("metrics") or {}
    eda = state.get("eda_summary") or ""
    eda_struct = state.get("eda_structured") if isinstance(state.get("eda_structured"), dict) else {}
    runs = state.get("model_runs") or {}
    er = state.get("evaluation_report") or {}

    debate_analysis = build_debate_analysis(
        runs,
        metrics,
        er if isinstance(er, dict) else {},
        task,
        eda_struct,
    )
    base_transcript = format_debate_transcript(debate_analysis)

    ctx_hits = mem.similarity_search_with_run("model performance and data issues", run_id, k=4)
    mx = max(2000, settings.openrouter_max_llm_input_chars)
    ctx = "\n".join(h.page_content for h in ctx_hits)[: min(3500, mx // 3)]

    rank_summary = er.get("ranking") if isinstance(er, dict) else []

    llm = get_chat_model(temperature=0.15)
    if llm:
        prompt = (
            f"Task type: {task}\n\n"
            "You are given evaluation results for multiple models in the JSON below. Each model may include "
            "holdout metrics (e.g. for classification: accuracy, precision, recall, f1 / f1_macro, train vs test; "
            "for regression: r2, rmse, mae, train vs test as present).\n\n"
            "Perform a strict, data-driven comparison of models.\n\n"
            "Rules:\n"
            "1. You MUST reference actual numeric metrics in every bullet (no bullet without numbers).\n"
            "2. You MUST explicitly call out when supported by the data:\n"
            "   - overfitting (large train–test gap; use train vs test and overfitting_risk / gap fields when present)\n"
            "   - underfitting (low overall performance vs peers on the same split)\n"
            "   - imbalance issues for classification (precision vs recall spread, macro vs weighted F1 when both exist)\n"
            "3. DO NOT write generic phrases like 'performs well' or 'strong model' without citing metrics.\n"
            "4. DO NOT summarize without justification — every claim needs metric evidence from the payload.\n"
            "5. Compare models relative to each other (name the better/worse peer and quantify the gap).\n\n"
            "Output format:\n"
            "- Bullet points only (no long prose blocks).\n"
            "- Each bullet MUST include: model name (exact key from metrics) · issue or strength · supporting metrics.\n"
            "- Style examples (use only if your numbers match):\n"
            '  - "random_forest achieves f1_macro=0.81 vs logistic_regression f1_macro=0.72 (Δ=0.09), indicating stronger holdout discrimination on this split."\n'
            '  - "xgboost shows overfitting: train f1_macro=0.97 vs test f1_macro=0.83 (gap=0.14)."\n'
            "- After the bullets, add exactly two lines:\n"
            "  Key disagreements: <one line citing metrics>\n"
            "  Consensus: <one line citing metrics>\n\n"
            "Use ONLY numbers and facts from debate_analysis, Holdout metrics JSON, Evaluation ranking/table, and "
            "model run evaluation blocks embedded in debate_analysis. EDA and memory are secondary context — do not "
            "infer metrics not present.\n\n"
            f"debate_analysis:\n{json.dumps(debate_analysis, default=str)[:mx]}\n\n"
            f"Holdout metrics JSON:\n{json.dumps(metrics)}\n\n"
            f"Evaluation ranking:\n{json.dumps(rank_summary)}\n\n"
            f"EDA excerpt (context only; prefer structured numbers above):\n{eda[: min(2500, mx // 3)]}\n\n"
            f"Memory snippets:\n{ctx}"
        )
        out = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a senior machine learning reviewer performing a critical comparison of multiple models. "
                        "Be precise, analytical, and critical. Every statement must be justified with cited metrics from "
                        "the user payload; compare models relative to each other, not in isolation."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        transcript = merge_transcript_with_llm_narrative(base_transcript, str(out.content))
    else:
        transcript = base_transcript

    mem.add_texts(
        [transcript[:8000]],
        metadatas=[{"run_id": run_id, "agent": "debate_agent", "kind": "debate"}],
    )

    reasoning_payload = json.dumps(
        {"debate_analysis": debate_analysis, "transcript_preview": transcript[:4000]},
        default=str,
    )
    return {
        "debate_transcript": transcript,
        "debate_analysis": debate_analysis,
        **_log("debate_agent", "debate", reasoning_payload[:8000]),
    }


def node_judge_agent(state: DebateGraphState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    task = state["task_type"] or "classification"
    metrics = state.get("metrics") or {}
    debate = state.get("debate_transcript") or ""
    er = state.get("evaluation_report") if isinstance(state.get("evaluation_report"), dict) else {}
    da = state.get("debate_analysis") if isinstance(state.get("debate_analysis"), dict) else {}

    heuristic = build_judge_decision(task, metrics, er, da)
    valid_keys = set(metrics.keys())

    llm = get_chat_model(temperature=0.0)
    if llm:
        mx = max(2000, settings.openrouter_max_llm_input_chars)
        prompt = (
            f"Task type: {task}\n\n"
            "Input (use only these sources for numbers):\n"
            f"- Holdout metrics by model: {json.dumps(metrics)}\n"
            f"- Evaluation report ranking: {json.dumps(er.get('ranking', []))}\n"
            f"- Comparison table (train_test_gap, overfitting_gap_magnitude, overfitting_risk per model): "
            f"{json.dumps(er.get('comparison_table', []))[: min(5000, mx // 2)]}\n"
            f"- Debate analysis (structured strengths/weaknesses): {json.dumps(da)[: min(6000, mx // 2)]}\n"
            f"- Debate transcript excerpt: {debate[: min(3500, mx // 3)]}\n\n"
            "You are the Judge Agent: a senior ML system architect selecting the best model from objective evidence.\n\n"
            "Decision criteria:\n"
            "1. Primary metric: for classification, prefer higher holdout F1 (use f1_macro if present, else f1). "
            "For regression, prefer higher R² and lower RMSE (state both if available).\n"
            "2. Penalize overfitting: if the primary train–test gap (e.g. |train f1_macro − test f1_macro| from "
            "comparison_table.train_test_gap, or overfitting_gap_magnitude) is greater than 0.10, you MUST lower "
            "confidence substantially versus a model with gap ≤ 0.10, even if F1 is slightly lower.\n"
            "3. Prefer models with balanced precision and recall when those metrics exist (cite |precision − recall|).\n"
            "4. Prefer stable generalization: smaller overfitting_gap_magnitude and lower overfitting_risk vs peers.\n"
            "5. Use debate_analysis to corroborate overfitting/underperformance only when it repeats the same numeric "
            "patterns — do not contradict the tables.\n\n"
            "Rules:\n"
            "- You MUST justify the winner using concrete metrics (no vague language).\n"
            "- reason MUST explicitly cite at least: (a) F1 (f1_macro or f1) for classification, or R² and RMSE for "
            "regression; (b) generalization (train–test gap and/or overfitting_gap_magnitude).\n"
            "- winner must be exactly one of these model keys: "
            f"{sorted(valid_keys)} (use the key string as stored in metrics, e.g. \"rf\", not a display name).\n\n"
            "Output STRICT JSON only, no markdown, no code fences:\n"
            '{"winner":"<model_key>","reason":"<metric-based explanation>","confidence":0.91}\n'
            "confidence is a float in [0,1]: lower it when gaps > 0.1, close F1 races, or conflicting metrics.\n\n"
            "Example shape (illustrative keys/numbers — yours must match the payload):\n"
            '{"winner":"rf","reason":"Highest f1_macro=0.82 with train–test |Δf1_macro|=0.05 vs xgb |Δ|=0.14; '
            'precision 0.80 vs recall 0.84 (|Δ|=0.04) indicates balanced tradeoff and stronger generalization.",'
            '"confidence":0.91}'
        )
        out = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a senior ML system architect selecting the best model based on evaluation metrics "
                        "and debate analysis. Be objective: every clause in reason must tie to numbers from the user "
                        "payload. Do not use vague reasoning."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        text = str(out.content)
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            parsed = json.loads(text[start:end])
            decision = normalize_judge_payload(parsed if isinstance(parsed, dict) else {})
        except Exception:
            decision = heuristic
        w = str(decision.get("winner", ""))
        if w not in valid_keys and w != "none":
            decision = heuristic
    else:
        decision = heuristic

    return {
        "judge_decision": decision,
        **_log("judge_agent", "decide", json.dumps(decision)),
    }
