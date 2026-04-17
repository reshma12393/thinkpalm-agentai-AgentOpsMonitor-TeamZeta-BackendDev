from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from langchain_core.tools import StructuredTool

from app.tools.ml_tools import evaluate_model, load_bundle_train_test_numpy, train_model


def _proposal_model_to_train_name(proposal: dict[str, Any]) -> str:
    m = str(proposal.get("model", "")).strip()
    mapping = {
        "RandomForest": "RandomForest",
        "XGBoost": "XGBoost",
        "LogisticRegression": "LogisticRegression",
        "Ridge": "Ridge",
    }
    if m in mapping:
        return mapping[m]
    raise ValueError(f"Unknown proposal model label: {m!r}")


def _family_tag(proposal: dict[str, Any]) -> str:
    m = str(proposal.get("model", ""))
    if m == "RandomForest":
        return "random_forest"
    if m == "XGBoost":
        return "xgboost"
    if m == "LogisticRegression":
        return "logistic_regression"
    if m == "Ridge":
        return "ridge"
    return "unknown"


def evaluation_to_pipeline_metrics(evaluation: dict[str, Any], task_type: str) -> dict[str, float | str]:
    """Flatten evaluate_model() output for debate/judge metrics dict."""
    mt = evaluation.get("metrics_test") or {}
    if task_type == "classification":
        return {
            "accuracy": float(mt["accuracy"]),
            "f1_macro": float(mt.get("f1_macro", mt.get("f1", 0.0))),
            "precision": float(mt["precision"]),
            "recall": float(mt["recall"]),
            "f1": float(mt["f1"]),
        }
    return {
        "rmse": float(mt["rmse"]),
        "mae": float(mt["mae"]),
        "r2": float(mt["r2"]),
    }


def run_proposal_with_train_eval_tools(
    bundle: dict[str, Any],
    proposal: dict[str, Any],
    model_key: str,
    agent_name: str,
) -> dict[str, Any]:
    """
    After EDA proposal: invoke tool-calling wrappers for train_model() and evaluate_model(),
    persist the fitted estimator, and attach real metrics (no synthetic scores).
    """
    task_type = str(bundle["task_type"])
    X_train, y_train, X_test, y_test = load_bundle_train_test_numpy(bundle)
    model_name = _proposal_model_to_train_name(proposal)
    params = dict(proposal["params"])

    ctx: dict[str, Any] = {}

    def _train_impl(model_name: str, params_json: str) -> str:
        p = json.loads(params_json)
        out = train_model(
            model_name,
            p,
            X_train,
            y_train,
            task_type=task_type if task_type in ("classification", "regression") else None,
        )
        ctx["train_result"] = out
        ctx["estimator"] = out["model"]
        serializable = {k: v for k, v in out.items() if k != "model"}
        return json.dumps(serializable, default=str)

    def _eval_impl(confirm: str = "run") -> str:
        est = ctx.get("estimator")
        if est is None:
            raise RuntimeError("train_model must run before evaluate_model.")
        ev = evaluate_model(est, X_test, y_test, X_train, y_train)
        ctx["evaluation"] = ev
        return json.dumps(ev, default=str)

    train_tool = StructuredTool.from_function(
        name="train_model",
        description="Fit sklearn/XGBoost estimator on training rows from the dataset bundle.",
        func=_train_impl,
    )
    eval_tool = StructuredTool.from_function(
        name="evaluate_model",
        description="Compute held-out metrics and optional train/test gap on the fitted model.",
        func=_eval_impl,
    )

    params_json = json.dumps(params, default=str)
    train_tool.invoke({"model_name": model_name, "params_json": params_json})
    eval_tool.invoke({"confirm": "run"})

    train_result = ctx["train_result"]
    evaluation = ctx["evaluation"]
    estimator = train_result["model"]

    run_dir = Path(bundle["run_dir"])
    artifact_path = run_dir / f"model_{model_key}.joblib"
    joblib.dump(estimator, artifact_path)

    train_meta = {k: v for k, v in train_result.items() if k != "model"}
    pipeline_metrics = evaluation_to_pipeline_metrics(evaluation, task_type)

    return {
        "agent": agent_name,
        "model_key": model_key,
        "model_type": _family_tag(proposal),
        "artifact_path": str(artifact_path),
        "params": params,
        "proposal": proposal,
        "train_model_output": train_meta,
        "evaluation": evaluation,
        "metrics": pipeline_metrics,
        "tool_calls": [
            {"tool": "train_model", "input": {"model_name": model_name, "params": params}},
            {"tool": "evaluate_model", "input": {"confirm": "run"}},
        ],
    }
