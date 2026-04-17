from __future__ import annotations

import math
from typing import Any


def _det_block(eda_report: dict[str, Any] | None) -> dict[str, Any]:
    if not eda_report:
        return {}
    return eda_report.get("deterministic") or {}


def propose_random_forest_agent(
    eda_report: dict[str, Any] | None, *, prior_memory: str | None = None
) -> dict[str, Any]:
    """
    RandomForest agent: proposes RF(R) hyperparameters from EDA signals.
    Output schema: model, params, reasoning (data-driven).
    """
    det = _det_block(eda_report)
    n_rows = int(det.get("n_rows") or 0) or 100
    n_features = int(det.get("n_features") or 1)
    tp = det.get("target_profile") or {}
    task = str(tp.get("task_hint") or "classification")

    mv = det.get("missing_values") or {}
    frac_miss = float(mv.get("fraction_rows_any_missing") or 0)

    corr_block = det.get("correlations") or {}
    pairs = corr_block.get("top_feature_pairs") or []
    max_abs_corr = max((abs(float(p.get("pearson", 0))) for p in pairs), default=0.0)

    ft = det.get("feature_types") or {}
    n_numeric = sum(1 for v in ft.values() if v.get("role") == "numeric")
    n_cat = sum(1 for v in ft.values() if v.get("role") == "categorical")

    imb = det.get("class_imbalance") or {}
    ratio = float(imb.get("imbalance_ratio") or 1.0)
    if math.isinf(ratio):
        ratio = 10.0

    # Trees scale with data size; cap depth on small samples to limit overfitting
    n_estimators = int(min(500, max(150, n_rows // 4)))
    max_depth = None
    if n_rows < 400:
        max_depth = 12
    if n_rows < 150:
        max_depth = 8
        n_estimators = min(n_estimators, 300)

    min_samples_leaf = 1
    if frac_miss > 0.1 or max_abs_corr > 0.92:
        min_samples_leaf = 2

    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "random_state": 42,
        "n_jobs": -1,
    }

    reasons: list[str] = []
    reasons.append(
        f"Training set has n≈{n_rows} rows and {n_features} features "
        f"({n_numeric} numeric, {n_cat} categorical), so tree ensembles can model mixed types without manual monotonic transforms."
    )
    if task == "classification" and ratio > 2.5:
        params["class_weight"] = "balanced_subsample"
        reasons.append(
            f"Class imbalance ratio≈{ratio:.2f} (EDA): using class_weight='balanced_subsample' to reduce majority bias."
        )
    if max_abs_corr > 0.85:
        reasons.append(
            f"Top feature-feature |ρ|≈{max_abs_corr:.2f}; tree splits remain stable under multicollinearity versus linear models."
        )
    if frac_miss > 0.05:
        reasons.append(
            f"~{frac_miss:.1%} rows carry missing features; RF tolerates sparse splits better than naive linear baselines."
        )
    reasons.append(
        "Bagging many randomized trees reduces variance and captures non-linear interactions among features."
    )

    reasoning = " ".join(reasons)
    if prior_memory:
        reasoning = (
            f"Prior knowledge (similar dataset patterns from vector memory): {prior_memory} || "
            + reasoning
        )

    model_name = "RandomForestClassifier" if task == "classification" else "RandomForestRegressor"
    return {
        "model": "RandomForest",
        "sklearn_estimator": model_name,
        "params": {k: v for k, v in params.items() if v is not None},
        "reasoning": reasoning,
    }


def propose_xgboost_agent(
    eda_report: dict[str, Any] | None, *, prior_memory: str | None = None
) -> dict[str, Any]:
    """XGBoost agent: gradient boosting with regularization tuned from EDA."""
    det = _det_block(eda_report)
    n_rows = int(det.get("n_rows") or 0) or 100
    tp = det.get("target_profile") or {}
    task = str(tp.get("task_hint") or "classification")

    imb = det.get("class_imbalance") or {}
    ratio = float(imb.get("imbalance_ratio") or 1.0)
    if math.isinf(ratio):
        ratio = 10.0
    n_classes = int(tp.get("n_classes") or 2)

    mv = det.get("missing_values") or {}
    frac_miss = float(mv.get("fraction_rows_any_missing") or 0)

    n_estimators = int(min(800, max(200, n_rows // 3)))
    learning_rate = 0.08 if n_rows < 800 else 0.05
    max_depth = 5 if n_rows > 2000 else 6
    if n_rows < 300:
        learning_rate = 0.1
        max_depth = 4
        n_estimators = min(n_estimators, 400)

    subsample = 0.85 if frac_miss > 0.08 else 0.9
    colsample = 0.85 if frac_miss > 0.08 else 0.9

    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    reasons: list[str] = []
    reasons.append(
        f"Sample size n≈{n_rows} supports boosted trees with learning_rate={learning_rate} and depth {max_depth} to balance bias/variance."
    )
    if task == "classification" and n_classes == 2 and ratio > 2.0:
        # scale_pos_weight ~ neg/pos
        params["scale_pos_weight"] = round(ratio, 3)
        reasons.append(
            f"Binary imbalance ratio≈{ratio:.2f}: setting scale_pos_weight≈{params['scale_pos_weight']} (EDA-driven)."
        )
    if frac_miss > 0.07:
        reasons.append(
            f"Non-trivial missingness (~{frac_miss:.1%} rows); subsample={subsample} and colsample_bytree={colsample} add stochasticity to reduce overfitting."
        )
    reasons.append(
        "XGBoost applies sequential residual fitting with L1/L2-style regularization—strong default for heterogeneous tabular data."
    )

    reasoning = " ".join(reasons)
    if prior_memory:
        reasoning = (
            f"Prior knowledge (similar dataset patterns from vector memory): {prior_memory} || "
            + reasoning
        )

    model_name = "XGBClassifier" if task == "classification" else "XGBRegressor"
    return {
        "model": "XGBoost",
        "sklearn_estimator": model_name,
        "params": params,
        "reasoning": reasoning,
    }


def propose_logistic_regression_agent(
    eda_report: dict[str, Any] | None, *, prior_memory: str | None = None
) -> dict[str, Any]:
    """
    Logistic Regression agent: classification uses LogisticRegression; regression uses Ridge
    (logistic is classification-only—justified explicitly in reasoning).
    """
    det = _det_block(eda_report)
    tp = det.get("target_profile") or {}
    task = str(tp.get("task_hint") or "classification")

    imb = det.get("class_imbalance") or {}
    ratio = float(imb.get("imbalance_ratio") or 1.0)
    if math.isinf(ratio):
        ratio = 10.0

    corr_block = det.get("correlations") or {}
    pairs = corr_block.get("top_feature_pairs") or []
    max_abs_corr = max((abs(float(p.get("pearson", 0))) for p in pairs), default=0.0)

    n_rows = int(det.get("n_rows") or 0) or 100

    if task == "regression":
        # Linear baseline for regression (LogisticRegression N/A)
        alpha = 1.0
        if max_abs_corr > 0.75:
            alpha = 5.0
        if n_rows < 200:
            alpha = max(0.5, alpha * 0.5)
        params = {"alpha": float(alpha)}
        reasoning = (
            f"EDA marks a regression target; LogisticRegression is classification-only. "
            f"Ridge (L2) with alpha={params['alpha']} is chosen instead: "
            f"top |feature–feature ρ| reaches ~{max_abs_corr:.2f}, so stronger shrinkage mitigates collinearity; "
            f"n≈{n_rows} guides regularization strength."
        )
        if prior_memory:
            reasoning = (
                f"Prior knowledge (similar dataset patterns from vector memory): {prior_memory} || "
                + reasoning
            )
        return {
            "model": "Ridge",
            "sklearn_estimator": "Ridge",
            "params": params,
            "reasoning": reasoning,
            "linear_family": "ridge_regression",
        }

    # Classification → LogisticRegression
    C = 1.0
    if max_abs_corr > 0.85:
        C = 0.3
    elif max_abs_corr < 0.25:
        C = 2.0
    if ratio > 2.5:
        cw = "balanced"
    else:
        cw = None

    params: dict[str, Any] = {
        "max_iter": 2000,
        "random_state": 42,
        "solver": "lbfgs",
        "C": float(C),
    }
    if cw:
        params["class_weight"] = cw

    reasoning_parts = [
        f"EDA task is classification with up to {int(tp.get('n_classes') or 2)} classes; "
        f"logistic regression gives calibrated linear decision boundaries on the preprocessed numeric design.",
        f"Feature–feature correlations reach ~{max_abs_corr:.2f}; C={C} controls L2 shrinkage to stabilize coefficients.",
    ]
    if ratio > 2.5:
        reasoning_parts.append(
            f"Imbalance ratio≈{ratio:.2f} → class_weight='balanced' reweights loss toward minority classes."
        )
    reasoning_parts.append(
        "Linear model complements tree ensembles: good when signal is near-linear or for interpretable coefficients."
    )

    reasoning = " ".join(reasoning_parts)
    if prior_memory:
        reasoning = (
            f"Prior knowledge (similar dataset patterns from vector memory): {prior_memory} || "
            + reasoning
        )

    return {
        "model": "LogisticRegression",
        "sklearn_estimator": "LogisticRegression",
        "params": params,
        "reasoning": reasoning,
        "linear_family": "logistic",
    }
