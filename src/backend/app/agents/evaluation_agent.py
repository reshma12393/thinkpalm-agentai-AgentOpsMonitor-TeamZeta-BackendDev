from __future__ import annotations

import math
from typing import Any


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _min_max_normalize(values: list[float], *, higher_is_better: bool) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(hi, lo):
        return [1.0 for _ in values]
    if higher_is_better:
        return [(v - lo) / (hi - lo) for v in values]
    return [(hi - v) / (hi - lo) for v in values]


def _extract_gaps(evaluation: dict[str, Any] | None) -> dict[str, float]:
    if not evaluation:
        return {}
    gap = evaluation.get("train_test_gap")
    if not isinstance(gap, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in gap.items():
        fv = _safe_float(v)
        if fv is not None:
            out[str(k)] = fv
    return out


def _gap_magnitude_classification(gaps: dict[str, float]) -> float:
    parts: list[float] = []
    for key in ("f1_macro", "f1", "accuracy", "precision", "recall"):
        if key in gaps:
            parts.append(abs(gaps[key]))
    return float(sum(parts) / len(parts)) if parts else 0.0


def _gap_magnitude_regression(gaps: dict[str, float]) -> float:
    # Positive RMSE gap (train_rmse - test_rmse negative is good) — use abs for magnitude
    parts: list[float] = []
    for key in ("rmse", "mae", "r2"):
        if key in gaps:
            parts.append(abs(gaps[key]))
    return float(sum(parts) / len(parts)) if parts else 0.0


def _overfitting_label(mag: float, task: str) -> str:
    if task == "classification":
        t_lo, t_hi = 0.02, 0.08
    else:
        t_lo, t_hi = 0.05, 0.15
    if mag <= t_lo:
        return "low"
    if mag <= t_hi:
        return "medium"
    return "high"


def build_evaluation_report(
    model_runs: dict[str, dict[str, Any]],
    task_type: str,
    metrics_by_key: dict[str, dict[str, float | str]] | None = None,
) -> dict[str, Any]:
    """
    Collect all model-agent outputs, normalize comparable metrics, compute ranking scores
    and overfitting (train vs test gap), and return a comparison table structure.
    """
    metrics_by_key = metrics_by_key or {}
    keys = list(model_runs.keys())
    if not keys:
        return {
            "task_type": task_type,
            "models": [],
            "comparison_table": [],
            "ranking": [],
            "notes": "No model runs to evaluate.",
        }

    rows: list[dict[str, Any]] = []

    for mk in keys:
        info = model_runs[mk]
        raw = dict(metrics_by_key.get(mk) or info.get("metrics") or {})
        # Coerce numeric
        raw_f: dict[str, float] = {}
        for k, v in raw.items():
            fv = _safe_float(v)
            if fv is not None:
                raw_f[k] = fv

        ev = info.get("evaluation")
        gaps = _extract_gaps(ev if isinstance(ev, dict) else None)

        if task_type == "classification":
            mag = _gap_magnitude_classification(gaps)
        else:
            mag = _gap_magnitude_regression(gaps)

        rows.append(
            {
                "model_key": mk,
                "agent": info.get("agent", ""),
                "model_type": info.get("model_type", ""),
                "raw_metrics": raw_f,
                "train_test_gap": gaps,
                "overfitting_gap_magnitude": round(mag, 6),
                "overfitting_risk": _overfitting_label(mag, task_type),
            }
        )

    # Normalize metrics across models
    if task_type == "classification":
        metric_keys = ["f1_macro", "accuracy", "f1", "precision", "recall"]
        primary_for_rank = ("f1_macro", "accuracy")
    else:
        metric_keys = ["r2", "rmse", "mae"]
        primary_for_rank = ("r2", "rmse")

    for row in rows:
        row["normalized"] = {}

    for mk in metric_keys:
        vals: list[float] = []
        idx_map: list[int] = []
        for i, row in enumerate(rows):
            v = row["raw_metrics"].get(mk)
            if v is not None:
                vals.append(v)
                idx_map.append(i)
        if len(vals) < 2:
            for i, row in enumerate(rows):
                v = row["raw_metrics"].get(mk)
                if v is not None:
                    row["normalized"][mk] = 1.0
                else:
                    row["normalized"][mk] = None
            continue
        higher = mk != "rmse" and mk != "mae"
        normed = _min_max_normalize(vals, higher_is_better=higher)
        for j, idx in enumerate(idx_map):
            rows[idx]["normalized"][mk] = round(float(normed[j]), 6)

    # Ranking score + normalize overfitting penalty (lower gap magnitude = better)
    mag_values = [float(r["overfitting_gap_magnitude"]) for r in rows]
    norm_mag = _min_max_normalize(mag_values, higher_is_better=False) if mag_values else []

    for i, row in enumerate(rows):
        n_m = norm_mag[i] if i < len(norm_mag) else 1.0
        if task_type == "classification":
            nf1 = row["normalized"].get("f1_macro") or row["normalized"].get("f1")
            nacc = row["normalized"].get("accuracy")
            if nf1 is None:
                nf1 = 0.0
            if nacc is None:
                nacc = 0.0
            # Penalize overfitting: weight quality 0.7, stability 0.3
            row["ranking_score"] = round(0.35 * nf1 + 0.35 * float(nacc) + 0.3 * n_m, 6)
        else:
            nr2 = row["normalized"].get("r2")
            nrmse = row["normalized"].get("rmse")
            if nr2 is None:
                nr2 = 0.0
            if nrmse is None:
                nrmse = 0.0
            row["ranking_score"] = round(0.4 * float(nr2) + 0.4 * float(nrmse) + 0.2 * n_m, 6)

    # Sort by ranking_score desc
    ranked = sorted(range(len(rows)), key=lambda i: rows[i]["ranking_score"], reverse=True)
    for rank, idx in enumerate(ranked, start=1):
        rows[idx]["rank"] = rank

    comparison_table = sorted(rows, key=lambda r: r["ranking_score"], reverse=True)

    return {
        "task_type": task_type,
        "primary_metrics_for_ranking": list(primary_for_rank),
        "normalization": "min_max per metric column; RMSE/MAE inverted so higher is better",
        "ranking_score_formula": (
            "classification: 0.35*n(f1_macro)+0.35*n(accuracy)+0.3*n(inverted gap magnitude); "
            "regression: 0.4*n(r2)+0.4*n(inverted rmse)+0.2*n(inverted gap magnitude)"
        ),
        "models": rows,
        "comparison_table": comparison_table,
        "ranking": [{"rank": r["rank"], "model_key": r["model_key"], "ranking_score": r["ranking_score"]} for r in comparison_table],
    }

