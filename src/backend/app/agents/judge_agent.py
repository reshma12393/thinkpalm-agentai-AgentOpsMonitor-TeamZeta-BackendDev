from __future__ import annotations

import math
from typing import Any


AGENT_BY_MODEL_KEY: dict[str, str] = {
    "rf": "model_agent_random_forest",
    "xgb": "model_agent_xgboost",
    "lr": "model_agent_logistic_regression",
}


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _min_max(values: list[float], *, higher_is_better: bool) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(hi, lo):
        return [1.0 for _ in values]
    if higher_is_better:
        return [(v - lo) / (hi - lo) for v in values]
    return [(hi - v) / (hi - lo) for v in values]


def _row_by_key(evaluation_report: dict[str, Any], mk: str) -> dict[str, Any] | None:
    for row in evaluation_report.get("comparison_table") or []:
        if isinstance(row, dict) and row.get("model_key") == mk:
            return row
    return None


def _primary_train_test_gap_abs(row: dict[str, Any] | None, task_type: str) -> float | None:
    """Absolute gap on the primary metric (F1 for classification; RMSE/R² for regression), else table magnitude."""
    if not row:
        return None
    gaps = row.get("train_test_gap") if isinstance(row.get("train_test_gap"), dict) else {}
    if task_type == "classification":
        for k in ("f1_macro", "f1"):
            v = _safe_float(gaps.get(k))
            if v is not None:
                return abs(v)
    else:
        for k in ("rmse", "r2"):
            v = _safe_float(gaps.get(k))
            if v is not None:
                return abs(v)
    return _safe_float(row.get("overfitting_gap_magnitude"))


def _debate_score_from_block(block: dict[str, Any]) -> float:
    """Map debate strengths/weaknesses to [0, 1]; grounded on substring cues from debate_agent."""
    strengths = block.get("strengths") or []
    weaknesses = block.get("weaknesses") or []
    if not isinstance(strengths, list):
        strengths = []
    if not isinstance(weaknesses, list):
        weaknesses = []
    adj = 0.0
    for w in weaknesses:
        ws = str(w).lower()
        if "overfitting signal" in ws:
            adj -= 0.35
        if "underperforms on holdout" in ws:
            adj -= 0.12
        if "weakest holdout" in ws and "r2" in ws:
            adj -= 0.12
    for s in strengths:
        ss = str(s).lower()
        if "best holdout f1_macro" in ss or "lowest holdout rmse" in ss:
            adj += 0.22
        elif "tightest train" in ss and "gap magnitude" in ss:
            adj += 0.12
        elif "lowest holdout rmse among" in ss:
            adj += 0.22
    return max(0.0, min(1.0, 0.5 + adj))


def build_judge_decision(
    task_type: str,
    metrics: dict[str, dict[str, float | str]],
    evaluation_report: dict[str, Any],
    debate_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Select best model using holdout F1 (classification) or R²/RMSE (regression), generalization gaps,
    precision–recall balance when available, and debate signals. Confidence is reduced when primary
    train–test gap exceeds 0.1 (overfitting).
    Returns: {"winner", "reason", "confidence"} with confidence in [0, 1].
    """
    keys = list(metrics.keys())
    if not keys:
        return {
            "winner": "none",
            "reason": "No holdout metrics were available to score models.",
            "confidence": 0.0,
            "winner_agent": "none",
        }

    er = evaluation_report if isinstance(evaluation_report, dict) else {}
    da = debate_analysis if isinstance(debate_analysis, dict) else {}
    per_model = {str(b.get("model_key")): b for b in (da.get("per_model") or []) if isinstance(b, dict)}

    if task_type == "classification":
        f1s: list[float] = []
        for mk in keys:
            m = metrics[mk]
            v = _safe_float(m.get("f1_macro"))
            if v is None:
                v = _safe_float(m.get("f1"))
            f1s.append(float(v) if v is not None else 0.0)
        nf1 = _min_max(f1s, higher_is_better=True)

        gaps: list[float] = []
        for mk in keys:
            row = _row_by_key(er, mk)
            g = _safe_float(row.get("overfitting_gap_magnitude")) if row else None
            gaps.append(float(g) if g is not None else 0.0)
        ngap = _min_max(gaps, higher_is_better=False)  # lower gap magnitude -> higher score

        spreads: list[float] = []
        for mk in keys:
            m = metrics[mk]
            p = _safe_float(m.get("precision"))
            r = _safe_float(m.get("recall"))
            if p is not None and r is not None:
                spreads.append(abs(p - r))
            else:
                spreads.append(1.0)
        nspread = _min_max(spreads, higher_is_better=False)

        scores: list[float] = []
        for i, mk in enumerate(keys):
            blk = per_model.get(mk, {})
            ds = _debate_score_from_block(blk if isinstance(blk, dict) else {})
            scores.append(0.48 * nf1[i] + 0.27 * ngap[i] + 0.15 * nspread[i] + 0.10 * ds)

        best_idx = max(range(len(keys)), key=lambda i: scores[i])
        sorted_s = sorted(scores, reverse=True)
        second = sorted_s[1] if len(sorted_s) > 1 else sorted_s[0]
        margin = scores[best_idx] - second
        winner = keys[best_idx]
        f1v = f1s[best_idx]
        gapv = gaps[best_idx]
        conf = _confidence_from_margin(margin, len(keys))
        pri_gap = _primary_train_test_gap_abs(_row_by_key(er, winner), "classification")
        if pri_gap is not None and pri_gap > 0.1:
            conf = max(0.22, conf - 0.18)

        pw = _safe_float(metrics[winner].get("precision"))
        rw = _safe_float(metrics[winner].get("recall"))
        pr_line = ""
        if pw is not None and rw is not None:
            pr_line = f" precision={pw:.4f}, recall={rw:.4f}, |precision−recall|={abs(pw - rw):.4f}."
        reason = (
            f"Winner {winner}: holdout f1_macro={f1v:.4f}; "
            f"generalization overfitting_gap_magnitude={gapv:.4f}"
            + (f", primary |train−test| on F1={pri_gap:.4f}" if pri_gap is not None else "")
            + f".{pr_line} "
            f"Chosen vs peers on the same split (composite {scores[best_idx]:.3f} "
            f"weights F1, gap, precision–recall balance, debate)."
        )
    else:
        r2s: list[float] = []
        rmses: list[float] = []
        for mk in keys:
            m = metrics[mk]
            r2s.append(float(_safe_float(m.get("r2")) or 0.0))
            rmses.append(float(_safe_float(m.get("rmse")) or 1e9))
        nr2 = _min_max(r2s, higher_is_better=True)
        nrmse = _min_max(rmses, higher_is_better=False)

        gaps = []
        for mk in keys:
            row = _row_by_key(er, mk)
            g = _safe_float(row.get("overfitting_gap_magnitude")) if row else None
            gaps.append(float(g) if g is not None else 0.0)
        ngap = _min_max(gaps, higher_is_better=False)

        scores = []
        for i, mk in enumerate(keys):
            blk = per_model.get(mk, {})
            ds = _debate_score_from_block(blk if isinstance(blk, dict) else {})
            scores.append(0.35 * nr2[i] + 0.35 * nrmse[i] + 0.25 * ngap[i] + 0.05 * ds)

        best_idx = max(range(len(keys)), key=lambda i: scores[i])
        sorted_s = sorted(scores, reverse=True)
        second = sorted_s[1] if len(sorted_s) > 1 else sorted_s[0]
        margin = scores[best_idx] - second
        winner = keys[best_idx]
        conf = _confidence_from_margin(margin, len(keys))
        pri_gap = _primary_train_test_gap_abs(_row_by_key(er, winner), "regression")
        if pri_gap is not None and pri_gap > 0.1:
            conf = max(0.22, conf - 0.18)
        reason = (
            f"Winner {winner} (regression): holdout R²={r2s[best_idx]:.4f}, RMSE={rmses[best_idx]:.6f}; "
            f"generalization overfitting_gap_magnitude={gaps[best_idx]:.4f}"
            + (f", primary |train−test| metric gap={pri_gap:.4f}" if pri_gap is not None else "")
            + f". Composite {scores[best_idx]:.3f} (R², RMSE, gap magnitude, debate)."
        )

    agent = AGENT_BY_MODEL_KEY.get(winner, "unknown")
    out: dict[str, Any] = {
        "winner": winner,
        "reason": reason,
        "confidence": round(conf, 3),
        "winner_agent": agent,
    }
    return out


def _confidence_from_margin(margin: float, n_models: int) -> float:
    """Map score separation to [0.35, 0.95]; tighter races -> lower confidence."""
    base = 0.45 + min(0.5, margin * 3.5)
    if n_models <= 1:
        base = max(base, 0.55)
    return max(0.35, min(0.95, base))


def normalize_judge_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept LLM output with optional legacy keys; ensure winner/reason/confidence float."""
    if not raw:
        return {"winner": "none", "reason": "", "confidence": 0.0, "winner_agent": "none"}
    w = raw.get("winner") or raw.get("winner_model_key") or "none"
    r = raw.get("reason") or raw.get("rationale") or ""
    c = raw.get("confidence")
    cf: float
    if isinstance(c, (int, float)):
        cf = float(c)
    elif isinstance(c, str):
        low = c.strip().lower()
        if low == "high":
            cf = 0.85
        elif low == "medium":
            cf = 0.65
        elif low == "low":
            cf = 0.45
        else:
            try:
                cf = float(c)
            except ValueError:
                cf = 0.6
    else:
        cf = 0.55 if r else 0.0
    cf = max(0.0, min(1.0, cf))
    agent = raw.get("winner_agent") or AGENT_BY_MODEL_KEY.get(str(w), "unknown")
    return {"winner": str(w), "reason": str(r), "confidence": cf, "winner_agent": str(agent)}
