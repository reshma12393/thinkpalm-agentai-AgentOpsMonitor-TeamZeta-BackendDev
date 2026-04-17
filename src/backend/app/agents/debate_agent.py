from __future__ import annotations

import json
from typing import Any


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _class_imbalance_from_eda(eda_structured: dict[str, Any] | None) -> tuple[float | None, str | None]:
    if not isinstance(eda_structured, dict):
        return None, None
    det = eda_structured.get("deterministic")
    if not isinstance(det, dict):
        return None, None
    ci = det.get("class_imbalance")
    if not isinstance(ci, dict):
        return None, None
    ratio = ci.get("imbalance_ratio")
    fr = _safe_float(ratio)
    if fr is None or fr == float("inf"):
        return None, None
    return fr, str(ci.get("proportions", ""))[:200] if ci.get("proportions") else None


def _comparison_row_by_key(evaluation_report: dict[str, Any], mk: str) -> dict[str, Any] | None:
    for row in evaluation_report.get("comparison_table") or []:
        if isinstance(row, dict) and row.get("model_key") == mk:
            return row
    return None


def _primary_classification(m: dict[str, Any]) -> tuple[str, float]:
    if "f1_macro" in m:
        return "f1_macro", float(m["f1_macro"])
    return "f1", float(m.get("f1", 0.0))


def _primary_regression(m: dict[str, Any]) -> tuple[str, float]:
    return "rmse", float(m.get("rmse", 0.0))


def build_debate_analysis(
    model_runs: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, float | str]],
    evaluation_report: dict[str, Any],
    task_type: str,
    eda_structured: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Compare all models using holdout metrics, train vs test gaps, and EDA imbalance signals.
    Every bullet is tied to concrete numbers from the pipeline (no generic filler).
    """
    keys = list(metrics.keys())
    if not keys:
        return {
            "task_type": task_type,
            "per_model": [],
            "imbalance_context": None,
            "notes": "No metrics to debate.",
        }

    imb_ratio, _imb_props = _class_imbalance_from_eda(eda_structured)
    imbalance_context: dict[str, Any] | None = None
    if imb_ratio is not None and task_type == "classification" and imb_ratio >= 1.5:
        imbalance_context = {
            "imbalance_ratio": round(imb_ratio, 4),
            "note": "EDA class_imbalance: larger ratio means more skew between majority and minority classes.",
        }

    # Peer aggregates for classification / regression
    if task_type == "classification":
        f1m = {k: _safe_float(metrics[k].get("f1_macro")) for k in keys}
        f1m = {k: v for k, v in f1m.items() if v is not None}
        accs = {k: _safe_float(metrics[k].get("accuracy")) for k in keys}
        accs = {k: v for k, v in accs.items() if v is not None}
        wf1 = {k: _safe_float(metrics[k].get("f1")) for k in keys}
        wf1 = {k: v for k, v in wf1.items() if v is not None}
        best_f1m = max(f1m.values()) if f1m else None
        worst_f1m = min(f1m.values()) if f1m else None
    else:
        rmses = {k: _safe_float(metrics[k].get("rmse")) for k in keys}
        rmses = {k: v for k, v in rmses.items() if v is not None}
        r2s = {k: _safe_float(metrics[k].get("r2")) for k in keys}
        r2s = {k: v for k, v in r2s.items() if v is not None}
        best_rmse = min(rmses.values()) if rmses else None
        worst_rmse = max(rmses.values()) if rmses else None
        best_r2 = max(r2s.values()) if r2s else None
        worst_r2 = min(r2s.values()) if r2s else None

    train_f1m_peer: list[float] = []
    for mk in keys:
        ev = model_runs.get(mk, {}).get("evaluation")
        if isinstance(ev, dict) and ev.get("metrics_train"):
            mt = ev["metrics_train"]
            if isinstance(mt, dict):
                v = _safe_float(mt.get("f1_macro"))
                if v is not None:
                    train_f1m_peer.append(v)
    max_train_f1m = max(train_f1m_peer) if train_f1m_peer else None

    per_model: list[dict[str, Any]] = []

    # Stable order: evaluation ranking when present
    ranked_keys = [r.get("model_key") for r in (evaluation_report.get("comparison_table") or []) if r.get("model_key")]
    ordered = [k for k in ranked_keys if k in keys] + [k for k in keys if k not in ranked_keys]

    gap_mags = [
        float(_comparison_row_by_key(evaluation_report, mk)["overfitting_gap_magnitude"] or 0)
        for mk in ordered
        if _comparison_row_by_key(evaluation_report, mk)
    ]
    min_gap_mag = min(gap_mags) if gap_mags else None

    for mk in ordered:
        m = dict(metrics.get(mk) or {})
        info = model_runs.get(mk) or {}
        ev = info.get("evaluation") if isinstance(info.get("evaluation"), dict) else {}
        row = _comparison_row_by_key(evaluation_report, mk) or {}
        model_type = str(info.get("model_type") or mk)

        strengths: list[str] = []
        weaknesses: list[str] = []

        if task_type == "classification":
            te_f1m = _safe_float(m.get("f1_macro"))
            te_acc = _safe_float(m.get("accuracy"))
            te_wf1 = _safe_float(m.get("f1"))
            gap = ev.get("train_test_gap") if isinstance(ev.get("train_test_gap"), dict) else {}
            tr = ev.get("metrics_train") if isinstance(ev.get("metrics_train"), dict) else {}
            tr_f1m = _safe_float(tr.get("f1_macro")) if tr else None

            risk = str(row.get("overfitting_risk") or "")
            gmag = row.get("overfitting_gap_magnitude")
            gap_f1m = _safe_float(gap.get("f1_macro")) if gap else None
            gap_acc = _safe_float(gap.get("accuracy")) if gap else None

            # Strengths
            if te_f1m is not None and best_f1m is not None and te_f1m >= best_f1m - 1e-9:
                strengths.append(
                    f"Best holdout f1_macro in this run: {te_f1m:.4f} (peer min {worst_f1m:.4f}, max {best_f1m:.4f})."
                )
            elif te_f1m is not None and te_acc is not None:
                strengths.append(
                    f"Holdout metrics: f1_macro={te_f1m:.4f}, accuracy={te_acc:.4f} (same train/test split as other models)."
                )

            if risk == "low" and min_gap_mag is not None and gmag is not None and abs(float(gmag) - min_gap_mag) < 1e-9:
                strengths.append(
                    f"Tightest train–test gap magnitude in this cohort ({float(gmag):.4f}) with '{risk}' overfitting risk — most stable generalization by this measure."
                )

            if (
                imbalance_context
                and te_f1m is not None
                and te_wf1 is not None
                and imb_ratio is not None
                and imb_ratio >= 1.8
            ):
                spread = te_wf1 - te_f1m
                if spread <= 0.03:
                    strengths.append(
                        f"Imbalance handling: under EDA imbalance_ratio≈{imb_ratio:.2f}, weighted F1 {te_wf1:.4f} vs macro F1 {te_f1m:.4f} (difference {spread:.4f}) — macro stays close to weighted, so minority classes are not dramatically left behind vs the headline F1."
                    )

            # Weaknesses — overfitting
            if gap_f1m is not None and gap_f1m > 0.04:
                tr_str = f"{tr_f1m:.4f}" if tr_f1m is not None else "n/a"
                te_str = f"{te_f1m:.4f}" if te_f1m is not None else "n/a"
                weaknesses.append(
                    f"Overfitting signal: train f1_macro ({tr_str}) minus test f1_macro ({te_str}) = {gap_f1m:.4f} (positive gap means train fits tighter than holdout); overfitting_risk={risk or 'n/a'}, gap_magnitude={gmag}."
                )
            elif risk == "high":
                weaknesses.append(
                    f"High overfitting risk in the evaluation table (gap_magnitude={gmag}) despite multi-metric gaps — compare train vs test in the evaluation JSON."
                )

            # Underfitting / linear vs peers
            if te_f1m is not None and worst_f1m is not None and te_f1m <= worst_f1m + 1e-9 and len(keys) > 1:
                others = [metrics[k] for k in keys if k != mk]
                oth_f1 = [
                    float(x.get("f1_macro", x.get("f1", 0.0)))
                    for x in others
                    if x is not None
                ]
                if oth_f1:
                    best_other = max(oth_f1)
                    if best_other - te_f1m >= 0.02:
                        weaknesses.append(
                            f"Underperforms on holdout vs peers: f1_macro {te_f1m:.4f} vs best other model {best_other:.4f} on the same test split."
                        )
            if (
                tr_f1m is not None
                and max_train_f1m is not None
                and te_f1m is not None
                and tr_f1m < max_train_f1m - 0.05
                and te_f1m <= (worst_f1m or 0) + 1e-9
            ):
                weaknesses.append(
                    f"Underfitting-style pattern: training f1_macro {tr_f1m:.4f} is at least 0.05 below the strongest training f1_macro among models ({max_train_f1m:.4f}) while holdout f1_macro {te_f1m:.4f} is weakest — limited fit rather than memorization gap."
                )

            mt_lower = model_type.lower()
            if (
                "logistic" in mt_lower
                and te_f1m is not None
                and best_f1m is not None
                and best_f1m - te_f1m >= 0.03
            ):
                weaknesses.append(
                    f"Linear model ({model_type}) trails the best holdout f1_macro by {best_f1m - te_f1m:.4f}; tree ensembles lead on this split — consistent with a larger non-linear separability margin on these metrics (not a generic claim)."
                )

            # Imbalance — weakness when macro lags weighted
            if (
                imbalance_context
                and te_f1m is not None
                and te_wf1 is not None
                and imb_ratio is not None
                and imb_ratio >= 1.8
            ):
                spread = te_wf1 - te_f1m
                if spread > 0.05:
                    weaknesses.append(
                        f"Imbalance stress: EDA imbalance_ratio≈{imb_ratio:.2f}; weighted F1 {te_wf1:.4f} exceeds macro F1 {te_f1m:.4f} by {spread:.4f} — accuracy/weighted scores can hide weaker minority-class behavior (compare macro vs weighted)."
                    )

        else:
            # Regression
            te_rmse = _safe_float(m.get("rmse"))
            te_r2 = _safe_float(m.get("r2"))
            te_mae = _safe_float(m.get("mae"))
            gap = ev.get("train_test_gap") if isinstance(ev.get("train_test_gap"), dict) else {}
            tr = ev.get("metrics_train") if isinstance(ev.get("metrics_train"), dict) else {}
            gap_rmse = _safe_float(gap.get("rmse")) if gap else None
            tr_rmse = _safe_float(tr.get("rmse")) if tr else None
            risk = str(row.get("overfitting_risk") or "")
            gmag = row.get("overfitting_gap_magnitude")

            if te_rmse is not None and te_r2 is not None:
                strengths.append(f"Holdout rmse={te_rmse:.6f}, r2={te_r2:.6f}" + (f", mae={te_mae:.6f}" if te_mae else "") + ".")
            if best_rmse is not None and te_rmse is not None and te_rmse <= best_rmse + 1e-12:
                strengths.append(f"Lowest holdout RMSE among models: {te_rmse:.6f} (worst in cohort {worst_rmse:.6f}).")

            # Overfitting: train RMSE much lower than test => negative gap (train - test)
            if gap_rmse is not None and gap_rmse < -0.01 and tr_rmse is not None and te_rmse is not None:
                weaknesses.append(
                    f"Overfitting signal: train RMSE {tr_rmse:.6f} vs test RMSE {te_rmse:.6f} (train_minus_test gap {gap_rmse:.6f}); evaluation overfitting_risk={risk or 'n/a'}, magnitude={gmag}."
                )
            elif risk == "high":
                weaknesses.append(f"High overfitting risk in the evaluation row (gap_magnitude={gmag}) — compare metrics_train vs metrics_test in evaluation JSON.")

            if te_r2 is not None and worst_r2 is not None and te_r2 <= worst_r2 + 1e-9 and len(keys) > 1:
                others = [_safe_float(metrics[k].get("r2")) for k in keys if k != mk]
                others_v = [x for x in others if x is not None]
                if others_v and best_r2 is not None:
                    weaknesses.append(
                        f"Weakest holdout r2={te_r2:.6f} vs best peer r2={best_r2:.6f} on the same split."
                    )

            if "ridge" in model_type.lower() and te_rmse is not None and best_rmse is not None and te_rmse > best_rmse + 1e-9:
                weaknesses.append(
                    f"Ridge regression rmse {te_rmse:.6f} vs best cohort rmse {best_rmse:.6f} — linear structure may underfit non-linear signal if trees beat it materially on these metrics."
                )

        # De-duplicate near-identical lines
        def _dedupe(lines: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                key = line[:220]
                if key in seen:
                    continue
                seen.add(key)
                out.append(line)
            return out[:8]

        per_model.append(
            {
                "model_key": mk,
                "model_type": model_type,
                "primary_metric": (
                    _primary_classification(m)[0]
                    if task_type == "classification"
                    else _primary_regression(m)[0]
                ),
                "strengths": _dedupe(strengths),
                "weaknesses": _dedupe(weaknesses),
            }
        )

    # Ensure each model has at least one substantive line
    for block in per_model:
        if not block["strengths"] and not block["weaknesses"]:
            mk = block["model_key"]
            raw = json.dumps(metrics.get(mk, {}), sort_keys=True)
            block["strengths"].append(f"Reported holdout metrics (verbatim): {raw}")

    return {
        "task_type": task_type,
        "imbalance_context": imbalance_context,
        "per_model": per_model,
    }


def format_debate_transcript(analysis: dict[str, Any]) -> str:
    lines: list[str] = ["## Debate agent — metric-grounded comparison", ""]
    ic = analysis.get("imbalance_context")
    if isinstance(ic, dict):
        lines.append(f"Context: {json.dumps(ic)}")
        lines.append("")
    for block in analysis.get("per_model") or []:
        mk = block.get("model_key", "?")
        mt = block.get("model_type", "")
        lines.append(f"### {mk} ({mt})")
        lines.append("")
        lines.append("**Strengths**")
        for s in block.get("strengths") or []:
            lines.append(f"- {s}")
        if not block.get("strengths"):
            lines.append("- (none)")
        lines.append("")
        lines.append("**Weaknesses**")
        for w in block.get("weaknesses") or []:
            lines.append(f"- {w}")
        if not block.get("weaknesses"):
            lines.append("- (none)")
        lines.append("")
    note = analysis.get("notes")
    if note:
        lines.append(note)
    return "\n".join(lines).strip()


def merge_transcript_with_llm_narrative(base_transcript: str, narrative: str) -> str:
    narrative = narrative.strip()
    if not narrative:
        return base_transcript
    return (
        base_transcript
        + "\n\n---\n\n## Senior ML reviewer (bullets; must match metrics above)\n\n"
        + narrative
    )
