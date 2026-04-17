from __future__ import annotations

from typing import Any

import pandas as pd

from app.tools.ml_tools import detect_class_imbalance, infer_task_from_y


def compute_deterministic_eda(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Pandas-only EDA: feature roles, missingness, imbalance, correlations.
    Returns a concise nested dict suitable for JSON serialization.
    """
    if target_column not in df.columns:
        raise ValueError(f"target_column {target_column!r} not in dataframe.")

    n_rows = len(df)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    feature_types: dict[str, Any] = {}
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            role = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            role = "datetime"
        else:
            role = "categorical"
        feature_types[str(col)] = {
            "role": role,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=False)),
        }

    missing_by_column = {str(c): round(float(X[c].isna().mean()), 6) for c in X.columns}
    rows_with_any_feature_missing = int(X.isna().any(axis=1).sum())
    fraction_rows_any_missing = round(float(rows_with_any_feature_missing / n_rows), 6) if n_rows else 0.0

    y_clean = y.dropna()
    task_hint = infer_task_from_y(y_clean)

    target_profile: dict[str, Any] = {
        "task_hint": task_hint,
        "missing_count": int(y.isna().sum()),
        "missing_fraction": round(float(y.isna().mean()), 6) if n_rows else 0.0,
    }
    if task_hint == "classification":
        target_profile["n_classes"] = int(y_clean.nunique())
    else:
        y_num = pd.to_numeric(y_clean, errors="coerce").dropna()
        if len(y_num):
            target_profile["numeric_summary"] = {
                "min": float(y_num.min()),
                "max": float(y_num.max()),
                "mean": float(y_num.mean()),
            }

    class_imbalance: dict[str, Any] | None = None
    if task_hint == "classification" and len(y_clean):
        class_imbalance = detect_class_imbalance(y_clean)

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    correlations: dict[str, Any] = {"method": "pearson", "top_feature_pairs": [], "target_vs_numeric": []}

    if len(num_cols) >= 2:
        cmat = X[num_cols].corr(numeric_only=True, method="pearson")
        pairs: list[tuple[str, str, float]] = []
        for i, a in enumerate(num_cols):
            for b in num_cols[i + 1 :]:
                v = cmat.loc[a, b]
                if pd.notna(v):
                    pairs.append((a, b, float(v)))
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
        correlations["top_feature_pairs"] = [
            {"feature_a": a, "feature_b": b, "pearson": round(c, 6)} for a, b, c in pairs[:12]
        ]

    # Target vs numeric (encoded target for classification)
    if num_cols:
        if task_hint == "classification":
            y_enc = pd.Series(pd.factorize(y)[0], index=y.index, dtype=float)
        else:
            y_enc = pd.to_numeric(y, errors="coerce")
        tvc: list[dict[str, Any]] = []
        for col in num_cols:
            sub = pd.DataFrame({"x": pd.to_numeric(X[col], errors="coerce"), "yt": y_enc}).dropna()
            if len(sub) < 3:
                continue
            r = sub["x"].corr(sub["yt"], method="pearson")
            if pd.notna(r):
                tvc.append({"feature": str(col), "pearson_vs_target": round(float(r), 6)})
        tvc.sort(key=lambda d: abs(d["pearson_vs_target"]), reverse=True)
        correlations["target_vs_numeric"] = tvc[:15]

    return {
        "schema_version": "1.0",
        "n_rows": n_rows,
        "n_features": int(X.shape[1]),
        "target_column": target_column,
        "feature_types": feature_types,
        "missing_values": {
            "by_column": missing_by_column,
            "rows_with_any_feature_missing": rows_with_any_feature_missing,
            "fraction_rows_any_missing": fraction_rows_any_missing,
        },
        "target_profile": target_profile,
        "class_imbalance": class_imbalance,
        "correlations": correlations,
    }
