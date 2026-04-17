from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import joblib
import json
import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool
from sklearn.base import BaseEstimator, is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


# --- Public tool API (structured dicts, real sklearn/xgboost, no mocks) ---


def _infer_task_from_y(y: pd.Series) -> Literal["classification", "regression"]:
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 25 and not pd.api.types.is_float_dtype(y):
            return "classification"
        if y.nunique() > 25:
            return "regression"
        return "classification"
    return "classification"


def infer_task_from_y(y: pd.Series) -> Literal["classification", "regression"]:
    """Public alias for EDA agents and external callers."""
    return _infer_task_from_y(y)


def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Impute missing values, encode categoricals, scale numerics, and split train/test.

    Parameters
    ----------
    df : Full dataframe including the target column.
    target_column : Name of the label column (required for stratified splitting).

    Returns
    -------
    Structured dict with arrays, fitted preprocessor, label metadata, and task type.
    """
    if target_column not in df.columns:
        raise ValueError(f"target_column {target_column!r} not in dataframe columns.")

    work = df.copy()
    y_raw = work[target_column]
    X = work.drop(columns=[target_column])

    # Drop rows with missing target
    valid = y_raw.notna()
    X = X.loc[valid].reset_index(drop=True)
    y_raw = y_raw.loc[valid].reset_index(drop=True)

    task_type = _infer_task_from_y(y_raw)

    y_label_encoder: LabelEncoder | None = None
    if task_type == "classification":
        if not pd.api.types.is_numeric_dtype(y_raw):
            y_label_encoder = LabelEncoder()
            y = pd.Series(y_label_encoder.fit_transform(y_raw), name=target_column)
        else:
            y = y_raw.astype(np.int64)
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        drop_idx = y.isna()
        if drop_idx.any():
            X = X.loc[~drop_idx].reset_index(drop=True)
            y = y.loc[~drop_idx].reset_index(drop=True)

    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))

    if not transformers:
        raise ValueError("No feature columns after removing target.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    preprocessor.set_output(transform="pandas")

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task_type == "classification" and y.nunique() > 1 else None,
    )

    preprocessor.fit(X_train_df)
    X_train = preprocessor.transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    feature_names_out: list[str] = list(preprocessor.get_feature_names_out())

    return {
        "task_type": task_type,
        "target_column": target_column,
        "X_train": X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train),
        "X_test": X_test.to_numpy() if hasattr(X_test, "to_numpy") else np.asarray(X_test),
        "y_train": np.asarray(y_train),
        "y_test": np.asarray(y_test),
        "preprocessor": preprocessor,
        "target_label_encoder": y_label_encoder,
        "feature_names_out": feature_names_out,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
    }


def train_model(
    model_name: str,
    params: dict[str, Any],
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    *,
    task_type: Literal["classification", "regression"] | None = None,
) -> dict[str, Any]:
    """
    Fit RandomForest, LogisticRegression, XGBoost, or Ridge on training arrays.

    Parameters
    ----------
    model_name : ``RandomForest``, ``LogisticRegression``, ``XGBoost``, or ``Ridge`` (case-insensitive).
    params : Estimator hyperparameters (merged with sensible defaults).
    """
    name = model_name.strip().lower().replace(" ", "_")
    X = np.asarray(X_train)
    y = np.asarray(y_train).ravel()

    if task_type is None:
        task_type = _infer_task_from_y(pd.Series(y))

    merged: dict[str, Any] = dict(params)

    estimator: BaseEstimator

    if name in ("random_forest", "randomforest"):
        if task_type == "classification":
            estimator = RandomForestClassifier(
                n_estimators=int(merged.pop("n_estimators", 200)),
                max_depth=merged.pop("max_depth", None),
                random_state=int(merged.pop("random_state", 42)),
                n_jobs=int(merged.pop("n_jobs", -1)),
                **merged,
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=int(merged.pop("n_estimators", 400)),
                max_depth=merged.pop("max_depth", None),
                random_state=int(merged.pop("random_state", 42)),
                n_jobs=int(merged.pop("n_jobs", -1)),
                **merged,
            )
    elif name in ("logistic_regression", "logisticregression", "lr"):
        if task_type != "classification":
            raise ValueError("LogisticRegression supports classification task_type only.")
        estimator = LogisticRegression(
            max_iter=int(merged.pop("max_iter", 1000)),
            random_state=int(merged.pop("random_state", 42)),
            **merged,
        )
    elif name in ("xgboost", "xgb"):
        common = {
            "n_estimators": int(merged.pop("n_estimators", 400)),
            "max_depth": int(merged.pop("max_depth", 6)),
            "learning_rate": float(merged.pop("learning_rate", 0.05)),
            "subsample": float(merged.pop("subsample", 0.9)),
            "colsample_bytree": float(merged.pop("colsample_bytree", 0.9)),
            "random_state": int(merged.pop("random_state", 42)),
            "n_jobs": int(merged.pop("n_jobs", -1)),
            "tree_method": str(merged.pop("tree_method", "hist")),
        }
        if task_type == "classification":
            estimator = XGBClassifier(**common, **merged)
        else:
            estimator = XGBRegressor(**common, **merged)
    elif name in ("ridge",):
        if task_type != "regression":
            raise ValueError("Ridge supports regression task_type only.")
        estimator = Ridge(alpha=float(merged.pop("alpha", 1.0)), **merged)
    else:
        raise ValueError(
            f"Unsupported model_name: {model_name!r}. "
            "Use RandomForest, LogisticRegression, XGBoost, or Ridge."
        )

    estimator.fit(X, y)

    return {
        "model": estimator,
        "model_name": model_name,
        "task_type": task_type,
        "params": params,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }


def _avg_for_classification(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    n_classes = len(np.unique(y_true))
    return "binary" if n_classes == 2 else "weighted"


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    X_train: np.ndarray | pd.DataFrame | None = None,
    y_train: np.ndarray | pd.Series | None = None,
) -> dict[str, Any]:
    """
    Evaluate fitted model. For classification: accuracy, precision, recall, f1, and train/test gaps
    (when train arrays are provided). For regression: RMSE, MAE, R2 and gaps.
    """
    Xte = np.asarray(X_test)
    yte = np.asarray(y_test).ravel()

    if is_classifier(model):
        y_pred_test = model.predict(Xte)
        avg = _avg_for_classification(yte, y_pred_test)

        def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
            labels = np.unique(np.concatenate([y_true, y_pred]))
            f1_macro = (
                float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                if labels.size > 1
                else 0.0
            )
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
                "f1_macro": f1_macro,
            }

        out: dict[str, Any] = {
            "metrics_test": _clf_metrics(yte, y_pred_test),
            "metrics_train": None,
            "train_test_gap": None,
        }

        if X_train is not None and y_train is not None:
            Xtr = np.asarray(X_train)
            ytr = np.asarray(y_train).ravel()
            y_pred_train = model.predict(Xtr)
            avg_tr = _avg_for_classification(ytr, y_pred_train)
            m_train = {
                "accuracy": float(accuracy_score(ytr, y_pred_train)),
                "precision": float(precision_score(ytr, y_pred_train, average=avg_tr, zero_division=0)),
                "recall": float(recall_score(ytr, y_pred_train, average=avg_tr, zero_division=0)),
                "f1": float(f1_score(ytr, y_pred_train, average=avg_tr, zero_division=0)),
                "f1_macro": float(f1_score(ytr, y_pred_train, average="macro", zero_division=0)),
            }
            m_test = cast(dict[str, float], out["metrics_test"])
            out["metrics_train"] = m_train
            out["train_test_gap"] = {
                k: float(m_train[k] - m_test[k])
                for k in ("accuracy", "precision", "recall", "f1", "f1_macro")
            }
        return out

    y_pred_test = model.predict(Xte)

    def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    out_reg: dict[str, Any] = {
        "metrics_test": _reg_metrics(yte, y_pred_test),
        "metrics_train": None,
        "train_test_gap": None,
    }

    if X_train is not None and y_train is not None:
        Xtr = np.asarray(X_train)
        ytr = np.asarray(y_train).ravel()
        y_pred_train = model.predict(Xtr)
        m_train = _reg_metrics(ytr, y_pred_train)
        m_test = cast(dict[str, float], out_reg["metrics_test"])
        out_reg["metrics_train"] = m_train
        out_reg["train_test_gap"] = {
            "rmse": float(m_train["rmse"] - m_test["rmse"]),
            "mae": float(m_train["mae"] - m_test["mae"]),
            "r2": float(m_train["r2"] - m_test["r2"]),
        }
    return out_reg


def detect_class_imbalance(y: pd.Series | np.ndarray | list[Any]) -> dict[str, Any]:
    """Return per-class counts and proportions for a label vector."""
    if isinstance(y, (list, tuple)):
        s = pd.Series(y)
    elif isinstance(y, np.ndarray):
        s = pd.Series(y.ravel())
    else:
        s = y

    vc = s.value_counts(dropna=False)
    total = int(vc.sum())
    counts = {str(k): int(v) for k, v in vc.items()}
    proportions = {str(k): float(v / total) if total else 0.0 for k, v in vc.items()}

    vals = list(counts.values())
    imbalance_ratio = float(max(vals) / min(vals)) if vals and min(vals) > 0 else float("inf")

    return {
        "n_samples": total,
        "n_classes": int(len(counts)),
        "counts": counts,
        "proportions": proportions,
        "imbalance_ratio": imbalance_ratio,
    }


def _load_xy(bundle: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    X_train = pd.read_parquet(bundle["X_train_path"])
    y_train = pd.read_parquet(bundle["y_train_path"])[bundle["target_column"]]
    return X_train, y_train


def _load_xy_test(bundle: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    X_test = pd.read_parquet(bundle["X_test_path"])
    y_test = pd.read_parquet(bundle["y_test_path"])[bundle["target_column"]]
    return X_test, y_test


def load_bundle_train_test_numpy(
    bundle: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed train/test matrices from bundle parquet paths (same data as train_estimator)."""
    X_train, y_train = _load_xy(bundle)
    X_test, y_test = _load_xy_test(bundle)
    return (
        np.asarray(X_train),
        np.asarray(y_train).ravel(),
        np.asarray(X_test),
        np.asarray(y_test).ravel(),
    )


def train_estimator(
    bundle: dict[str, Any],
    model_key: str,
    agent_name: str,
    family: Literal[
        "random_forest",
        "xgboost",
        "gradient_boosting",
        "logistic_regression",
        "ridge",
    ],
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Train a scikit-learn or XGBoost estimator on preprocessed training data.
    Persists the fitted estimator to disk and returns metadata (real training — no mocks).
    """
    hyperparams = hyperparams or {}
    X_train, y_train = _load_xy(bundle)
    task = bundle["task_type"]

    if family == "random_forest":
        if task == "classification":
            est = RandomForestClassifier(
                n_estimators=int(hyperparams.get("n_estimators", 200)),
                max_depth=hyperparams.get("max_depth"),
                random_state=int(hyperparams.get("random_state", 42)),
                n_jobs=-1,
            )
        else:
            est = RandomForestRegressor(
                n_estimators=int(hyperparams.get("n_estimators", 400)),
                max_depth=hyperparams.get("max_depth"),
                random_state=int(hyperparams.get("random_state", 42)),
                n_jobs=-1,
            )
    elif family == "xgboost":
        if task == "classification":
            est = XGBClassifier(
                n_estimators=int(hyperparams.get("n_estimators", 400)),
                max_depth=int(hyperparams.get("max_depth", 6)),
                learning_rate=float(hyperparams.get("learning_rate", 0.05)),
                subsample=float(hyperparams.get("subsample", 0.9)),
                colsample_bytree=float(hyperparams.get("colsample_bytree", 0.9)),
                random_state=int(hyperparams.get("random_state", 42)),
                n_jobs=-1,
                tree_method="hist",
            )
        else:
            est = XGBRegressor(
                n_estimators=int(hyperparams.get("n_estimators", 600)),
                max_depth=int(hyperparams.get("max_depth", 6)),
                learning_rate=float(hyperparams.get("learning_rate", 0.05)),
                subsample=float(hyperparams.get("subsample", 0.9)),
                colsample_bytree=float(hyperparams.get("colsample_bytree", 0.9)),
                random_state=int(hyperparams.get("random_state", 42)),
                n_jobs=-1,
                tree_method="hist",
            )
    elif family == "gradient_boosting":
        if task == "classification":
            est = GradientBoostingClassifier(
                n_estimators=int(hyperparams.get("n_estimators", 200)),
                learning_rate=float(hyperparams.get("learning_rate", 0.08)),
                max_depth=int(hyperparams.get("max_depth", 3)),
                random_state=int(hyperparams.get("random_state", 42)),
            )
        else:
            est = GradientBoostingRegressor(
                n_estimators=int(hyperparams.get("n_estimators", 300)),
                learning_rate=float(hyperparams.get("learning_rate", 0.08)),
                max_depth=int(hyperparams.get("max_depth", 3)),
                random_state=int(hyperparams.get("random_state", 42)),
            )
    elif family == "logistic_regression":
        if task != "classification":
            raise ValueError("logistic_regression family requires classification task_type.")
        est = LogisticRegression(
            max_iter=int(hyperparams.get("max_iter", 2000)),
            random_state=int(hyperparams.get("random_state", 42)),
            solver=str(hyperparams.get("solver", "lbfgs")),
            C=float(hyperparams.get("C", 1.0)),
            class_weight=hyperparams.get("class_weight"),
        )
    elif family == "ridge":
        if task != "regression":
            raise ValueError("ridge family requires regression task_type.")
        est = Ridge(alpha=float(hyperparams.get("alpha", 1.0)))
    else:
        raise ValueError(f"Unknown family: {family}")

    est.fit(X_train, y_train)

    run_dir = Path(bundle["run_dir"])
    artifact_path = run_dir / f"model_{model_key}.joblib"
    joblib.dump(est, artifact_path)

    return {
        "agent": agent_name,
        "model_key": model_key,
        "model_type": family,
        "artifact_path": str(artifact_path),
        "params": hyperparams,
    }


def evaluate_model_artifact(
    bundle: dict[str, Any],
    model_key: str,
    artifact_path: str,
) -> dict[str, float | str]:
    """Compute held-out metrics for a persisted estimator."""
    est = joblib.load(artifact_path)
    X_test, y_test = _load_xy_test(bundle)
    task = bundle["task_type"]

    if task == "classification":
        y_hat = est.predict(X_test)
        metrics: dict[str, float | str] = {
            "accuracy": float(accuracy_score(y_test, y_hat)),
        }
        labels = np.unique(np.concatenate([y_test.to_numpy(), y_hat]))
        if labels.size > 1:
            metrics["f1_macro"] = float(f1_score(y_test, y_hat, average="macro"))
        try:
            if hasattr(est, "predict_proba") and len(labels) == 2:
                proba = est.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            pass
        return metrics

    y_hat = est.predict(X_test)
    return {
        "rmse": float(mean_squared_error(y_test, y_hat) ** 0.5),
        "mae": float(mean_absolute_error(y_test, y_hat)),
        "r2": float(r2_score(y_test, y_hat)),
    }


def build_training_toolkit(bundle: dict[str, Any]) -> dict[str, StructuredTool]:
    """LangChain tools that wrap real fit/persist logic (invoked by agent nodes via `.invoke`)."""

    def _train_rf() -> str:
        r = train_estimator(
            bundle,
            model_key="rf",
            agent_name="model_agent_rf",
            family="random_forest",
            hyperparams={"n_estimators": 300, "max_depth": None, "random_state": 42},
        )
        return json.dumps(r)

    def _train_xgb() -> str:
        r = train_estimator(
            bundle,
            model_key="xgb",
            agent_name="model_agent_xgb",
            family="xgboost",
            hyperparams={
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42,
            },
        )
        return json.dumps(r)

    def _train_gbm() -> str:
        r = train_estimator(
            bundle,
            model_key="gbm",
            agent_name="model_agent_sklearn_gbm",
            family="gradient_boosting",
            hyperparams={"n_estimators": 250, "learning_rate": 0.08, "max_depth": 3, "random_state": 42},
        )
        return json.dumps(r)

    def _evaluate(model_key: str, artifact_path: str) -> str:
        m = evaluate_model_artifact(bundle, model_key=model_key, artifact_path=artifact_path)
        return json.dumps({"model_key": model_key, "metrics": m})

    return {
        "train_random_forest": StructuredTool.from_function(
            name="train_random_forest",
            description="Fit sklearn RandomForest on preprocessed train parquet; persist joblib artifact.",
            func=_train_rf,
        ),
        "train_xgboost": StructuredTool.from_function(
            name="train_xgboost",
            description="Fit XGBoost classifier/regressor on preprocessed train parquet; persist joblib artifact.",
            func=_train_xgb,
        ),
        "train_sklearn_gradient_boosting": StructuredTool.from_function(
            name="train_sklearn_gradient_boosting",
            description="Fit sklearn GradientBoosting on preprocessed train parquet; persist joblib artifact.",
            func=_train_gbm,
        ),
        "evaluate_artifact": StructuredTool.from_function(
            name="evaluate_artifact",
            description="Evaluate one persisted model on the hold-out parquet test set.",
            func=_evaluate,
        ),
    }
