from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    run_id: str
    run_dir: Path
    target_column: str
    task_type: str
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    preprocessor_path: Path
    X_train_path: Path
    X_test_path: Path
    y_train_path: Path
    y_test_path: Path
    classes_path: Path | None

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "target_column": self.target_column,
            "task_type": self.task_type,
            "feature_columns": self.feature_columns,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "preprocessor_path": str(self.preprocessor_path),
            "X_train_path": str(self.X_train_path),
            "X_test_path": str(self.X_test_path),
            "y_train_path": str(self.y_train_path),
            "y_test_path": str(self.y_test_path),
            "classes_path": str(self.classes_path) if self.classes_path else None,
        }


def _infer_task_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        return "regression"
    if pd.api.types.is_numeric_dtype(y) and y.nunique() <= 20:
        # Could be classification with encoded labels
        return "classification"
    return "classification"


def build_dataset_bundle(
    csv_path: Path,
    target_column: str,
    run_id: str,
    run_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in CSV columns: {list(df.columns)}")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Drop rows with missing target
    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    task_type = _infer_task_type(y)

    if task_type == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            y_codes, uniques = pd.factorize(y)
            y = pd.Series(y_codes, index=y.index)
            classes = uniques.tolist()
        else:
            classes = np.sort(np.unique(y)).tolist()
    else:
        classes = None
        y = pd.to_numeric(y, errors="coerce")
        drop_idx = y.isna()
        if drop_idx.any():
            X = X.loc[~drop_idx].reset_index(drop=True)
            y = y.loc[~drop_idx].reset_index(drop=True)

    feature_columns = list(X.columns)
    numeric_features = [c for c in feature_columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_columns if c not in numeric_features]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        # High cardinality: use ordinal for tree models consistency in single preprocessor
        transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    preprocessor.set_output(transform="pandas")

    strat = None
    if task_type == "classification" and y.nunique() > 1:
        strat = y
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    preprocessor.fit(X_train_df)
    X_train_t = preprocessor.transform(X_train_df)
    X_test_t = preprocessor.transform(X_test_df)

    run_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = run_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    X_train_path = run_dir / "X_train.parquet"
    X_test_path = run_dir / "X_test.parquet"
    y_train_path = run_dir / "y_train.parquet"
    y_test_path = run_dir / "y_test.parquet"
    classes_path = run_dir / "classes.json" if classes is not None else None

    X_train_t.to_parquet(X_train_path, index=False)
    X_test_t.to_parquet(X_test_path, index=False)
    y_train.to_frame(name=target_column).to_parquet(y_train_path, index=False)
    y_test.to_frame(name=target_column).to_parquet(y_test_path, index=False)
    if classes_path is not None:
        classes_path.write_text(json.dumps({"classes": [str(c) for c in classes]}), encoding="utf-8")

    return DatasetBundle(
        run_id=run_id,
        run_dir=run_dir,
        target_column=target_column,
        task_type=task_type,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        preprocessor_path=preprocessor_path,
        X_train_path=X_train_path,
        X_test_path=X_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
        classes_path=classes_path,
    )
