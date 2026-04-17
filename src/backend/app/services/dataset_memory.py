"""
Vector memory for *dataset → outcome* patterns using Chroma (same stack as ``MemoryService``).

Stores embedded documents describing dataset characteristics, judge-selected best model, and holdout
metrics so we can retrieve similar past runs and inject short priors into model agents.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

import app.sqlite_patch  # noqa: F401
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app.config import settings


@dataclass
class PatternHit:
    page_content: str
    metadata: dict[str, Any]
    distance: float | None = None


def _det(eda_structured: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(eda_structured, dict):
        return {}
    return eda_structured.get("deterministic") or {}


def build_dataset_query_text(eda_structured: dict[str, Any] | None, task_type: str) -> str:
    """
    Text used for similarity search — mirrors the vocabulary in stored documents so retrieval aligns
    with dataset shape, leakage of target stats, and task type.
    """
    d = _det(eda_structured)
    tp = d.get("target_profile") or {}
    mv = d.get("missing_values") or {}
    imb = d.get("class_imbalance") or {}
    ratio = imb.get("imbalance_ratio")
    try:
        ratio_f = float(ratio) if ratio is not None and ratio != float("inf") else None
    except (TypeError, ValueError):
        ratio_f = None

    parts = [
        f"task_type={task_type}",
        f"n_rows={d.get('n_rows', 0)}",
        f"n_features={d.get('n_features', 0)}",
        f"fraction_rows_any_missing={mv.get('fraction_rows_any_missing', 0)}",
        f"target_task_hint={tp.get('task_hint', '')}",
    ]
    if task_type == "classification":
        parts.append(f"n_classes={tp.get('n_classes', '')}")
        if ratio_f is not None:
            parts.append(f"class_imbalance_ratio={ratio_f:.4f}")
    corr = d.get("correlations") or {}
    pairs = corr.get("top_feature_pairs") or []
    if pairs:
        top = pairs[0]
        parts.append(f"top_feature_pair_pearson={top.get('pearson', 0)}")
    return " | ".join(str(p) for p in parts)


def build_stored_document(
    run_id: str,
    task_type: str,
    eda_structured: dict[str, Any] | None,
    winner: str,
    judge_confidence: float,
    metrics_by_model: dict[str, dict[str, Any]],
) -> str:
    """Rich natural-language + JSON snippet for embedding (dataset characteristics + outcomes)."""
    d = _det(eda_structured)
    tp = d.get("target_profile") or {}
    mv = d.get("missing_values") or {}
    imb = d.get("class_imbalance") or {}
    metrics_compact = {
        k: {kk: round(float(vv), 6) if isinstance(vv, (int, float)) else vv for kk, vv in m.items()}
        for k, m in metrics_by_model.items()
    }
    payload = {
        "holdout_metrics_by_model": metrics_compact,
        "judge_winner": winner,
        "judge_confidence": judge_confidence,
    }
    lines = [
        f"run_id={run_id}",
        f"task={task_type}",
        f"rows={d.get('n_rows')}",
        f"features={d.get('n_features')}",
        f"missing_row_fraction={mv.get('fraction_rows_any_missing')}",
        f"target_hint={tp.get('task_hint')}",
    ]
    if task_type == "classification":
        lines.append(f"n_classes={tp.get('n_classes')}")
        r = imb.get("imbalance_ratio")
        lines.append(f"class_imbalance_ratio={r}")
    lines.append(f"best_model={winner}")
    lines.append(f"performance_json={json.dumps(payload, sort_keys=True)}")
    return "\n".join(lines)


class DatasetPatternMemory:
    """Chroma collection dedicated to dataset-pattern retrieval (separate from debate snippet memory)."""

    COLLECTION = "dataset_patterns"

    def __init__(self) -> None:
        self._ef = DefaultEmbeddingFunction()
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            embedding_function=self._ef,
        )

    def find_similar_dataset_patterns(
        self,
        eda_structured: dict[str, Any] | None,
        task_type: str,
        *,
        k: int = 5,
        exclude_run_id: str | None = None,
    ) -> list[PatternHit]:
        """
        Retrieve past runs whose dataset characterization is closest to the current EDA (embedding similarity).
        Optionally drop the current ``run_id`` from results (e.g. after re-indexing).
        """
        q = build_dataset_query_text(eda_structured, task_type)
        n_fetch = max(k * 6, 24)
        raw = self._col.query(query_texts=[q], n_results=n_fetch)
        docs = raw.get("documents", [[]])[0] or []
        metas = raw.get("metadatas", [[]])[0] or []
        dists = raw.get("distances", [[]])[0] or []

        hits: list[PatternHit] = []
        for doc, meta, dist in zip(docs, metas, dists):
            if not doc:
                continue
            m = meta or {}
            if exclude_run_id and m.get("run_id") == exclude_run_id:
                continue
            hits.append(PatternHit(page_content=doc, metadata=m, distance=float(dist) if dist is not None else None))
            if len(hits) >= k:
                break
        return hits

    def index_completed_run(
        self,
        run_id: str,
        task_type: str,
        eda_structured: dict[str, Any] | None,
        winner: str,
        judge_confidence: float,
        metrics_by_model: dict[str, dict[str, Any]],
    ) -> str:
        """Persist one vector record for a finished successful pipeline run."""
        doc = build_stored_document(
            run_id,
            task_type,
            eda_structured,
            winner,
            judge_confidence,
            metrics_by_model,
        )
        d = _det(eda_structured)
        meta: dict[str, Any] = {
            "run_id": run_id,
            "task_type": task_type,
            "winner": winner,
            "judge_confidence": float(judge_confidence),
            "n_rows": int(d.get("n_rows") or 0),
            "n_features": int(d.get("n_features") or 0),
            "kind": "dataset_run_outcome",
        }
        tp = d.get("target_profile") or {}
        if task_type == "classification":
            imb = d.get("class_imbalance") or {}
            r = imb.get("imbalance_ratio")
            try:
                if r is not None and r != float("inf"):
                    meta["imbalance_ratio"] = float(r)
            except (TypeError, ValueError):
                pass
            mf = metrics_by_model.get(winner) or {}
            fm = mf.get("f1_macro", mf.get("f1"))
            if fm is not None:
                try:
                    meta["winner_f1_macro"] = float(fm)
                except (TypeError, ValueError):
                    pass
        else:
            mf = metrics_by_model.get(winner) or {}
            rm = mf.get("rmse")
            if rm is not None:
                try:
                    meta["winner_rmse"] = float(rm)
                except (TypeError, ValueError):
                    pass

        uid = f"{run_id}_{uuid.uuid4().hex[:8]}"
        self._col.add(ids=[uid], documents=[doc], metadatas=[meta])
        return uid


def format_priors_for_model_agents(hits: list[PatternHit], *, max_chars: int = 2200) -> str:
    """Turn retrieval hits into a compact paragraph for proposal reasoning."""
    if not hits:
        return ""
    chunks: list[str] = []
    for i, h in enumerate(hits[:5], start=1):
        w = (h.metadata or {}).get("winner", "?")
        rid = (h.metadata or {}).get("run_id", "?")
        conf = (h.metadata or {}).get("judge_confidence", "")
        chunks.append(f"[{i}] run {rid}: winner={w} (judge_conf≈{conf}). Snippet: {h.page_content[:420].strip()}…")
    text = " ".join(chunks)
    return text[:max_chars]


def index_completed_run_from_state(run_id: str, final_state: dict[str, Any]) -> None:
    """Index a successful graph final state into dataset pattern memory (no-op on error or missing data)."""
    if final_state.get("error"):
        return
    metrics = final_state.get("metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        return
    jd = final_state.get("judge_decision") or {}
    winner = str(jd.get("winner") or "")
    if not winner or winner == "none":
        return
    try:
        jconf = float(jd.get("confidence", 0.0))
    except (TypeError, ValueError):
        jconf = 0.0
    task = str(final_state.get("task_type") or "classification")
    eda = final_state.get("eda_structured") if isinstance(final_state.get("eda_structured"), dict) else None

    mem = DatasetPatternMemory()
    mem.index_completed_run(
        run_id=run_id,
        task_type=task,
        eda_structured=eda,
        winner=winner,
        judge_confidence=jconf,
        metrics_by_model={k: dict(v) for k, v in metrics.items() if isinstance(v, dict)},
    )
