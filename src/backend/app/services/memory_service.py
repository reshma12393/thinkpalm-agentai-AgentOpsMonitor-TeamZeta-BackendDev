from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import app.sqlite_patch  # noqa: F401 — before chromadb
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app.config import settings


@dataclass
class MemoryHit:
    """Lightweight retrieval result (replaces LangChain Document for Chroma-native usage)."""

    page_content: str
    metadata: dict[str, Any]


class MemoryService:
    """Vector memory backed by Chroma for agent reasoning and EDA snippets."""

    def __init__(self, collection_name: str | None = None) -> None:
        self._collection_name = collection_name or "automl_debate"
        self._ef = DefaultEmbeddingFunction()
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._ef,
        )

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]
        # Chroma metadata values must be str/int/float/bool
        clean_meta: list[dict[str, Any]] = []
        for m in metadatas:
            cm = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    cm[k] = v
                else:
                    cm[k] = str(v)
            clean_meta.append(cm)
        self._col.add(ids=ids, documents=texts, metadatas=clean_meta)
        return ids

    def similarity_search_with_run(self, query: str, run_id: str, k: int = 6) -> list[MemoryHit]:
        n = max(k * 4, 12)
        raw = self._col.query(query_texts=[query], n_results=n)
        docs = raw.get("documents", [[]])[0] or []
        metas = raw.get("metadatas", [[]])[0] or []
        hits: list[MemoryHit] = []
        for doc, meta in zip(docs, metas):
            if not doc:
                continue
            m = meta or {}
            if m.get("run_id") == run_id:
                hits.append(MemoryHit(page_content=doc, metadata=m))
        return hits[:k]
