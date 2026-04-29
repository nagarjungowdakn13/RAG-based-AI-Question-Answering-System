"""RAG orchestrator: the public surface used by the API and CLI.

Composition root for the whole pipeline. Holds singletons for the
embedder / vector store / generator so we don't reload models on every
request — startup-time cost only.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable

from app.config import settings
from app.logger import get_logger, log_query_event
from app.pipeline.chunking import chunk_documents
from app.pipeline.embeddings import Embedder, build_embedder
from app.pipeline.generator import UNKNOWN_ANSWER, Generator, build_generator
from app.pipeline.ingestion import load_documents
from app.pipeline.retriever import Retriever
from app.pipeline.vector_store import FaissVectorStore

logger = get_logger(__name__)


class RAGPipeline:
    """High-level facade. Thread-safe for concurrent FastAPI requests."""

    _instance: "RAGPipeline | None" = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> "RAGPipeline":
        # Double-checked lazy singleton: model load is expensive (~1–3s)
        # so we only want to do it once per process.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.embedder: Embedder = build_embedder()
        self.generator: Generator = build_generator()
        self.store: FaissVectorStore = (
            FaissVectorStore.load() or FaissVectorStore(dim=self.embedder.dim)
        )
        self._mut = threading.Lock()  # guards index writes

    # ─── ingestion ───────────────────────────────────────────────────
    def ingest(
        self,
        paths: Iterable[str | Path],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict:
        docs = load_documents(paths)
        if not docs:
            return {"ingested_documents": 0, "total_chunks": 0, "files": []}
        chunks = chunk_documents(docs, chunk_size, chunk_overlap)
        vectors = self.embedder.embed([c.text for c in chunks])

        with self._mut:
            # Re-init the index if the embedder dim ever changed (e.g. user
            # switched backends between runs). Cheaper than failing silently.
            if self.store.dim != vectors.shape[1]:
                logger.warning(
                    "Embedding dim changed (%d → %d); rebuilding index.",
                    self.store.dim, vectors.shape[1],
                )
                self.store = FaissVectorStore(dim=vectors.shape[1])
            added = self.store.add(vectors, chunks)
            if added:
                self.store.save()

        return {
            "ingested_documents": len(docs),
            "total_chunks": added,
            "files": [d.metadata.get("source", "") for d in docs],
        }

    def delete_source(self, source: str) -> dict:
        with self._mut:
            deleted = self.store.delete_by_source(source)
            if deleted:
                self.store.save()
        return {
            "deleted_chunks": deleted,
            "index_size": self.store.size,
            "source": source,
        }

    # ─── querying ────────────────────────────────────────────────────
    def query(
        self,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> dict:
        retriever = Retriever(self.embedder, self.store)
        result = retriever.retrieve(question, top_k=top_k, score_threshold=score_threshold)

        # Hallucination guard #1: low retrieval confidence → short-circuit.
        if not result.confident:
            answer = UNKNOWN_ANSWER
        else:
            answer = self.generator.generate(question, result.hits)

        sources = [
            {
                "rank": i + 1,
                "score": h.score,
                "source": h.metadata.get("source", ""),
                "chunk_id": h.chunk_id,
                "snippet": (h.text[:240] + "…") if len(h.text) > 240 else h.text,
            }
            for i, h in enumerate(result.hits)
        ]

        payload = {
            "question": question,
            "answer": answer,
            "confident": result.confident,
            "confidence_score": result.top_score,
            "sources": sources,
            "metadata": {
                "embedding_backend": settings.embedding_backend,
                "llm_backend": settings.llm_backend,
                "top_k": top_k or settings.top_k,
                "score_threshold": score_threshold
                if score_threshold is not None
                else settings.score_threshold,
                "index_size": self.store.size,
            },
        }
        log_query_event(payload)
        return payload
