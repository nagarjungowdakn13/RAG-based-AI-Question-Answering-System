"""RAG orchestrator: retrieve, generate, validate, observe.

This facade keeps ingestion, retrieval, and generation independently tunable:
retrieval produces scored chunks; generation consumes only those chunks; the
post-generation reliability layer decides whether the answer is grounded.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterable, Iterator

from app.config import settings
from app.generation.prompts import PromptStrategy
from app.logger import get_logger
from app.observability.query_trace import log_query_trace
from app.pipeline.chunking import chunk_documents
from app.pipeline.embeddings import Embedder, build_embedder
from app.pipeline.generator import UNKNOWN_ANSWER, Generator, build_generator
from app.pipeline.ingestion import load_documents
from app.pipeline.query_cache import CacheKey, QueryCache
from app.pipeline.retriever import Retriever, RetrievalResult
from app.pipeline.vector_store import FaissVectorStore, RetrievalHit
from app.retrieval.bm25 import BM25Index
from app.retrieval.confidence import ConfidenceResult, score_confidence
from app.retrieval.hallucination import HallucinationCheck, validate_grounding
from app.retrieval.reranker import Reranker, build_reranker

logger = get_logger(__name__)


class RAGPipeline:
    """High-level facade. Thread-safe for concurrent FastAPI requests."""

    _instance: "RAGPipeline | None" = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> "RAGPipeline":
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
        self._mut = threading.Lock()
        self.reranker: Reranker = build_reranker()
        self.cache: QueryCache = QueryCache(maxsize=settings.query_cache_size)
        # BM25 lives alongside the FAISS index but is rebuilt lazily so we
        # don't pay the cost when EBR-only mode is active.
        self._bm25: BM25Index | None = None
        self._bm25_version: int = -1
        if self.store.size:
            self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        bm25 = BM25Index()
        bm25.build(self.store.texts())
        self._bm25 = bm25
        self._bm25_version = self.store.version

    def _ensure_bm25(self) -> BM25Index | None:
        if settings.retrieval_mode not in ("hybrid", "bm25"):
            return None
        if self._bm25 is None or self._bm25_version != self.store.version:
            with self._mut:
                if self._bm25 is None or self._bm25_version != self.store.version:
                    self._rebuild_bm25()
        return self._bm25

    def ingest(
        self,
        paths: Iterable[str | Path],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict:
        """Load documents, chunk them, embed chunks, and upsert into FAISS."""
        docs = load_documents(paths)
        if not docs:
            return {"ingested_documents": 0, "total_chunks": 0, "files": []}
        chunks = chunk_documents(docs, chunk_size, chunk_overlap)
        vectors = self.embedder.embed([c.text for c in chunks])

        with self._mut:
            if self.store.dim != vectors.shape[1]:
                logger.warning(
                    "Embedding dim changed (%d -> %d); rebuilding index.",
                    self.store.dim,
                    vectors.shape[1],
                )
                self.store = FaissVectorStore(dim=vectors.shape[1])
            added = self.store.add(vectors, chunks)
            if added:
                self.store.save()
            if added:
                self.cache.clear()
                # BM25 is rebuilt lazily on next query; invalidate the snapshot now.
                self._bm25_version = -1

        return {
            "ingested_documents": len(docs),
            "total_chunks": added,
            "files": [d.metadata.get("source", "") for d in docs],
        }

    def delete_source(self, source: str) -> dict:
        """Remove all indexed chunks for a source path."""
        with self._mut:
            deleted = self.store.delete_by_source(source)
            if deleted:
                self.store.save()
                self.cache.clear()
                self._bm25_version = -1
        return {
            "deleted_chunks": deleted,
            "index_size": self.store.size,
            "source": source,
        }

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        source_filter: list[str] | None = None,
    ) -> RetrievalResult:
        """Run retrieval only: embed query and return scored top-k chunks."""
        retriever = Retriever(
            self.embedder, self.store, bm25=self._ensure_bm25(), reranker=self.reranker
        )
        return retriever.retrieve(
            question,
            top_k=top_k,
            score_threshold=score_threshold,
            source_filter=source_filter,
        )

    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        prompt_strategy: PromptStrategy | None = None,
    ) -> str:
        """Run generation only using retrieved context."""
        strategy = prompt_strategy or settings.prompt_strategy
        return self.generator.generate(question, hits, strategy)

    def stream_query(
        self,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        prompt_strategy: PromptStrategy | None = None,
        source_filter: list[str] | None = None,
    ) -> Iterator[dict]:
        """Yield event dicts: a 'sources' header, then 'token' deltas, then 'done'.

        The retrieval-side guards still fire; if they reject the query we
        emit a single 'done' event with the safe unknown answer instead of
        streaming tokens from a model we don't trust.
        """
        strategy = prompt_strategy or settings.prompt_strategy
        threshold = (
            score_threshold if score_threshold is not None else settings.score_threshold
        )

        retrieval = self.retrieve(
            question, top_k=top_k, score_threshold=threshold,
            source_filter=source_filter,
        )
        if not retrieval.hits or not retrieval.confident:
            reason = (
                "no_relevant_documents_retrieved" if not retrieval.hits
                else "low_retrieval_similarity"
            )
            yield {
                "type": "done",
                "answer": UNKNOWN_ANSWER,
                "rejected": True,
                "rejection_reason": reason,
                "sources": self._source_payload(retrieval.hits),
            }
            return

        yield {"type": "sources", "sources": self._source_payload(retrieval.hits)}

        chunks: list[str] = []
        for piece in self.generator.stream(question, retrieval.hits, strategy):
            chunks.append(piece)
            yield {"type": "token", "delta": piece}

        full_answer = "".join(chunks).strip()
        if not full_answer or full_answer == UNKNOWN_ANSWER:
            yield {
                "type": "done",
                "answer": UNKNOWN_ANSWER,
                "rejected": True,
                "rejection_reason": "empty_or_unknown_generation",
            }
            return

        grounding = validate_grounding(
            full_answer, retrieval.hits, min_coverage=settings.grounding_min_coverage
        )
        confidence = score_confidence(full_answer, retrieval.hits, threshold)
        if not grounding.supported or confidence.score < settings.confidence_threshold:
            yield {
                "type": "done",
                "answer": UNKNOWN_ANSWER,
                "rejected": True,
                "rejection_reason": (
                    grounding.reason if not grounding.supported
                    else "confidence_below_threshold"
                ),
                "confidence_score": confidence.score,
                "grounding_coverage": grounding.coverage,
            }
            return

        yield {
            "type": "done",
            "answer": full_answer,
            "rejected": False,
            "rejection_reason": None,
            "confidence_score": confidence.score,
            "confidence_explanation": confidence.explanation,
            "grounding_coverage": grounding.coverage,
        }

    def _source_payload(self, hits: list[RetrievalHit]) -> list[dict]:
        return [
            {
                "rank": i + 1,
                "score": h.score,
                "source": h.metadata.get("source", ""),
                "chunk_id": h.chunk_id,
                "snippet": (h.text[:240] + "...") if len(h.text) > 240 else h.text,
            }
            for i, h in enumerate(hits)
        ]

    def _reject(
        self,
        reason: str,
        question: str,
        retrieval: RetrievalResult,
        confidence: ConfidenceResult | None,
        grounding: HallucinationCheck | None,
        started: float,
        top_k: int | None,
        score_threshold: float | None,
        prompt_strategy: str,
        source_filter: list[str] | None = None,
    ) -> dict:
        score = confidence.score if confidence else 0.0
        explanation = confidence.explanation if confidence else reason
        return self._build_payload(
            question=question,
            answer=UNKNOWN_ANSWER,
            retrieval=retrieval,
            confidence_score=score,
            confidence_explanation=explanation,
            rejected=True,
            rejection_reason=reason,
            grounding=grounding,
            started=started,
            top_k=top_k,
            score_threshold=score_threshold,
            prompt_strategy=prompt_strategy,
            source_filter=source_filter,
        )

    def _build_payload(
        self,
        question: str,
        answer: str,
        retrieval: RetrievalResult,
        confidence_score: float,
        confidence_explanation: str,
        rejected: bool,
        rejection_reason: str | None,
        grounding: HallucinationCheck | None,
        started: float,
        top_k: int | None,
        score_threshold: float | None,
        prompt_strategy: str,
        source_filter: list[str] | None = None,
    ) -> dict:
        elapsed_ms = (time.perf_counter() - started) * 1000
        threshold = score_threshold if score_threshold is not None else settings.score_threshold
        payload = {
            "question": question,
            "answer": answer,
            "confident": (not rejected) and confidence_score >= settings.confidence_threshold,
            "confidence_score": round(confidence_score, 4),
            "confidence_explanation": confidence_explanation,
            "rejected": rejected,
            "rejection_reason": rejection_reason,
            "sources": self._source_payload(retrieval.hits),
            "metadata": {
                "embedding_backend": settings.embedding_backend,
                "llm_backend": settings.llm_backend,
                "prompt_strategy": prompt_strategy,
                "top_k": top_k or settings.top_k,
                "score_threshold": threshold,
                "confidence_threshold": settings.confidence_threshold,
                "grounding_min_coverage": settings.grounding_min_coverage,
                "index_size": self.store.size,
                "latency_ms": round(elapsed_ms, 2),
                "retrieval": {
                    "top_score": retrieval.top_score,
                    "confident": retrieval.confident,
                    "scores": [round(h.score, 4) for h in retrieval.hits],
                    "mode": retrieval.mode,
                    "rerank_applied": retrieval.rerank_applied,
                    "mmr_applied": retrieval.mmr_applied,
                    "stages": retrieval.stages,
                    "source_filter": source_filter or [],
                },
                "grounding": {
                    "supported": grounding.supported if grounding else False,
                    "reason": grounding.reason if grounding else None,
                    "coverage": grounding.coverage if grounding else 0.0,
                    "unsupported_claims": grounding.unsupported_claims if grounding else [],
                },
            },
        }
        log_query_trace(payload)
        return payload

    def query(
        self,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        prompt_strategy: PromptStrategy | None = None,
        source_filter: list[str] | None = None,
    ) -> dict:
        """End-to-end query: retrieve, generate, validate, and trace."""
        started = time.perf_counter()
        strategy = prompt_strategy or settings.prompt_strategy
        threshold = score_threshold if score_threshold is not None else settings.score_threshold

        cache_key = CacheKey(
            question=question.strip(),
            top_k=top_k,
            score_threshold=threshold,
            prompt_strategy=str(strategy),
            sources=tuple(sorted(source_filter)) if source_filter else (),
            index_version=self.store.version,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            cached = dict(cached)
            cached.setdefault("metadata", {})["served_from_cache"] = True
            return cached

        retrieval = self.retrieve(
            question, top_k=top_k, score_threshold=threshold,
            source_filter=source_filter,
        )
        if not retrieval.hits:
            return self._reject(
                "no_relevant_documents_retrieved",
                question, retrieval, None, None, started,
                top_k, threshold, strategy, source_filter,
            )
        if not retrieval.confident:
            return self._reject(
                "low_retrieval_similarity",
                question, retrieval, None, None, started,
                top_k, threshold, strategy, source_filter,
            )

        answer = self.generate(question, retrieval.hits, strategy).strip()
        if not answer or answer == UNKNOWN_ANSWER:
            return self._reject(
                "empty_or_unknown_generation",
                question, retrieval, None, None, started,
                top_k, threshold, strategy, source_filter,
            )

        confidence = score_confidence(answer, retrieval.hits, threshold)
        grounding = validate_grounding(
            answer,
            retrieval.hits,
            min_coverage=settings.grounding_min_coverage,
        )
        if not grounding.supported:
            return self._reject(
                grounding.reason,
                question, retrieval, confidence, grounding, started,
                top_k, threshold, strategy, source_filter,
            )
        if confidence.score < settings.confidence_threshold:
            return self._reject(
                "confidence_below_threshold",
                question, retrieval, confidence, grounding, started,
                top_k, threshold, strategy, source_filter,
            )

        payload = self._build_payload(
            question=question,
            answer=answer,
            retrieval=retrieval,
            confidence_score=confidence.score,
            confidence_explanation=confidence.explanation,
            rejected=False,
            rejection_reason=None,
            grounding=grounding,
            started=started,
            top_k=top_k,
            score_threshold=threshold,
            prompt_strategy=strategy,
            source_filter=source_filter,
        )
        self.cache.put(cache_key, payload)
        return payload
