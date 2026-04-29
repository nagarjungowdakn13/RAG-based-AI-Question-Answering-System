"""RAG orchestrator: retrieve, generate, validate, observe.

This facade keeps ingestion, retrieval, and generation independently tunable:
retrieval produces scored chunks; generation consumes only those chunks; the
post-generation reliability layer decides whether the answer is grounded.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterable

from app.config import settings
from app.generation.prompts import PromptStrategy
from app.logger import get_logger
from app.observability.query_trace import log_query_trace
from app.pipeline.chunking import chunk_documents
from app.pipeline.embeddings import Embedder, build_embedder
from app.pipeline.generator import UNKNOWN_ANSWER, Generator, build_generator
from app.pipeline.ingestion import load_documents
from app.pipeline.retriever import Retriever, RetrievalResult
from app.pipeline.vector_store import FaissVectorStore, RetrievalHit
from app.retrieval.confidence import ConfidenceResult, score_confidence
from app.retrieval.hallucination import HallucinationCheck, validate_grounding

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
    ) -> RetrievalResult:
        """Run retrieval only: embed query and return scored top-k chunks."""
        retriever = Retriever(self.embedder, self.store)
        return retriever.retrieve(question, top_k=top_k, score_threshold=score_threshold)

    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        prompt_strategy: PromptStrategy | None = None,
    ) -> str:
        """Run generation only using retrieved context."""
        strategy = prompt_strategy or settings.prompt_strategy
        return self.generator.generate(question, hits, strategy)

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
    ) -> dict:
        """End-to-end query: retrieve, generate, validate, and trace."""
        started = time.perf_counter()
        strategy = prompt_strategy or settings.prompt_strategy
        threshold = score_threshold if score_threshold is not None else settings.score_threshold

        retrieval = self.retrieve(question, top_k=top_k, score_threshold=threshold)
        if not retrieval.hits:
            return self._reject(
                "no_relevant_documents_retrieved",
                question,
                retrieval,
                None,
                None,
                started,
                top_k,
                threshold,
                strategy,
            )
        if not retrieval.confident:
            return self._reject(
                "low_retrieval_similarity",
                question,
                retrieval,
                None,
                None,
                started,
                top_k,
                threshold,
                strategy,
            )

        answer = self.generate(question, retrieval.hits, strategy).strip()
        if not answer or answer == UNKNOWN_ANSWER:
            return self._reject(
                "empty_or_unknown_generation",
                question,
                retrieval,
                None,
                None,
                started,
                top_k,
                threshold,
                strategy,
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
                question,
                retrieval,
                confidence,
                grounding,
                started,
                top_k,
                threshold,
                strategy,
            )
        if confidence.score < settings.confidence_threshold:
            return self._reject(
                "confidence_below_threshold",
                question,
                retrieval,
                confidence,
                grounding,
                started,
                top_k,
                threshold,
                strategy,
            )

        return self._build_payload(
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
        )
