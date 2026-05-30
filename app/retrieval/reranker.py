"""Optional cross-encoder reranker.

A bi-encoder (the default sentence-transformers embedder) is fast but
treats the query and document independently. A cross-encoder reads the
pair jointly and is consistently more accurate at the cost of latency.

We pull a larger candidate set from the dense+sparse stage, then let the
cross-encoder pick the final top-k. The whole thing is opt-in via
`RERANKER_ENABLED=true`; if the model can't be loaded we degrade
silently to the unreranked order rather than blowing up the request.
"""
from __future__ import annotations

from typing import Protocol

from app.config import settings
from app.logger import get_logger
from app.pipeline.vector_store import RetrievalHit

logger = get_logger(__name__)


class Reranker(Protocol):
    available: bool

    def rerank(
        self, query: str, hits: list[RetrievalHit], top_k: int
    ) -> list[RetrievalHit]: ...


class NoopReranker:
    available = False

    def rerank(
        self, query: str, hits: list[RetrievalHit], top_k: int
    ) -> list[RetrievalHit]:
        return hits[:top_k]


class CrossEncoderReranker:
    """Lazy-loaded sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or settings.reranker_model
        logger.info("Loading cross-encoder reranker: %s", self.model_name)
        self._model = CrossEncoder(self.model_name)
        self.available = True

    def rerank(
        self, query: str, hits: list[RetrievalHit], top_k: int
    ) -> list[RetrievalHit]:
        if not hits or top_k <= 0:
            return []
        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(pairs)
        scored = sorted(zip(hits, scores), key=lambda kv: float(kv[1]), reverse=True)
        out: list[RetrievalHit] = []
        for hit, score in scored[:top_k]:
            out.append(
                RetrievalHit(
                    score=float(score),
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    metadata=hit.metadata,
                )
            )
        return out


def build_reranker() -> Reranker:
    if not settings.reranker_enabled:
        return NoopReranker()
    try:
        return CrossEncoderReranker()
    except Exception as e:
        logger.warning(
            "Cross-encoder reranker unavailable (%s); proceeding without rerank.", e
        )
        return NoopReranker()
