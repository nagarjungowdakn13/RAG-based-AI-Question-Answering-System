"""Top-k retrieval with confidence gating.

The retriever is a thin layer that:
  1) embeds the user query,
  2) asks FAISS for the top-k nearest chunks,
  3) reports a `confident` flag if the top score clears `score_threshold`.

The confidence flag is the first hallucination guard — if it's False
the orchestrator skips the LLM entirely and returns "I don't know".
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.pipeline.embeddings import Embedder
from app.pipeline.vector_store import FaissVectorStore, RetrievalHit


@dataclass
class RetrievalResult:
    hits: list[RetrievalHit]
    top_score: float
    confident: bool


class Retriever:
    def __init__(self, embedder: Embedder, store: FaissVectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        k = top_k or settings.top_k
        thresh = score_threshold if score_threshold is not None else settings.score_threshold

        q_vec = self.embedder.embed([query])
        hits = self.store.search(q_vec, top_k=k)
        top = hits[0].score if hits else 0.0
        return RetrievalResult(hits=hits, top_score=top, confident=top >= thresh)
