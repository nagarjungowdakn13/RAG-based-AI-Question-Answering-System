"""Top-k retrieval with hybrid search, MMR, and optional reranking.

The retriever composes several stages, each individually toggleable:

  1. Dense ANN over FAISS.
  2. Sparse BM25 over the same chunks (kept in lockstep with the index
     via the vector store's monotonic `version`).
  3. Reciprocal Rank Fusion of the two lists.
  4. Optional source filter (restrict to a caller-supplied set of files).
  5. Optional MMR diversification on the fused candidate pool.
  6. Optional cross-encoder reranking on the surviving candidates.
  7. Confidence gate on the final top score.

Stages 2 and 5-6 degrade silently when their inputs are unavailable
(empty corpus, no candidates, no reranker model) so the pipeline never
fails closed for the wrong reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.config import settings
from app.logger import get_logger
from app.pipeline.embeddings import Embedder
from app.pipeline.vector_store import FaissVectorStore, RetrievalHit
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.mmr import mmr_select
from app.retrieval.reranker import Reranker

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    hits: list[RetrievalHit]
    top_score: float
    confident: bool
    mode: str = "dense"
    rerank_applied: bool = False
    mmr_applied: bool = False
    stages: dict = field(default_factory=dict)


class Retriever:
    """Composable retrieval. Holds references; owns no mutable index state."""

    def __init__(
        self,
        embedder: Embedder,
        store: FaissVectorStore,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
    ):
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker

    # ─── pipeline ────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        source_filter: list[str] | None = None,
    ) -> RetrievalResult:
        k = top_k or settings.top_k
        thresh = score_threshold if score_threshold is not None else settings.score_threshold
        mode = settings.retrieval_mode
        stages: dict = {}

        if self.store.size == 0:
            return RetrievalResult(hits=[], top_score=0.0, confident=False, mode=mode)

        allowed_rows = None
        if source_filter:
            allowed_rows = set(self.store.rows_for_sources(source_filter))
            stages["source_filter_size"] = len(allowed_rows)
            if not allowed_rows:
                logger.info("source_filter matched no chunks: %s", source_filter)
                return RetrievalResult(
                    hits=[], top_score=0.0, confident=False, mode=mode,
                    stages=stages,
                )

        # Stage 1+2: fetch candidates from both retrievers.
        candidate_pool_size = max(k * settings.hybrid_candidate_multiplier, k)
        if settings.reranker_enabled and self.reranker and self.reranker.available:
            candidate_pool_size = max(
                candidate_pool_size, k * settings.reranker_candidate_multiplier
            )

        q_vec = self.embedder.embed([query])
        dense_rows = self._dense_rows(q_vec, candidate_pool_size, allowed_rows)
        sparse_rows = self._sparse_rows(query, candidate_pool_size, allowed_rows) if mode == "hybrid" else []
        stages["dense_candidates"] = len(dense_rows)
        stages["sparse_candidates"] = len(sparse_rows)

        # Stage 3: fuse (or just use dense ranking when mode=dense / BM25 empty).
        if sparse_rows:
            ranked_lists = {"dense": dense_rows, "bm25": sparse_rows}
            fused = reciprocal_rank_fusion(ranked_lists, k=settings.rrf_k)
            fused_rows = [f.row for f in fused]
        else:
            fused_rows = dense_rows
        stages["fused_candidates"] = len(fused_rows)

        if not fused_rows:
            return RetrievalResult(
                hits=[], top_score=0.0, confident=False, mode=mode, stages=stages
            )

        # Stage 5: MMR. We need the candidate vectors anyway, so we materialize
        # them once and reuse the matrix for the final scoring.
        cand_vecs = np.asarray(
            [self.store.get_vector(r) for r in fused_rows], dtype="float32"
        )
        mmr_applied = False
        if settings.mmr_enabled and len(fused_rows) > k:
            select_k = max(
                k,
                k * (settings.reranker_candidate_multiplier if (
                    settings.reranker_enabled and self.reranker and self.reranker.available
                ) else 1),
            )
            select_k = min(select_k, len(fused_rows))
            picked = mmr_select(
                q_vec[0], cand_vecs, fused_rows,
                top_k=select_k, lambda_mult=settings.mmr_lambda,
            )
            row_to_vec = {fused_rows[i]: cand_vecs[i] for i in range(len(fused_rows))}
            fused_rows = picked
            cand_vecs = np.asarray([row_to_vec[r] for r in fused_rows], dtype="float32")
            mmr_applied = True

        # Always score the surviving candidates against the query so the
        # returned hit.score is a true cosine value (RRF score is not).
        sims = (cand_vecs @ q_vec[0]).tolist()
        hits: list[RetrievalHit] = []
        for r, s in zip(fused_rows, sims):
            base = self.store.get_hit(r)
            hits.append(
                RetrievalHit(
                    score=float(s),
                    chunk_id=base.chunk_id,
                    text=base.text,
                    metadata=base.metadata,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)

        # Stage 6: cross-encoder rerank, if configured.
        rerank_applied = False
        if (
            settings.reranker_enabled
            and self.reranker
            and self.reranker.available
            and len(hits) > 1
        ):
            hits = self.reranker.rerank(query, hits, top_k=k)
            rerank_applied = True
        else:
            hits = hits[:k]

        top = hits[0].score if hits else 0.0
        return RetrievalResult(
            hits=hits,
            top_score=top,
            confident=top >= thresh,
            mode=mode,
            rerank_applied=rerank_applied,
            mmr_applied=mmr_applied,
            stages=stages,
        )

    # ─── stage helpers ───────────────────────────────────────────────
    def _dense_rows(
        self, q_vec: np.ndarray, n: int, allowed_rows: set[int] | None
    ) -> list[int]:
        if allowed_rows is not None:
            # Restricted set: compute exact cosine over the subset.
            rows = list(allowed_rows)
            vecs = np.asarray([self.store.get_vector(r) for r in rows], dtype="float32")
            sims = (vecs @ q_vec[0]).tolist()
            ranked = sorted(zip(rows, sims), key=lambda kv: kv[1], reverse=True)[:n]
            return [r for r, _ in ranked]
        # Unrestricted: ask FAISS for n nearest.
        if q_vec.dtype != np.float32:
            q_vec = q_vec.astype("float32")
        size = self.store.size
        k = min(n, size)
        scores, idxs = self.store.index.search(q_vec, k)
        return [int(i) for i in idxs[0] if i != -1]

    def _sparse_rows(
        self, query: str, n: int, allowed_rows: set[int] | None
    ) -> list[int]:
        if self.bm25 is None or self.bm25.size == 0:
            return []
        hits = self.bm25.search(query, top_k=n if allowed_rows is None else n * 4)
        if allowed_rows is None:
            return [h.row for h in hits]
        return [h.row for h in hits if h.row in allowed_rows][:n]
