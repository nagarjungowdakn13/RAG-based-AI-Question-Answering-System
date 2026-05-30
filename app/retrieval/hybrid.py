"""Reciprocal Rank Fusion of dense + sparse retrievers.

RRF (Cormack et al., 2009) ignores raw scores and ranks only by position,
so we don't have to normalize across two different score scales. It's
boring, but it's the production default for hybrid search and beats
arithmetic-mean / weighted-sum on most corpora.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusedHit:
    row: int
    score: float
    sources: tuple[str, ...]


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[int]],
    k: int = 60,
    top_n: int | None = None,
) -> list[FusedHit]:
    """Fuse multiple ranked lists by RRF score = sum(1 / (k + rank)).

    `ranked_lists` maps a retriever name to its ordered list of row IDs
    (rank 0 = best). The returned list is sorted by fused score, descending.
    """
    accum: dict[int, float] = {}
    contributors: dict[int, list[str]] = {}
    for name, rows in ranked_lists.items():
        for rank, row in enumerate(rows):
            accum[row] = accum.get(row, 0.0) + 1.0 / (k + rank + 1)
            contributors.setdefault(row, []).append(name)
    fused = [
        FusedHit(row=row, score=score, sources=tuple(contributors[row]))
        for row, score in accum.items()
    ]
    fused.sort(key=lambda h: h.score, reverse=True)
    if top_n is not None:
        fused = fused[:top_n]
    return fused
