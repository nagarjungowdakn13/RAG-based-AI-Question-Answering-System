"""Maximal Marginal Relevance reranking.

With chunk overlap turned on, naive top-k often surfaces several
near-duplicate chunks. MMR (Carbonell & Goldstein, 1998) trades a little
relevance for diversity by picking each next chunk to maximize:

    lambda * sim(query, chunk) - (1 - lambda) * max(sim(chunk, picked))

`lambda=1.0` is pure top-k; `lambda=0.0` is pure diversity. The default
0.6 keeps things relevance-leaning.
"""
from __future__ import annotations

import numpy as np


def mmr_select(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_rows: list[int],
    top_k: int,
    lambda_mult: float = 0.6,
) -> list[int]:
    """Return the row IDs picked by MMR, in selection order.

    Vectors are assumed L2-normalized (our embedder normalizes), so dot
    product equals cosine similarity.
    """
    if not candidate_rows or top_k <= 0:
        return []
    if candidate_vecs.ndim != 2 or candidate_vecs.shape[0] != len(candidate_rows):
        raise ValueError("candidate_vecs shape does not match candidate_rows length")

    q = query_vec.reshape(-1).astype("float32")
    q_sim = candidate_vecs @ q  # (n,)
    selected_local: list[int] = []
    remaining = list(range(len(candidate_rows)))

    # Pick the highest-relevance candidate first; MMR formula degenerates
    # to argmax(q_sim) when nothing has been picked yet.
    first = int(np.argmax(q_sim))
    selected_local.append(first)
    remaining.remove(first)

    while remaining and len(selected_local) < top_k:
        sel_matrix = candidate_vecs[selected_local]  # (s, d)
        # For each remaining candidate, its max similarity to anything picked.
        rem_vecs = candidate_vecs[remaining]
        sims_to_selected = rem_vecs @ sel_matrix.T  # (r, s)
        max_redundancy = sims_to_selected.max(axis=1)
        relevance = q_sim[remaining]
        mmr_scores = lambda_mult * relevance - (1.0 - lambda_mult) * max_redundancy
        pick_idx = int(np.argmax(mmr_scores))
        selected_local.append(remaining[pick_idx])
        remaining.pop(pick_idx)

    return [candidate_rows[i] for i in selected_local]
