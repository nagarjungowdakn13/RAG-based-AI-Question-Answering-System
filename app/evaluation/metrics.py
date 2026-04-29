"""Three orthogonal metrics measuring complementary failure modes.

  - Exact Match (EM): brittle but unambiguous. Catches near-perfect answers.
  - Semantic similarity: cosine sim of sentence embeddings. Tolerant to
    paraphrase; the right metric for free-form generative answers.
  - Retrieval accuracy@k: was the *right document* surfaced at all?
    A miss here means EM/sim failures are guaranteed — useful to
    distinguish retrieval bugs from generation bugs.
"""
from __future__ import annotations

import re
import string
from typing import Iterable

import numpy as np


_PUNCT = str.maketrans("", "", string.punctuation)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def semantic_similarity(pred_vec: np.ndarray, ref_vec: np.ndarray) -> float:
    """Cosine similarity. Inputs assumed L2-normalized (our embedder does this)."""
    return float(np.clip(np.dot(pred_vec, ref_vec), -1.0, 1.0))


def retrieval_hit(
    retrieved_sources: Iterable[str],
    expected_source: str | None,
    reference_answer: str,
    retrieved_texts: Iterable[str],
) -> float:
    """1 if (a) any retrieved chunk's source matches expected_source, OR
    (b) the normalized reference answer appears as a substring in any
    retrieved chunk. (b) makes the metric usable on QA datasets that
    don't carry ground-truth source IDs.
    """
    if expected_source:
        for s in retrieved_sources:
            if expected_source.lower() in (s or "").lower():
                return 1.0
    ref_norm = normalize(reference_answer)
    if ref_norm:
        for t in retrieved_texts:
            if ref_norm in normalize(t):
                return 1.0
    return 0.0


def aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"exact_match": 0.0, "semantic_similarity": 0.0, "retrieval_accuracy": 0.0}
    n = len(rows)
    return {
        "exact_match": sum(r["exact_match"] for r in rows) / n,
        "semantic_similarity": sum(r["semantic_similarity"] for r in rows) / n,
        "retrieval_accuracy": sum(r["retrieval_accuracy"] for r in rows) / n,
        "answered": sum(1 for r in rows if not r["abstained"]) / n,
    }
