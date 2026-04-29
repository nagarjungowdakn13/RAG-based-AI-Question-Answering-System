"""Confidence scoring for RAG answers.

The score combines retrieval quality, grounding coverage, and generation
heuristics. Each component stays simple and explainable so failures are
auditable during research experiments.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass

from app.generation.prompts import UNKNOWN_ANSWER
from app.pipeline.vector_store import RetrievalHit

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_CITATION_RE = re.compile(r"\[#\d+\]")
_PUNCT = str.maketrans("", "", string.punctuation)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "how", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "what", "when", "where", "which",
    "who", "why", "with", "you", "your", "about", "based", "provided",
    "context", "answer", "using", "does", "do",
}


@dataclass(frozen=True)
class ConfidenceResult:
    score: float
    explanation: str
    retrieval_score: float
    coverage_score: float
    output_score: float


def _content_words(text: str) -> set[str]:
    words = {
        w.lower().translate(_PUNCT)
        for w in _WORD_RE.findall(text)
        if len(w) > 2
    }
    return {w for w in words if w and w not in _STOPWORDS}


def context_coverage(answer: str, hits: list[RetrievalHit]) -> float:
    """Fraction of answer content words found in retrieved context."""
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    context_words = _content_words("\n".join(h.text for h in hits))
    if not context_words:
        return 0.0
    return len(answer_words & context_words) / len(answer_words)


def output_heuristic_score(answer: str, hits: list[RetrievalHit]) -> float:
    """Score answer shape: non-empty, cited, not overly evasive."""
    clean = answer.strip()
    if not clean or clean == UNKNOWN_ANSWER:
        return 0.0
    score = 0.45
    if _CITATION_RE.search(clean):
        score += 0.35
    if len(clean.split()) >= 6:
        score += 0.10
    if hits and any((h.metadata.get("filename") or "") in clean for h in hits):
        score += 0.05
    if len(clean) > 1600:
        score -= 0.15
    return max(0.0, min(score, 1.0))


def score_confidence(
    answer: str,
    hits: list[RetrievalHit],
    retrieval_threshold: float,
) -> ConfidenceResult:
    """Return a weighted, explainable confidence score in [0, 1]."""
    top_score = hits[0].score if hits else 0.0
    retrieval_score = min(max(top_score / max(retrieval_threshold, 0.01), 0.0), 1.0)
    coverage = context_coverage(answer, hits)
    output_score = output_heuristic_score(answer, hits)
    score = (0.45 * retrieval_score) + (0.35 * coverage) + (0.20 * output_score)
    score = max(0.0, min(score, 1.0))
    explanation = (
        f"retrieval={retrieval_score:.2f} (top_score={top_score:.3f}), "
        f"context_coverage={coverage:.2f}, output_heuristics={output_score:.2f}"
    )
    return ConfidenceResult(
        score=round(score, 4),
        explanation=explanation,
        retrieval_score=round(retrieval_score, 4),
        coverage_score=round(coverage, 4),
        output_score=round(output_score, 4),
    )

