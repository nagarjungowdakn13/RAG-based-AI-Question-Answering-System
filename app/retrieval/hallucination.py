"""Post-generation grounding checks."""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.generation.prompts import UNKNOWN_ANSWER
from app.pipeline.vector_store import RetrievalHit
from app.retrieval.confidence import context_coverage

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class HallucinationCheck:
    supported: bool
    reason: str
    coverage: float
    unsupported_claims: list[str]


def validate_grounding(
    answer: str,
    hits: list[RetrievalHit],
    min_coverage: float,
) -> HallucinationCheck:
    """Detect unsupported generated claims using lexical grounding.

    This is deliberately conservative: if enough answer terms are not
    recoverable from retrieved context, the orchestrator rejects the answer.
    """
    clean = answer.strip()
    if not clean or clean == UNKNOWN_ANSWER:
        return HallucinationCheck(False, "empty_or_unknown_answer", 0.0, [])
    if not hits:
        return HallucinationCheck(False, "no_retrieved_context", 0.0, [clean])

    coverage = context_coverage(clean, hits)
    unsupported: list[str] = []
    for sentence in _SENTENCE_RE.split(clean):
        sentence = sentence.strip()
        if len(sentence.split()) < 5:
            continue
        if context_coverage(sentence, hits) < min_coverage:
            unsupported.append(sentence)

    if coverage < min_coverage:
        return HallucinationCheck(
            False,
            f"answer_context_coverage_below_{min_coverage:.2f}",
            round(coverage, 4),
            unsupported[:5],
        )
    if unsupported:
        return HallucinationCheck(
            False,
            "unsupported_sentence_claims",
            round(coverage, 4),
            unsupported[:5],
        )
    return HallucinationCheck(True, "grounded_in_retrieved_context", round(coverage, 4), [])

