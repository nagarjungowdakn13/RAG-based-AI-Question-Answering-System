"""Prompt strategy management.

Prompt text is intentionally isolated from model clients so generation
experiments can change instructions without touching retrieval or API code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.pipeline.vector_store import RetrievalHit

PromptStrategy = Literal["strict", "fallback"]

UNKNOWN_ANSWER = "I don't know based on the provided context."


@dataclass(frozen=True)
class PromptBundle:
    strategy: PromptStrategy
    system: str
    user: str


STRICT_SYSTEM_PROMPT = """You are a precise question-answering assistant for a retrieval-augmented system.

RULES (non-negotiable):
1. Answer ONLY using facts present in the numbered CONTEXT blocks below.
2. If the answer is not contained in the context, reply EXACTLY:
   "I don't know based on the provided context."
3. Never invent facts, names, numbers, dates, or citations.
4. Cite the context blocks you used inline with tokens like [#1], [#2].
5. Keep the answer concise (1-4 sentences) unless the user explicitly asks for detail.
"""

FALLBACK_SYSTEM_PROMPT = """You are a cautious assistant in a RAG system.

Use the provided context first. If the context is weak or incomplete, say:
"I don't know based on the provided context."
Do not add facts from memory. Prefer a short answer with citations.
"""


def format_context(hits: list[RetrievalHit]) -> str:
    """Format retrieved chunks as numbered context blocks."""
    parts = []
    for i, h in enumerate(hits, 1):
        src = h.metadata.get("filename") or h.metadata.get("source") or "unknown"
        parts.append(f"[#{i}] (source: {src}, score: {h.score:.3f})\n{h.text.strip()}")
    return "\n\n".join(parts)


class PromptManager:
    """Build prompts for named RAG generation strategies."""

    def build(
        self,
        question: str,
        hits: list[RetrievalHit],
        strategy: PromptStrategy = "strict",
    ) -> PromptBundle:
        context = format_context(hits)
        if strategy == "fallback":
            system = FALLBACK_SYSTEM_PROMPT
        else:
            strategy = "strict"
            system = STRICT_SYSTEM_PROMPT
        user = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        return PromptBundle(strategy=strategy, system=system, user=user)

