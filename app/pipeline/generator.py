"""Answer generation.

Two interchangeable generators:
  - OpenAIGenerator: instructs the model to answer ONLY from numbered
    context blocks and to emit `[#i]` citations. The system prompt is
    designed to fail closed — if the answer isn't in context, the model
    is told to say "I don't know based on the provided context."
  - ExtractiveGenerator: zero-LLM fallback. Returns the highest-ranked
    chunk verbatim with a clear notice. Useful for offline tests, CI,
    and as a sane default when no API key is configured.
"""
from __future__ import annotations

from typing import Protocol

from app.config import settings
from app.logger import get_logger
from app.pipeline.vector_store import RetrievalHit

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a precise question-answering assistant for a retrieval-augmented system.

RULES (non-negotiable):
1. Answer ONLY using facts present in the numbered CONTEXT blocks below.
2. If the answer is not contained in the context, reply EXACTLY:
   "I don't know based on the provided context."
3. Never invent facts, names, numbers, dates, or citations.
4. Cite the context blocks you used inline with tokens like [#1], [#2].
5. Keep the answer concise (1–4 sentences) unless the user explicitly asks for detail.
"""

UNKNOWN_ANSWER = "I don't know based on the provided context."


def _format_context(hits: list[RetrievalHit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        src = h.metadata.get("filename") or h.metadata.get("source") or "unknown"
        parts.append(f"[#{i}] (source: {src})\n{h.text.strip()}")
    return "\n\n".join(parts)


class Generator(Protocol):
    def generate(self, question: str, hits: list[RetrievalHit]) -> str: ...


class OpenAIGenerator:
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI generation")
        self.model = model or settings.openai_chat_model
        self._client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, question: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return UNKNOWN_ANSWER
        context = _format_context(hits)
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        return (resp.choices[0].message.content or "").strip() or UNKNOWN_ANSWER


class ExtractiveGenerator:
    """No-LLM fallback. Returns the highest-scoring chunk verbatim.

    This is intentionally honest: it tells the user the answer was
    extracted, not synthesized, so they don't mistake it for an LLM
    response. Lets the rest of the pipeline run end-to-end without a key.
    """

    def generate(self, question: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return UNKNOWN_ANSWER
        top = hits[0]
        src = top.metadata.get("filename") or top.metadata.get("source") or "unknown"
        snippet = top.text.strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + "…"
        return (
            f"[extractive answer — no LLM configured]\n"
            f"Most relevant passage from {src} [#1]:\n\n{snippet}"
        )


def build_generator() -> Generator:
    if settings.llm_backend == "openai":
        try:
            return OpenAIGenerator()
        except Exception as e:
            logger.warning("OpenAI generator unavailable (%s); falling back to extractive.", e)
    return ExtractiveGenerator()
