"""Answer generation boundary.

Generation is intentionally independent of retrieval. It accepts only a
question plus retrieved chunks, and it never reaches into the vector store.
Prompt construction lives in `app.generation.prompts` so prompt strategies
can be tested without changing model client code.
"""
from __future__ import annotations

from typing import Protocol

from app.config import settings
from app.generation.prompts import (
    UNKNOWN_ANSWER,
    PromptManager,
    PromptStrategy,
    format_context,
)
from app.logger import get_logger
from app.pipeline.vector_store import RetrievalHit

logger = get_logger(__name__)


class Generator(Protocol):
    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        prompt_strategy: PromptStrategy = "strict",
    ) -> str: ...


class OpenAIGenerator:
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI generation")
        self.model = model or settings.openai_chat_model
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._prompts = PromptManager()

    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        prompt_strategy: PromptStrategy = "strict",
    ) -> str:
        if not hits:
            return UNKNOWN_ANSWER
        prompt = self._prompts.build(question, hits, prompt_strategy)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        return (resp.choices[0].message.content or "").strip() or UNKNOWN_ANSWER


class ExtractiveGenerator:
    """No-LLM fallback that returns the top-ranked retrieved passage."""

    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        prompt_strategy: PromptStrategy = "strict",
    ) -> str:
        if not hits:
            return UNKNOWN_ANSWER
        top = hits[0]
        src = top.metadata.get("filename") or top.metadata.get("source") or "unknown"
        snippet = top.text.strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + "..."
        return (
            "[extractive answer - no LLM configured]\n"
            f"Most relevant passage from {src} [#1]:\n\n{snippet}"
        )


def build_generator() -> Generator:
    if settings.llm_backend == "openai":
        try:
            return OpenAIGenerator()
        except Exception as e:
            logger.warning("OpenAI generator unavailable (%s); falling back to extractive.", e)
    return ExtractiveGenerator()


__all__ = [
    "UNKNOWN_ANSWER",
    "Generator",
    "OpenAIGenerator",
    "ExtractiveGenerator",
    "build_generator",
    "format_context",
]
