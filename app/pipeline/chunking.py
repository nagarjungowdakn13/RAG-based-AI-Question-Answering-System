"""Chunking strategy.

Uses LangChain's `RecursiveCharacterTextSplitter` because it splits on
semantically meaningful boundaries first (paragraphs → sentences →
words → chars), giving better context preservation than naive fixed-size
slicing. Falls back to a stdlib splitter if the dependency is missing.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable

from app.config import settings
from app.logger import get_logger
from app.pipeline.ingestion import Document

logger = get_logger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


def _splitter(chunk_size: int, chunk_overlap: int, unit: str = "chars"):
    """Build a splitter. `unit='tokens'` uses tiktoken-based length; 'chars' uses len."""
    if unit == "tokens":
        return _token_splitter(chunk_size, chunk_overlap)
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
    except ImportError:  # pragma: no cover
        logger.warning("langchain-text-splitters not installed; using fallback splitter")

        class _Fallback:
            def __init__(self, size, overlap):
                self.size, self.overlap = size, overlap

            def split_text(self, text: str) -> list[str]:
                step = max(1, self.size - self.overlap)
                return [text[i : i + self.size] for i in range(0, len(text), step)]

        return _Fallback(chunk_size, chunk_overlap)


def _token_splitter(chunk_size: int, chunk_overlap: int):
    """Token-budgeted splitter.

    Prefer langchain's RecursiveCharacterTextSplitter with a tiktoken
    `length_function` so we keep semantic boundary preservation but measure
    in tokens. Fall back to a pure tiktoken slicer if either dep is missing.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding(settings.tiktoken_encoding)
    except Exception as e:  # pragma: no cover
        logger.warning("tiktoken unavailable (%s); reverting to char chunker", e)
        return _splitter(chunk_size * 4, chunk_overlap * 4, unit="chars")

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=lambda s: len(enc.encode(s)),
        )
    except ImportError:  # pragma: no cover
        class _TokenFallback:
            def __init__(self, size, overlap, enc):
                self.size, self.overlap, self.enc = size, overlap, enc

            def split_text(self, text: str) -> list[str]:
                ids = self.enc.encode(text)
                step = max(1, self.size - self.overlap)
                return [
                    self.enc.decode(ids[i : i + self.size])
                    for i in range(0, len(ids), step)
                ]

        return _TokenFallback(chunk_size, chunk_overlap, enc)


def chunk_documents(
    docs: Iterable[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split each document and tag every chunk with stable metadata."""
    cs = chunk_size or settings.chunk_size
    co = chunk_overlap or settings.chunk_overlap
    splitter = _splitter(cs, co, unit=settings.chunk_unit)

    chunks: list[Chunk] = []
    for doc in docs:
        pieces = splitter.split_text(doc.text)
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            cid = f"{uuid.uuid5(uuid.NAMESPACE_URL, doc.metadata.get('source', '') + str(i))}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=piece,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_size": cs,
                        "chunk_overlap": co,
                        "chunk_unit": settings.chunk_unit,
                    },
                )
            )
    logger.info(
        "Produced %d chunks (chunk_size=%d, overlap=%d, unit=%s)",
        len(chunks), cs, co, settings.chunk_unit,
    )
    return chunks
