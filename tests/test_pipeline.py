"""Smoke tests for the pipeline.

Run:  python -m pytest tests -q

These hit the real embedder (sentence-transformers) but use the
extractive generator, so no API keys are needed and tests are
deterministic.
"""
from __future__ import annotations

import os
from pathlib import Path

# Force a fully-offline configuration before any app modules are imported.
os.environ.setdefault("EMBEDDING_BACKEND", "huggingface")
os.environ.setdefault("LLM_BACKEND", "extractive")
os.environ.setdefault("INDEX_DIR", "storage/test_index")

import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.evaluation import metrics
from app.pipeline.chunking import Chunk, chunk_documents
from app.pipeline.ingestion import Document, load_documents
from app.pipeline.vector_store import FaissVectorStore


DOCS_DIR = Path("data/docs")


def test_load_documents():
    docs = load_documents([DOCS_DIR])
    assert len(docs) >= 3
    assert all(isinstance(d, Document) and d.text for d in docs)


def test_chunking_respects_size():
    docs = load_documents([DOCS_DIR])
    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 0
    # Splitter is allowed slack at boundaries; assert the spirit of the limit.
    assert all(len(c.text) <= 600 for c in chunks)


def test_normalize_and_em():
    assert metrics.normalize("  Hello, World!  ") == "hello world"
    assert metrics.exact_match("Hello, World", "hello world") == 1.0
    assert metrics.exact_match("foo", "bar") == 0.0


def test_aggregate_handles_empty():
    agg = metrics.aggregate([])
    assert agg["exact_match"] == 0.0


def test_delete_by_source_rebuilds_index():
    store = FaissVectorStore(dim=2)
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype="float32")
    chunks = [
        Chunk("a", "alpha", {"source": "one.txt"}),
        Chunk("b", "bravo", {"source": "two.txt"}),
        Chunk("c", "charlie", {"source": "one.txt"}),
    ]

    assert store.add(vectors, chunks) == 3
    assert store.delete_by_source("one.txt") == 2
    assert store.size == 1
    hits = store.search(np.asarray([0.0, 1.0], dtype="float32"), top_k=3)
    assert [h.chunk_id for h in hits] == ["b"]


def test_api_key_protects_mutating_endpoints(monkeypatch):
    from app.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "api_key", "secret")
    client = TestClient(app)

    payload = {"qa_path": "missing-eval-file.json"}

    r = client.post("/evaluate", json=payload)
    assert r.status_code == 401

    r = client.post("/evaluate", json=payload, headers={"X-API-Key": "secret"})
    assert r.status_code != 401


@pytest.mark.slow
def test_end_to_end_query():
    """Full pipeline: ingest example docs and query → confident answer."""
    from app.pipeline.rag import RAGPipeline

    rag = RAGPipeline.instance()
    rag.ingest([DOCS_DIR])
    out = rag.query("Who coined the term machine learning?")
    assert out["confident"] is True
    assert "samuel" in out["answer"].lower() or "samuel" in out["sources"][0]["snippet"].lower()


@pytest.mark.slow
def test_out_of_domain_abstains():
    from app.pipeline.rag import RAGPipeline

    rag = RAGPipeline.instance()
    rag.ingest([DOCS_DIR])
    out = rag.query("What is the capital of France?", score_threshold=0.6)
    assert out["confident"] is False
    assert "i don't know" in out["answer"].lower()


@pytest.mark.slow
def test_ingest_is_idempotent():
    """Re-ingesting the same docs must not duplicate chunks (deterministic chunk_id)."""
    from app.pipeline.rag import RAGPipeline

    rag = RAGPipeline.instance()
    first = rag.ingest([DOCS_DIR])
    size_after_first = rag.store.size
    second = rag.ingest([DOCS_DIR])
    assert second["total_chunks"] == 0
    assert rag.store.size == size_after_first
