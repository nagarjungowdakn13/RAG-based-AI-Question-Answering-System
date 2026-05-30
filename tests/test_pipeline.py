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
from app.retrieval.confidence import score_confidence
from app.retrieval.hallucination import validate_grounding


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


def test_confidence_and_grounding_detect_supported_answer():
    from app.pipeline.vector_store import RetrievalHit

    hit = RetrievalHit(
        score=0.8,
        chunk_id="c1",
        text="Arthur Samuel coined the term machine learning in 1959.",
        metadata={"source": "machine_learning.txt"},
    )
    answer = "Arthur Samuel coined the term machine learning in 1959. [#1]"

    conf = score_confidence(answer, [hit], retrieval_threshold=0.3)
    grounding = validate_grounding(answer, [hit], min_coverage=0.35)

    assert conf.score > 0.6
    assert "context_coverage" in conf.explanation
    assert grounding.supported is True


def test_grounding_rejects_unsupported_answer():
    from app.pipeline.vector_store import RetrievalHit

    hit = RetrievalHit(
        score=0.8,
        chunk_id="c1",
        text="RAG retrieves external context before answering.",
        metadata={"source": "rag.txt"},
    )
    grounding = validate_grounding(
        "The system was invented in Paris by Ada Lovelace in 1843. [#1]",
        [hit],
        min_coverage=0.35,
    )

    assert grounding.supported is False
    assert grounding.unsupported_claims


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


def test_bm25_ranks_lexical_match_first():
    from app.retrieval.bm25 import BM25Index

    corpus = [
        "The transformer architecture relies on self-attention.",
        "BM25 is a sparse lexical retriever based on term frequency.",
        "Recipes for chocolate cake often include cocoa powder.",
    ]
    bm25 = BM25Index()
    bm25.build(corpus)
    hits = bm25.search("BM25 lexical retrieval", top_k=2)
    assert hits, "expected at least one BM25 hit"
    assert hits[0].row == 1


def test_rrf_fuses_independent_rankings():
    from app.retrieval.hybrid import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(
        {"dense": [10, 11, 12], "bm25": [12, 13, 10]}, k=60, top_n=3
    )
    rows = [f.row for f in fused]
    # Rows appearing in both lists should rank first.
    assert rows[0] in (10, 12)
    assert set(rows[:2]) == {10, 12}


def test_mmr_prefers_diverse_candidates():
    import numpy as np
    from app.retrieval.mmr import mmr_select

    # candidate 0 ≈ candidate 1 (near-duplicate); 2 is diverse.
    candidate_vecs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.10, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    candidate_vecs /= np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    # lambda=0.3 leans toward diversity so the near-duplicate of #0 is rejected.
    picked = mmr_select(query, candidate_vecs, [0, 1, 2], top_k=2, lambda_mult=0.3)
    assert picked[0] == 0
    assert picked[1] == 2  # MMR should pick the diverse one over the duplicate.


def test_query_cache_invalidates_on_version_bump():
    from app.pipeline.query_cache import CacheKey, QueryCache

    cache = QueryCache(maxsize=4)
    k1 = CacheKey("q", 4, 0.3, "strict", (), index_version=1)
    k1_v2 = CacheKey("q", 4, 0.3, "strict", (), index_version=2)
    cache.put(k1, {"answer": "v1"})
    assert cache.get(k1)["answer"] == "v1"
    assert cache.get(k1_v2) is None  # version change → miss
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_query_cache_disabled_when_size_zero():
    from app.pipeline.query_cache import CacheKey, QueryCache

    cache = QueryCache(maxsize=0)
    k = CacheKey("q", 4, 0.3, "strict", (), index_version=1)
    cache.put(k, {"answer": "v"})
    assert cache.get(k) is None


def test_rate_limiter_blocks_after_burst():
    from app.rate_limit import TokenBucketLimiter

    limiter = TokenBucketLimiter(rate_per_minute=60, burst=2)
    assert limiter.allow("1.2.3.4") == (True, 0.0)
    assert limiter.allow("1.2.3.4")[0] is True
    allowed, retry = limiter.allow("1.2.3.4")
    assert allowed is False
    assert retry > 0


def test_rate_limiter_disabled_when_rate_zero():
    from app.rate_limit import TokenBucketLimiter

    limiter = TokenBucketLimiter(rate_per_minute=0, burst=1)
    for _ in range(50):
        assert limiter.allow("1.2.3.4") == (True, 0.0)


def test_store_rows_for_sources_matches_basename():
    from app.pipeline.vector_store import FaissVectorStore

    store = FaissVectorStore(dim=2)
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    chunks = [
        Chunk("a", "alpha", {"source": "/abs/path/transformers.txt", "filename": "transformers.txt"}),
        Chunk("b", "bravo", {"source": "/abs/path/rag_systems.txt", "filename": "rag_systems.txt"}),
    ]
    store.add(vectors, chunks)
    assert store.rows_for_sources(["transformers.txt"]) == [0]
    assert store.rows_for_sources(["RAG_SYSTEMS.txt"]) == [1]
    assert store.rows_for_sources(["nothing.txt"]) == []


def test_vector_store_version_bumps_on_mutation():
    from app.pipeline.vector_store import FaissVectorStore

    store = FaissVectorStore(dim=2)
    v0 = store.version
    store.add(
        np.asarray([[1.0, 0.0]], dtype="float32"),
        [Chunk("a", "alpha", {"source": "x.txt"})],
    )
    assert store.version > v0
    v1 = store.version
    store.delete_by_source("x.txt")
    assert store.version > v1


@pytest.mark.slow
def test_source_filter_restricts_retrieval():
    from app.pipeline.rag import RAGPipeline

    rag = RAGPipeline.instance()
    rag.ingest([DOCS_DIR])
    out = rag.query(
        "what is attention",
        score_threshold=0.05,
        source_filter=["rag_systems.txt"],
    )
    for src in out["sources"]:
        assert "rag_systems.txt" in src["source"].lower()


@pytest.mark.slow
def test_query_cache_hit_on_repeat_query():
    from app.pipeline.rag import RAGPipeline

    rag = RAGPipeline.instance()
    rag.ingest([DOCS_DIR])
    rag.cache.clear()
    rag.query("Who coined the term machine learning?")
    stats_before = rag.cache.stats()
    rag.query("Who coined the term machine learning?")
    stats_after = rag.cache.stats()
    assert stats_after["hits"] == stats_before["hits"] + 1


@pytest.mark.slow
def test_streaming_emits_done_event():
    """Streaming endpoint yields a 'done' event with the full answer."""
    from app.pipeline.rag import RAGPipeline

    rag = RAGPipeline.instance()
    rag.ingest([DOCS_DIR])
    events = list(rag.stream_query("Who coined the term machine learning?"))
    types = [e["type"] for e in events]
    assert "done" in types
    done = events[-1]
    assert done["type"] == "done"
    assert done.get("rejected") is False
    assert "samuel" in done.get("answer", "").lower()


def test_health_endpoint_includes_new_fields():
    from app.main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    for field in (
        "uptime_seconds", "index_version", "embedding_dim",
        "retrieval_mode", "cache", "reranker",
    ):
        assert field in body, f"missing /health field: {field}"
