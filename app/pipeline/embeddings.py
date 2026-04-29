"""Embedding backends.

Three interchangeable implementations behind a common `Embedder` Protocol:
  - HuggingFaceEmbedder (default): runs locally via sentence-transformers,
    no API key, deterministic, free. High semantic quality.
  - OpenAIEmbedder: hosted, higher-quality on long-form / multilingual.
  - HashingEmbedder: scikit-learn HashingVectorizer fallback. Lower
    semantic quality but zero PyTorch deps — keeps the demo running
    when the HF stack is broken (torch/transformers version mismatch).

All embeddings are L2-normalized so a FAISS `IndexFlatIP` (inner product)
gives true cosine similarity at query time — the cheapest exact-search
configuration that still produces well-calibrated scores in [0, 1].
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HuggingFaceEmbedder:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.hf_embedding_model
        logger.info("Loading HF embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        ).astype("float32")
        return _l2_normalize(vectors)


class OpenAIEmbedder:
    def __init__(self, model_name: str | None = None):
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
        self.model_name = model_name or settings.openai_embedding_model
        self._client = OpenAI(api_key=settings.openai_api_key)
        # text-embedding-3-small=1536, -3-large=3072. Probe with a tiny call.
        probe = self._client.embeddings.create(model=self.model_name, input="probe")
        self.dim = len(probe.data[0].embedding)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        vectors = np.array([d.embedding for d in resp.data], dtype="float32")
        return _l2_normalize(vectors)


class HashingEmbedder:
    """Pure-Python / sklearn fallback. Zero PyTorch dependency.

    Uses sklearn's `HashingVectorizer` with word + bigram features hashed
    into a fixed-dimensional space, then L2-normalized. It's a feature-
    hashing bag-of-words, not a learned embedder, so semantic generalization
    is weak — but lexical retrieval on the bundled docs works fine and the
    pipeline (chunking → FAISS → reject mechanism → strict prompt → source
    attribution) is fully exercised.

    Trigger: set `EMBEDDING_BACKEND=hashing` explicitly, or let
    `build_embedder` fall back automatically when the HF stack fails to
    import.
    """

    def __init__(self, dim: int = 512):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._vec = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            ngram_range=(1, 2),
            norm=None,
            lowercase=True,
        )
        logger.info("Using HashingEmbedder (dim=%d, no PyTorch)", dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        matrix = self._vec.transform(texts).toarray().astype("float32")
        return _l2_normalize(matrix)


def build_embedder() -> Embedder:
    backend = settings.embedding_backend
    if backend == "openai":
        return OpenAIEmbedder()
    if backend == "hashing":
        return HashingEmbedder()
    # huggingface (default) — fall back to hashing on any import error so
    # the demo still runs in a broken torch/transformers env.
    try:
        return HuggingFaceEmbedder()
    except Exception as e:
        logger.warning(
            "HuggingFace embedder unavailable (%s). Falling back to "
            "HashingEmbedder. For proper semantic embeddings, run "
            "`pip install --upgrade torch` and restart, "
            "or set EMBEDDING_BACKEND=hashing in .env to silence this.",
            e,
        )
        return HashingEmbedder()
