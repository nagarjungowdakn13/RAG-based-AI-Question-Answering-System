"""FAISS vector store with on-disk persistence.

We use `IndexFlatIP` over L2-normalized embeddings: that's exact cosine
similarity, no approximation, no training step. It's the right default
up to ~1M chunks; swap for IVF/HNSW only when you've measured a need.

Persistence is two files in `index_dir`:
  - index.faiss : the FAISS binary
  - meta.pkl    : list[dict] aligned to row order, holding chunk text + metadata
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.config import settings
from app.logger import get_logger
from app.pipeline.chunking import Chunk

logger = get_logger(__name__)


@dataclass
class RetrievalHit:
    score: float
    chunk_id: str
    text: str
    metadata: dict


class FaissVectorStore:
    INDEX_FILE = "index.faiss"
    META_FILE = "meta.pkl"

    def __init__(self, dim: int, index_dir: Path | None = None):
        self.dim = dim
        self.index_dir: Path = Path(index_dir or settings.index_dir)
        self.index: faiss.Index = faiss.IndexFlatIP(dim)
        self._meta: list[dict] = []  # row i ↔ self._meta[i]
        self._known_ids: set[str] = set()

    # ─── core ops ────────────────────────────────────────────────────
    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> int:
        """Add vectors+chunks, deduping by chunk_id. Returns count actually added."""
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks length mismatch")
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")

        # Dedupe: skip any chunk whose id is already in the index. Chunk IDs
        # are deterministic (uuid5 of source+index), so re-ingesting the same
        # file is a no-op rather than a silent duplicate-and-bloat.
        keep_rows: list[int] = []
        for i, c in enumerate(chunks):
            if c.chunk_id in self._known_ids:
                continue
            keep_rows.append(i)
            self._known_ids.add(c.chunk_id)

        if not keep_rows:
            logger.info("Ingest dedupe: 0 new chunks (all %d already indexed)", len(chunks))
            return 0

        kept_vectors = vectors[keep_rows]
        self.index.add(kept_vectors)
        for i in keep_rows:
            c = chunks[i]
            self._meta.append(
                {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
            )
        skipped = len(chunks) - len(keep_rows)
        if skipped:
            logger.info(
                "Ingest dedupe: added %d, skipped %d duplicate(s); index now %d",
                len(keep_rows), skipped, self.index.ntotal,
            )
        else:
            logger.info("Index now contains %d vectors", self.index.ntotal)
        return len(keep_rows)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[RetrievalHit]:
        if self.index.ntotal == 0:
            return []
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")
        scores, idxs = self.index.search(query_vec, min(top_k, self.index.ntotal))
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            row = self._meta[idx]
            hits.append(
                RetrievalHit(
                    score=float(score),
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    metadata=row["metadata"],
                )
            )
        return hits

    def delete_by_source(self, source: str) -> int:
        """Delete chunks whose metadata source exactly matches `source`."""
        keep_idxs: list[int] = []
        delete_count = 0
        for i, row in enumerate(self._meta):
            if row.get("metadata", {}).get("source") == source:
                delete_count += 1
            else:
                keep_idxs.append(i)

        if delete_count == 0:
            return 0

        vectors = [self.index.reconstruct(i) for i in keep_idxs]
        self.index = faiss.IndexFlatIP(self.dim)
        if vectors:
            self.index.add(np.asarray(vectors, dtype="float32"))
        self._meta = [self._meta[i] for i in keep_idxs]
        self._known_ids = {m["chunk_id"] for m in self._meta if m.get("chunk_id")}
        logger.info("Deleted %d chunk(s) for source=%s", delete_count, source)
        return delete_count

    # ─── persistence ─────────────────────────────────────────────────
    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / self.INDEX_FILE))
        with (self.index_dir / self.META_FILE).open("wb") as f:
            pickle.dump({"dim": self.dim, "meta": self._meta}, f)
        logger.info("Saved FAISS index → %s", self.index_dir)

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "FaissVectorStore | None":
        d = Path(index_dir or settings.index_dir)
        idx_path = d / cls.INDEX_FILE
        meta_path = d / cls.META_FILE
        if not idx_path.exists() or not meta_path.exists():
            return None
        with meta_path.open("rb") as f:
            payload = pickle.load(f)
        store = cls(dim=payload["dim"], index_dir=d)
        store.index = faiss.read_index(str(idx_path))
        store._meta = payload["meta"]
        store._known_ids = {m["chunk_id"] for m in store._meta if m.get("chunk_id")}
        logger.info("Loaded FAISS index from %s (%d vectors)", d, store.index.ntotal)
        return store

    @property
    def size(self) -> int:
        return int(self.index.ntotal)
