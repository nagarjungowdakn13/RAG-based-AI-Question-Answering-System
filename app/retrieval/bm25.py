"""Lightweight BM25 index, kept in lockstep with the FAISS store.

Pure-Python (Okapi BM25), no extra dependencies. We rebuild from the
vector store's chunk metadata so there's only ever one source of truth
for what is indexed.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class BM25Hit:
    row: int
    score: float


class BM25Index:
    """Okapi BM25 over a fixed corpus snapshot."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._doc_lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._postings: dict[str, dict[int, int]] = {}
        self._avgdl: float = 0.0
        self._n_docs: int = 0

    def build(self, texts: list[str]) -> None:
        self._docs = [tokenize(t) for t in texts]
        self._doc_lens = [len(d) for d in self._docs]
        self._n_docs = len(self._docs)
        self._avgdl = (sum(self._doc_lens) / self._n_docs) if self._n_docs else 0.0
        self._df.clear()
        self._postings = {}
        for i, toks in enumerate(self._docs):
            seen: set[str] = set()
            tf = Counter(toks)
            for term, count in tf.items():
                self._postings.setdefault(term, {})[i] = count
                if term not in seen:
                    self._df[term] += 1
                    seen.add(term)

    def search(self, query: str, top_k: int) -> list[BM25Hit]:
        if not self._n_docs or top_k <= 0:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []
        scores: dict[int, float] = {}
        for term in set(q_terms):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = self._df[term]
            # Robertson-Sparck-Jones IDF, lower-bounded at 0 to avoid negatives.
            idf = math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings.items():
                dl = self._doc_lens[doc_id] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self._avgdl or 1.0))
                contrib = idf * (tf * (self.k1 + 1.0)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + contrib
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [BM25Hit(row=i, score=float(s)) for i, s in ranked]

    @property
    def size(self) -> int:
        return self._n_docs
