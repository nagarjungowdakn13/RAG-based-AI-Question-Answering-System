"""In-memory LRU cache for query results.

Cache key bundles the question plus all knobs that change the answer,
including the vector store's monotonic version — so the cache invalidates
automatically on ingest or delete.
"""
from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheKey:
    question: str
    top_k: int | None
    score_threshold: float | None
    prompt_strategy: str
    sources: tuple[str, ...]
    index_version: int


class QueryCache:
    """Thread-safe LRU. `maxsize=0` disables caching."""

    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self._store: "OrderedDict[CacheKey, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: CacheKey) -> dict | None:
        if self.maxsize <= 0:
            return None
        with self._lock:
            value = self._store.get(key)
            if value is None:
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return copy.deepcopy(value)

    def put(self, key: CacheKey, value: dict) -> None:
        if self.maxsize <= 0:
            return
        with self._lock:
            self._store[key] = copy.deepcopy(value)
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._store),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }
