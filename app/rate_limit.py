"""In-memory token-bucket rate limiter, per client IP.

Zero external deps. Suitable for single-process deployments — for multi-
worker uvicorn you'd want a shared store (Redis), but at that point a
proper gateway should be doing this anyway. We just want to keep a demo
endpoint from being trivially scraped.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketLimiter:
    def __init__(self, rate_per_minute: int, burst: int, max_clients: int = 4096):
        self.rate_per_second = rate_per_minute / 60.0 if rate_per_minute > 0 else 0.0
        self.burst = max(1, burst)
        self.max_clients = max_clients
        self._buckets: "OrderedDict[str, _Bucket]" = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate_per_second > 0.0

    def allow(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        if not self.enabled:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.burst), last_refill=now)
                self._buckets[key] = bucket
            else:
                self._buckets.move_to_end(key)
                elapsed = now - bucket.last_refill
                bucket.tokens = min(
                    self.burst, bucket.tokens + elapsed * self.rate_per_second
                )
                bucket.last_refill = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                allowed = True
                retry = 0.0
            else:
                allowed = False
                retry = (1.0 - bucket.tokens) / self.rate_per_second
            while len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
        return allowed, retry
