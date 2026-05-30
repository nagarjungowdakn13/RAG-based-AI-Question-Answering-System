"""Dashboard statistics aggregator.

Reads `storage/logs/queries.jsonl` and the live FAISS index to produce
summary metrics consumed by the web dashboard's `GET /stats` endpoint.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import settings
from app.pipeline.generator import UNKNOWN_ANSWER
from app.pipeline.rag import RAGPipeline


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _confidence_distribution(confidences: list[float]) -> list[dict]:
    bins = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    counts = [0] * 5
    for c in confidences:
        idx = max(0, min(int(c * 5), 4))
        counts[idx] += 1
    return [{"bin": bins[i], "count": counts[i]} for i in range(5)]


def _hourly_timeline(rows: list[dict], hours: int = 24) -> list[dict]:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets: dict[str, int] = {}
    for h in range(hours - 1, -1, -1):
        key = (now - timedelta(hours=h)).isoformat()
        buckets[key] = 0

    for r in rows:
        ts = r.get("ts")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        bucket = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()
        if bucket in buckets:
            buckets[bucket] += 1
    return [{"hour": k, "count": v} for k, v in buckets.items()]


def _top_sources(rows: list[dict], limit: int = 5) -> list[dict]:
    counter: Counter[str] = Counter()
    for r in rows:
        for s in r.get("sources", []) or []:
            src = s.get("source") or ""
            if src:
                counter[Path(src).name] += 1
    return [{"source": s, "count": c} for s, c in counter.most_common(limit)]


def _recent(rows: list[dict], limit: int = 10) -> list[dict]:
    out = []
    for r in reversed(rows[-limit:]):
        ans = r.get("answer", "") or ""
        out.append(
            {
                "ts": r.get("ts"),
                "question": r.get("question", ""),
                "answer": ans[:200] + ("…" if len(ans) > 200 else ""),
                "confident": bool(r.get("confident")),
                "confidence_score": float(r.get("confidence_score", 0.0) or 0.0),
            }
        )
    return out


def compute_stats() -> dict:
    rag = RAGPipeline.instance()

    # Distinct source documents currently in the index.
    sources = {
        m.get("metadata", {}).get("source")
        for m in rag.store._meta
        if m.get("metadata", {}).get("source")
    }
    indexed_sources = sorted({Path(s).name for s in sources if s})

    rows = _load_jsonl(settings.log_dir / "queries.jsonl")
    confidences = [
        float(r.get("confidence_score", 0.0) or 0.0)
        for r in rows
        if r.get("confidence_score") is not None
    ]
    answered = [r for r in rows if r.get("answer", "").strip() != UNKNOWN_ANSWER]
    abstained_count = len(rows) - len(answered)
    answered_confidences = [
        float(r.get("confidence_score", 0.0) or 0.0)
        for r in answered
        if r.get("confidence_score") is not None
    ]
    avg_conf = sum(answered_confidences) / len(answered_confidences) if answered_confidences else 0.0

    return {
        "index_size": rag.store.size,
        "index_version": rag.store.version,
        "documents": len(sources),
        "indexed_sources": indexed_sources,
        "embedding_backend": settings.embedding_backend,
        "llm_backend": settings.llm_backend,
        "retrieval_mode": settings.retrieval_mode,
        "mmr_enabled": settings.mmr_enabled,
        "reranker_enabled": settings.reranker_enabled,
        "cache": rag.cache.stats(),
        "queries": {
            "total": len(rows),
            "answered": len(answered),
            "abstained": abstained_count,
            "avg_confidence": round(avg_conf, 4),
        },
        "confidence_distribution": _confidence_distribution(confidences),
        "timeline": _hourly_timeline(rows, hours=24),
        "top_sources": _top_sources(rows, limit=5),
        "recent_queries": _recent(rows, limit=10),
    }
