"""Query-level structured traces."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from app.logger import log_query_event


def _safe(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    return value


def log_query_trace(payload: dict[str, Any]) -> None:
    """Persist a full query trace as one JSONL event."""
    log_query_event(_safe(payload))
