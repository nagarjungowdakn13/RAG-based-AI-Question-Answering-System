"""Structured logging.

- A standard rotating logger for human-readable lines (`storage/logs/app.log`).
- A JSONL sink for machine-readable Q/A traces (`storage/logs/queries.jsonl`)
  used downstream by analytics and the eval framework.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from app.config import settings

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level))
    formatter = logging.Formatter(_FMT)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    log_path = settings.log_dir / "app.log"
    try:
        file_handler = RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
    except OSError as e:
        logger.warning("File logging disabled for %s: %s", log_path, e)
    else:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def log_query_event(payload: dict[str, Any]) -> None:
    """Append a single Q/A trace as one JSON line.

    JSONL is grep-friendly, easy to load with pandas, and append-safe under
    concurrent writers on every OS we target.
    """
    payload = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
    path: Path = settings.log_dir / "queries.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as e:
        get_logger(__name__).warning("Query event logging skipped for %s: %s", path, e)
