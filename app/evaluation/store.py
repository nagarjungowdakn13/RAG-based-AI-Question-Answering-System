"""SQLite persistence for evaluation runs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.config import settings


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or settings.eval_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            experiment_name TEXT,
            qa_path TEXT NOT NULL,
            config_json TEXT NOT NULL,
            aggregate_json TEXT NOT NULL,
            results_path TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            question TEXT NOT NULL,
            reference TEXT,
            prediction TEXT,
            abstained INTEGER NOT NULL,
            confidence_score REAL NOT NULL,
            exact_match REAL NOT NULL,
            semantic_similarity REAL NOT NULL,
            retrieval_accuracy REAL NOT NULL,
            row_json TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES eval_runs(run_id)
        )
        """
    )
    conn.commit()


def save_eval_run(
    *,
    run_id: str,
    created_at: str,
    experiment_name: str | None,
    qa_path: str,
    config: dict[str, Any],
    aggregate: dict[str, float],
    rows: list[dict],
    results_path: str,
    db_path: Path | None = None,
) -> None:
    with _connect(db_path) as conn:
        init_db(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_runs
            (run_id, created_at, experiment_name, qa_path, config_json, aggregate_json, results_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                experiment_name,
                qa_path,
                json.dumps(config, ensure_ascii=False),
                json.dumps(aggregate, ensure_ascii=False),
                results_path,
            ),
        )
        conn.executemany(
            """
            INSERT INTO eval_items
            (run_id, question, reference, prediction, abstained, confidence_score,
             exact_match, semantic_similarity, retrieval_accuracy, row_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    row["question"],
                    row.get("reference", ""),
                    row.get("prediction", ""),
                    int(bool(row.get("abstained"))),
                    float(row.get("confidence_score", 0.0)),
                    float(row.get("exact_match", 0.0)),
                    float(row.get("semantic_similarity", 0.0)),
                    float(row.get("retrieval_accuracy", 0.0)),
                    json.dumps(row, ensure_ascii=False),
                )
                for row in rows
            ],
        )
        conn.commit()
