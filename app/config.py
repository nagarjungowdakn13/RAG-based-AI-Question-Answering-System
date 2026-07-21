"""Centralized configuration via pydantic-settings.

All knobs are env-driven so the same image runs in dev / staging / prod
without code changes. See `.env.example` for the full surface.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Embedding backend
    embedding_backend: Literal["huggingface", "openai", "hashing"] = "huggingface"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # Generation backend
    llm_backend: Literal["openai", "extractive"] = "extractive"
    openai_chat_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    # Chunking
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)
    chunk_unit: Literal["chars", "tokens"] = "chars"
    tiktoken_encoding: str = "cl100k_base"

    # Retrieval
    top_k: int = Field(default=4, ge=1, le=50)
    score_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    grounding_min_coverage: float = Field(default=0.35, ge=0.0, le=1.0)
    prompt_strategy: Literal["strict", "fallback"] = "strict"

    # Hybrid retrieval (BM25 + dense) + reciprocal rank fusion
    retrieval_mode: Literal["dense", "hybrid", "bm25"] = "hybrid"
    hybrid_candidate_multiplier: int = Field(default=3, ge=1, le=10)
    rrf_k: int = Field(default=60, ge=1, le=1000)

    # MMR diversification on the fused candidate pool
    mmr_enabled: bool = True
    mmr_lambda: float = Field(default=0.6, ge=0.0, le=1.0)

    # Optional cross-encoder reranker
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_candidate_multiplier: int = Field(default=4, ge=1, le=10)

    # Query result cache (LRU). Keyed by question + params + index version.
    query_cache_enabled: bool = True
    query_cache_size: int = Field(default=256, ge=0, le=10000)

    # PDF extraction: prefer pymupdf when installed, fall back to pypdf.
    pdf_backend: Literal["auto", "pymupdf", "pypdf"] = "auto"

    # Rate limit (token bucket, per client IP). 0 disables.
    rate_limit_per_minute: int = Field(default=0, ge=0, le=10000)
    rate_limit_burst: int = Field(default=20, ge=1, le=10000)

    # Storage
    index_dir: Path = Path("storage/faiss_index")
    log_dir: Path = Path("storage/logs")
    eval_dir: Path = Path("storage/eval_results")
    eval_db_path: Path = Path("storage/eval_results/evaluations.sqlite3")

    # Optional API protection. Leave blank for local/demo use.
    api_key: str = ""

    # Logging
    log_level: str = "INFO"

    # CORS — comma-separated list of allowed origins. "*" means any.
    # Tighten this in production (e.g. "https://your-app.example.com").
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Demo defaults: auto-load the bundled docs on first startup so the
    # web UI is interactive out of the box. Set false in production.
    auto_ingest_on_startup: bool = True
    auto_ingest_path: Path = Path("data/docs")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        level = v.upper()
        if level not in logging._nameToLevel:
            raise ValueError(f"Invalid LOG_LEVEL: {v}")
        return level

    def ensure_dirs(self) -> None:
        for p in (self.index_dir, self.log_dir, self.eval_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
