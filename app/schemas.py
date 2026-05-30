"""API request / response models.

Pydantic models double as runtime validators and OpenAPI schema, so the
public contract of the FastAPI app stays in one place.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestPathsRequest(BaseModel):
    paths: list[str] = Field(
        ...,
        description="Absolute or working-dir-relative paths to .txt/.md/.pdf files or directories.",
        min_length=1,
    )
    chunk_size: int | None = Field(default=None, ge=100, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)


class IngestResponse(BaseModel):
    ingested_documents: int
    total_chunks: int
    index_size: int
    files: list[str]


class DeleteDocumentsResponse(BaseModel):
    deleted_chunks: int
    index_size: int
    source: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    prompt_strategy: str | None = Field(default=None, pattern="^(strict|fallback)$")
    source_filter: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of source filenames or paths to restrict retrieval to. "
            "Both basename ('rag_systems.txt') and full path are accepted."
        ),
        max_length=50,
    )


class SourceChunk(BaseModel):
    rank: int
    score: float
    source: str
    chunk_id: str
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    confident: bool
    confidence_score: float
    confidence_explanation: str | None = None
    rejected: bool = False
    rejection_reason: str | None = None
    sources: list[SourceChunk]
    metadata: dict[str, Any]


class EvaluateRequest(BaseModel):
    qa_path: str = Field(
        default="data/eval/qa_pairs.json",
        description="Path to a JSON file: list of {question, answer, source?} objects.",
    )
    top_k: int | None = None
    score_threshold: float | None = None
    experiment_name: str | None = None
    chunk_size: int | None = Field(default=None, ge=100, le=4000)
    prompt_strategy: str | None = Field(default=None, pattern="^(strict|fallback)$")


class EvaluateResponse(BaseModel):
    num_questions: int
    aggregate: dict[str, float]
    results_path: str
    run_id: str | None = None
    db_path: str | None = None
