"""FastAPI application surface.

All endpoints are async. CPU-bound operations (embedding, FAISS) are
dispatched to a worker thread via `asyncio.to_thread` so concurrent
requests don't block the event loop.
"""
from __future__ import annotations

import asyncio
import hmac
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import json

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.evaluation.evaluator import evaluate
from app.logger import get_logger
from app.pipeline.rag import RAGPipeline
from app.rate_limit import TokenBucketLimiter
from app.schemas import (
    DeleteDocumentsResponse,
    EvaluateRequest,
    EvaluateResponse,
    IngestPathsRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.stats import compute_stats

logger = get_logger("api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
PROTECTED_PATHS = {
    ("POST", "/ingest"),
    ("POST", "/ingest/upload"),
    ("POST", "/query"),
    ("POST", "/query/stream"),
    ("POST", "/evaluate"),
    ("DELETE", "/documents"),
}

RATE_LIMITED_PATHS = {
    "/query", "/query/stream", "/ingest", "/ingest/upload", "/evaluate",
}

rate_limiter = TokenBucketLimiter(
    rate_per_minute=settings.rate_limit_per_minute,
    burst=settings.rate_limit_burst,
)


def _resolve_ingest_path(p: Path) -> Path | None:
    """Try the path as-is, then relative to the project root.

    Defends against the case where the server is started from a different
    working directory than the project root.
    """
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──────────────────────────────────────────────────────
    rag = await asyncio.to_thread(RAGPipeline.instance)

    if settings.auto_ingest_on_startup and rag.store.size == 0:
        logger.info(
            "Auto-ingest enabled. CWD=%s, project_root=%s, configured path=%s",
            Path.cwd(), PROJECT_ROOT, settings.auto_ingest_path,
        )
        resolved = _resolve_ingest_path(settings.auto_ingest_path)
        if resolved is None:
            logger.warning(
                "AUTO-INGEST SKIPPED: path '%s' not found relative to CWD or "
                "project root. Use the UI's '🚀 Initialize' button or the "
                "Ingest panel manually.",
                settings.auto_ingest_path,
            )
        else:
            try:
                logger.info("Auto-ingesting from resolved path: %s", resolved)
                result = await asyncio.to_thread(rag.ingest, [resolved])
                logger.info(
                    "AUTO-INGEST OK: %d document(s), %d chunk(s) added.",
                    result.get("ingested_documents", 0),
                    result.get("total_chunks", 0),
                )
                if result.get("total_chunks", 0) == 0:
                    logger.warning(
                        "AUTO-INGEST produced 0 chunks — directory exists but "
                        "contains no .txt/.md/.pdf files."
                    )
            except Exception as e:
                logger.exception("AUTO-INGEST FAILED: %s", e)

    logger.info(
        "RAG pipeline ready (embedder=%s, llm=%s, index_size=%d)",
        settings.embedding_backend,
        settings.llm_backend,
        rag.store.size,
    )
    yield
    # ── shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down RAG service")


app = FastAPI(
    title="RAG-Based AI QA System",
    version="1.0.0",
    description=(
        "Retrieval-Augmented Generation API. POST /ingest to add documents, "
        "POST /query to ask questions, POST /evaluate to score the system "
        "against a QA file."
    ),
    lifespan=lifespan,
)

# Permissive CORS so the bundled frontend works whether it's served from
# the same origin (default) or hosted separately during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Stamp every request with an ID + log latency. Cheap structured tracing."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    if settings.api_key and (request.method, request.url.path) in PROTECTED_PATHS:
        supplied = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(supplied, settings.api_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid X-API-Key header."},
                headers={"x-request-id": request_id},
            )
    if rate_limiter.enabled and request.url.path in RATE_LIMITED_PATHS:
        client_ip = (request.client.host if request.client else "") or "unknown"
        allowed, retry_after = rate_limiter.allow(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded.",
                    "retry_after_seconds": round(retry_after, 2),
                },
                headers={
                    "x-request-id": request_id,
                    "retry-after": str(max(1, int(retry_after))),
                },
            )
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request failed rid=%s method=%s path=%s elapsed_ms=%.1f",
            request_id, request.method, request.url.path, elapsed_ms,
        )
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = request_id
    # Skip noisy access logs for static assets and the polling /stats endpoint.
    if not (request.url.path.startswith("/static") or request.url.path == "/stats"):
        logger.info(
            "rid=%s %s %s -> %d in %.1fms",
            request_id, request.method, request.url.path,
            response.status_code, elapsed_ms,
        )
    return response


# ─── frontend ───────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")


START_TIME = time.time()


@app.get("/health")
async def health() -> dict:
    rag = RAGPipeline.instance()
    try:
        usage = shutil.disk_usage(str(settings.index_dir.parent.resolve()))
        disk_free_mb = round(usage.free / (1024 * 1024), 1)
    except OSError:
        disk_free_mb = None
    return {
        "status": "ok",
        "version": app.version,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "now": datetime.now(timezone.utc).isoformat(),
        "index_size": rag.store.size,
        "index_version": rag.store.version,
        "documents": len({
            m.get("metadata", {}).get("source") for m in rag.store._meta
            if m.get("metadata", {}).get("source")
        }),
        "embedding_backend": settings.embedding_backend,
        "embedding_dim": rag.embedder.dim,
        "llm_backend": settings.llm_backend,
        "retrieval_mode": settings.retrieval_mode,
        "mmr_enabled": settings.mmr_enabled,
        "reranker": {
            "enabled": settings.reranker_enabled,
            "available": getattr(rag.reranker, "available", False),
            "model": settings.reranker_model if settings.reranker_enabled else None,
        },
        "cache": rag.cache.stats(),
        "disk_free_mb": disk_free_mb,
    }


@app.get("/ready")
async def ready() -> JSONResponse:
    """Strict readiness probe: 200 only when the index has documents."""
    rag = RAGPipeline.instance()
    if rag.store.size == 0:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "empty_index"},
        )
    return JSONResponse(content={"status": "ready", "index_size": rag.store.size})


@app.get("/stats")
async def stats() -> dict:
    """Aggregate metrics for the dashboard (index, queries, confidence, sources)."""
    return await asyncio.to_thread(compute_stats)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_paths(req: IngestPathsRequest) -> IngestResponse:
    """Ingest documents by path. Paths can be files or directories."""
    rag = RAGPipeline.instance()
    # Resolve paths relative to the project root if needed so the API works
    # regardless of where the server was started from.
    resolved: list[str] = []
    for raw in req.paths:
        r = _resolve_ingest_path(Path(raw))
        if r is None:
            raise HTTPException(status_code=404, detail=f"Path not found: {raw}")
        resolved.append(str(r))
    try:
        result = await asyncio.to_thread(
            rag.ingest, resolved, req.chunk_size, req.chunk_overlap
        )
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))
    return IngestResponse(index_size=rag.store.size, **result)


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(files: list[UploadFile] = File(...)) -> IngestResponse:
    """Ingest one or more uploaded files (.txt/.md/.pdf)."""
    rag = RAGPipeline.instance()
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    try:
        saved: list[str] = []
        for f in files:
            # Strip any directory components from client-supplied filenames
            # so an upload can never write outside tmp_dir.
            safe_name = Path(f.filename or "upload.bin").name or "upload.bin"
            dest = tmp_dir / safe_name
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(str(dest))
        result = await asyncio.to_thread(rag.ingest, saved)
        return IngestResponse(index_size=rag.store.size, **result)
    except Exception as e:
        logger.exception("Upload ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.delete("/documents", response_model=DeleteDocumentsResponse)
async def delete_documents(source: str) -> DeleteDocumentsResponse:
    """Remove all indexed chunks for one source path."""
    rag = RAGPipeline.instance()
    candidates = [source]
    resolved = _resolve_ingest_path(Path(source))
    if resolved is not None and str(resolved) != source:
        candidates.append(str(resolved))

    for candidate in candidates:
        result = await asyncio.to_thread(rag.delete_source, candidate)
        if result["deleted_chunks"] > 0:
            return DeleteDocumentsResponse(**result)

    raise HTTPException(status_code=404, detail=f"Source not found: {source}")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    rag = RAGPipeline.instance()
    if rag.store.size == 0:
        raise HTTPException(
            status_code=409,
            detail="Index is empty. POST /ingest documents before querying.",
        )
    try:
        result = await asyncio.to_thread(
            rag.query,
            req.question, req.top_k, req.score_threshold, req.prompt_strategy,
            req.source_filter,
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(**result)


@app.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """Server-Sent Events version of /query.

    Emits one `data:` line per event with these `type` values:
      - `sources`: initial source-chunk array (so the UI can render before tokens arrive)
      - `token`:   incremental answer deltas
      - `done`:    final aggregate (answer, confidence, rejection status)

    Bridges a blocking iterator from the RAG pipeline into the async event
    loop via a thread pool — keeping the loop responsive while the LLM streams.
    """
    rag = RAGPipeline.instance()
    if rag.store.size == 0:
        raise HTTPException(
            status_code=409,
            detail="Index is empty. POST /ingest documents before querying.",
        )

    async def event_stream():
        loop = asyncio.get_running_loop()
        it = rag.stream_query(
            req.question, req.top_k, req.score_threshold,
            req.prompt_strategy, req.source_filter,
        )
        sentinel = object()
        try:
            while True:
                event = await loop.run_in_executor(None, lambda: next(it, sentinel))
                if event is sentinel:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("Streaming query failed")
            err = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def run_evaluation(req: EvaluateRequest) -> EvaluateResponse:
    qa_path = _resolve_ingest_path(Path(req.qa_path))
    if qa_path is None:
        raise HTTPException(status_code=404, detail=f"QA file not found: {req.qa_path}")
    try:
        result = await asyncio.to_thread(
            evaluate,
            str(qa_path),
            req.top_k,
            req.score_threshold,
            req.experiment_name,
            req.chunk_size,
            req.prompt_strategy,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))
    return EvaluateResponse(**result)


@app.exception_handler(ValueError)
async def _value_error_handler(_request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
