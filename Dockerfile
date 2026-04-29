# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────
# RAG-Based AI QA System — production image.
# Two-stage build keeps the runtime image lean (no build toolchain).
# ─────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


# ─── runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    AUTO_INGEST_ON_STARTUP=false

# Non-root user — never run web services as root in production.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app
COPY --from=builder /install /usr/local

# Application code (NOT data/ or storage/ — those mount in at runtime)
COPY app ./app
COPY frontend ./frontend
COPY run.py ./run.py

# Storage dirs are written at runtime; mount as volumes in compose/k8s.
RUN mkdir -p storage/faiss_index storage/logs storage/eval_results \
 && chown -R app:app /app

USER app
EXPOSE 8000

# Container-level healthcheck mirrors the orchestrator's readiness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ.get('PORT','8000')+'/health', timeout=3).status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
