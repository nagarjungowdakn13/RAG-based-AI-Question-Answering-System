"""CLI entry point.

Examples:
    python run.py ingest data/docs
    python run.py query "What is a transformer?"
    python run.py evaluate
    python run.py serve --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_ingest(args) -> int:
    from app.pipeline.rag import RAGPipeline

    rag = RAGPipeline.instance()
    result = rag.ingest(args.paths, args.chunk_size, args.chunk_overlap)
    print(json.dumps({"index_size": rag.store.size, **result}, indent=2))
    return 0


def _cmd_query(args) -> int:
    from app.pipeline.rag import RAGPipeline

    rag = RAGPipeline.instance()
    if rag.store.size == 0:
        print("Index is empty. Run `ingest` first.", file=sys.stderr)
        return 2
    result = rag.query(args.question, top_k=args.top_k, score_threshold=args.threshold)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_evaluate(args) -> int:
    from app.evaluation.evaluator import evaluate

    out = evaluate(args.qa_path, top_k=args.top_k, score_threshold=args.threshold)
    print(json.dumps(out, indent=2))
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="rag", description="RAG-Based AI QA System CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Ingest documents into the FAISS index")
    pi.add_argument("paths", nargs="+", help="Files or directories to ingest")
    pi.add_argument("--chunk-size", type=int, default=None)
    pi.add_argument("--chunk-overlap", type=int, default=None)
    pi.set_defaults(func=_cmd_ingest)

    pq = sub.add_parser("query", help="Ask the RAG system a question")
    pq.add_argument("question")
    pq.add_argument("--top-k", type=int, default=None)
    pq.add_argument("--threshold", type=float, default=None)
    pq.set_defaults(func=_cmd_query)

    pe = sub.add_parser("evaluate", help="Run evaluation against a QA file")
    pe.add_argument("--qa-path", default="data/eval/qa_pairs.json")
    pe.add_argument("--top-k", type=int, default=None)
    pe.add_argument("--threshold", type=float, default=None)
    pe.set_defaults(func=_cmd_evaluate)

    ps = sub.add_parser("serve", help="Run the FastAPI server")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--reload", action="store_true")
    ps.set_defaults(func=_cmd_serve)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
