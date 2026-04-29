"""Batch evaluation with JSON reports and SQLite persistence."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.evaluation import metrics
from app.evaluation.store import save_eval_run
from app.logger import get_logger
from app.pipeline.generator import UNKNOWN_ANSWER
from app.pipeline.rag import RAGPipeline

logger = get_logger(__name__)


def _load_qa(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"QA file must be a JSON array of objects, got {type(data).__name__}")
    return data


def _semantic_score(rag: RAGPipeline, prediction: str, reference: str) -> float:
    if not prediction or not reference or prediction.strip() == UNKNOWN_ANSWER:
        return 0.0
    vecs = rag.embedder.embed([prediction, reference])
    return metrics.semantic_similarity(vecs[0], vecs[1])


def evaluate(
    qa_path: str | Path = "data/eval/qa_pairs.json",
    top_k: int | None = None,
    score_threshold: float | None = None,
    experiment_name: str | None = None,
    chunk_size: int | None = None,
    prompt_strategy: str | None = None,
) -> dict:
    """Run a QA file through the pipeline and persist a research report."""
    rag = RAGPipeline.instance()
    qa_path = Path(qa_path)
    qa = _load_qa(qa_path)
    created_at = datetime.now(timezone.utc)
    run_id = f"eval-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    config = {
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.hf_embedding_model
        if settings.embedding_backend == "huggingface"
        else settings.openai_embedding_model,
        "llm_backend": settings.llm_backend,
        "prompt_strategy": prompt_strategy or settings.prompt_strategy,
        "chunk_size": chunk_size or settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": top_k or settings.top_k,
        "score_threshold": score_threshold
        if score_threshold is not None
        else settings.score_threshold,
        "confidence_threshold": settings.confidence_threshold,
        "grounding_min_coverage": settings.grounding_min_coverage,
    }

    rows: list[dict] = []
    for item in qa:
        question = item["question"]
        reference = item.get("answer", "")
        expected_source = item.get("source")

        result = rag.query(
            question,
            top_k=top_k,
            score_threshold=score_threshold,
            prompt_strategy=prompt_strategy,
        )
        prediction = result["answer"]
        abstained = prediction.strip() == UNKNOWN_ANSWER
        retrieved_sources = [s["source"] for s in result["sources"]]
        retrieved_texts = [s["snippet"] for s in result["sources"]]
        row = {
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "abstained": abstained,
            "confidence_score": result["confidence_score"],
            "confidence_explanation": result.get("confidence_explanation"),
            "rejected": result.get("rejected", False),
            "rejection_reason": result.get("rejection_reason"),
            "exact_match": metrics.exact_match(prediction, reference),
            "semantic_similarity": _semantic_score(rag, prediction, reference),
            "retrieval_accuracy": metrics.retrieval_hit(
                retrieved_sources, expected_source, reference, retrieved_texts
            ),
            "retrieved_sources": retrieved_sources,
            "retrieved_scores": [s["score"] for s in result["sources"]],
        }
        rows.append(row)

    agg = metrics.aggregate(rows)
    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.eval_dir / f"{run_id}.json"
    report = {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "experiment_name": experiment_name,
        "qa_path": str(qa_path),
        "config": config,
        "aggregate": agg,
        "results": rows,
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    save_eval_run(
        run_id=run_id,
        created_at=created_at.isoformat(),
        experiment_name=experiment_name,
        qa_path=str(qa_path),
        config=config,
        aggregate=agg,
        rows=rows,
        results_path=str(out_path),
    )
    logger.info("Eval finished - %s - aggregate=%s", out_path, agg)
    return {
        "num_questions": len(rows),
        "aggregate": agg,
        "results_path": str(out_path),
        "run_id": run_id,
        "db_path": str(settings.eval_db_path),
    }
