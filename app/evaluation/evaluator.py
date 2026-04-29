"""Run a QA dataset through the pipeline and persist scored results.

QA file format (`data/eval/qa_pairs.json`):
    [
      {"question": "...", "answer": "...", "source": "filename.txt"},
      ...
    ]

Output: `storage/eval_results/eval-<UTC-timestamp>.json` containing
per-question rows + aggregate metrics. Persisted JSON is the source of
truth for tracking eval drift across model / chunk-size changes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.evaluation import metrics
from app.logger import get_logger
from app.pipeline.rag import RAGPipeline
from app.pipeline.generator import UNKNOWN_ANSWER

logger = get_logger(__name__)


def _load_qa(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"QA file must be a JSON array of objects, got {type(data).__name__}")
    return data


def evaluate(
    qa_path: str | Path = "data/eval/qa_pairs.json",
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> dict:
    rag = RAGPipeline.instance()
    qa_path = Path(qa_path)
    qa = _load_qa(qa_path)

    rows: list[dict] = []
    for item in qa:
        question = item["question"]
        reference = item.get("answer", "")
        expected_source = item.get("source")

        result = rag.query(question, top_k=top_k, score_threshold=score_threshold)
        prediction = result["answer"]
        abstained = (prediction.strip() == UNKNOWN_ANSWER)

        # Embed prediction & reference once each for the semantic-sim metric.
        # Empty strings are guarded so we don't pass [""] to the model.
        pred_text = "" if abstained else prediction
        if pred_text and reference:
            vecs = rag.embedder.embed([pred_text, reference])
            sem = metrics.semantic_similarity(vecs[0], vecs[1])
        else:
            sem = 0.0

        retrieved_sources = [s["source"] for s in result["sources"]]
        retrieved_texts = [s["snippet"] for s in result["sources"]]
        ret_hit = metrics.retrieval_hit(
            retrieved_sources, expected_source, reference, retrieved_texts
        )

        rows.append(
            {
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "abstained": abstained,
                "confidence_score": result["confidence_score"],
                "exact_match": metrics.exact_match(prediction, reference),
                "semantic_similarity": sem,
                "retrieval_accuracy": ret_hit,
                "retrieved_sources": retrieved_sources,
            }
        )

    agg = metrics.aggregate(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = settings.eval_dir / f"eval-{ts}.json"
    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"qa_path": str(qa_path), "aggregate": agg, "results": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Eval finished — %s — aggregate=%s", out_path, agg)
    return {"num_questions": len(rows), "aggregate": agg, "results_path": str(out_path)}
