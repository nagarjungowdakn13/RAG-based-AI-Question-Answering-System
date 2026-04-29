# Sample Queries & Outputs

These are representative interactions with the system using the bundled
example dataset (`data/docs/`). Outputs are abbreviated for readability.

---

### Q1 — In-domain factual

```bash
python run.py query "Who coined the term machine learning?"
```

```json
{
  "question": "Who coined the term machine learning?",
  "answer": "Arthur Samuel coined the term machine learning in 1959, defining it as the field of study that gives computers the ability to learn without being explicitly programmed. [#1]",
  "confident": true,
  "confidence_score": 0.71,
  "sources": [
    {
      "rank": 1,
      "score": 0.71,
      "source": "data/docs/machine_learning.txt",
      "chunk_id": "…",
      "snippet": "Machine learning is a subfield of artificial intelligence … The term was coined by Arthur Samuel in 1959 …"
    }
  ]
}
```

---

### Q2 — In-domain reasoning

```bash
python run.py query "Why do Transformers need positional encodings?"
```

```json
{
  "answer": "Self-attention is permutation-invariant, so positional encodings are needed to inject information about token order. The original Transformer used fixed sinusoidal encodings; later variants use learned or relative encodings such as RoPE. [#1]",
  "confident": true,
  "confidence_score": 0.68,
  "sources": [{ "source": "data/docs/transformers.txt", "...": "..." }]
}
```

---

### Q3 — Multi-document synthesis

```bash
python run.py query "How does RAG reduce hallucination?"
```

```json
{
  "answer": "RAG reduces hallucination via three layered controls: a strict system prompt that forbids out-of-context answers, a retrieval confidence threshold that triggers abstention, and source attribution that lets users verify grounding. [#1]",
  "confident": true,
  "confidence_score": 0.74
}
```

---

### Q4 — Out-of-domain (hallucination guard)

```bash
python run.py query "What is the capital of France?"
```

```json
{
  "answer": "I don't know based on the provided context.",
  "confident": false,
  "confidence_score": 0.18,
  "sources": []
}
```

The retrieval guard fires before the LLM is even invoked.

---

### Evaluation

```bash
python run.py evaluate
```

```json
{
  "num_questions": 10,
  "aggregate": {
    "exact_match": 0.10,
    "semantic_similarity": 0.78,
    "retrieval_accuracy": 0.90,
    "answered": 0.90
  },
  "results_path": "storage/eval_results/eval-20260429T101530Z.json"
}
```

Exact match is intentionally low — the references are full sentences and
the generator paraphrases. Semantic similarity (cosine on
sentence-transformer embeddings) is the better quality signal here.
Retrieval accuracy isolates the retrieval component, so a regression in
that number points clearly at the chunker / embedder / index, not the
prompt.
