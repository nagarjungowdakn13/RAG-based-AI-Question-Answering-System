import json
import time
import os
import string
import re
import shutil
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# Force local extractive/sentence-transformers defaults if no API keys are loaded
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["LLM_BACKEND"] = "extractive"
os.environ["EMBEDDING_BACKEND"] = "huggingface"

from app.config import settings
from app.pipeline.rag import RAGPipeline
from app.pipeline.vector_store import FaissVectorStore

# Ensure index and output directories exist
settings.ensure_dirs()
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── METRIC UTILITIES ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def calculate_em(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0

def calculate_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_recall(retrieved_hits, gold_sources, top_k):
    if not gold_sources:
        return 1.0
    # Extract file basenames of retrieved documents
    retrieved_names = []
    for h in retrieved_hits[:top_k]:
        src = h.metadata.get("source", "") or h.metadata.get("filename", "")
        retrieved_names.append(Path(src).name.lower())
    
    hits = sum(1 for src in gold_sources if src.lower() in retrieved_names)
    return hits / len(gold_sources)

def check_faithfulness(question: str, answer: str, contexts: list[str], client) -> float:
    # Local fallback metric
    from app.retrieval.confidence import _content_words
    answer_words = _content_words(answer)
    if not answer_words:
        return 1.0
    context_words = _content_words("\n".join(contexts))
    if not context_words:
        return 0.0
    local_score = len(answer_words & context_words) / len(answer_words)

    # LLM-as-judge if API key is present and configured
    if client and settings.llm_backend == "openai":
        context_str = "\n---\n".join(contexts)
        prompt = f"""You are an expert evaluator. Rate the FAITHFULNESS of the answer to the provided context.
An answer is faithful if it is fully supported by the context and does not contain any information not present in the context, nor does it contradict it.

Context:
{context_str}

Question:
{question}

Answer:
{answer}

Respond with a single number between 0.0 and 1.0, representing the faithfulness of the answer. Do not include any explanation or other text.
Faithfulness:"""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            score_str = resp.choices[0].message.content.strip()
            match = re.search(r"\d+(\.\d+)?", score_str)
            if match:
                return min(1.0, max(0.0, float(match.group(0))))
        except Exception:
            pass
    return local_score

# ─── EXPERIMENT PIPELINE ─────────────────────────────────────────────────────

def reset_index(rag):
    if settings.index_dir.exists():
        shutil.rmtree(settings.index_dir, ignore_errors=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    rag.store = FaissVectorStore(dim=rag.embedder.dim)
    rag.store.save()
    rag.cache.clear()
    rag._bm25_version = -1
    rag._bm25 = None

def run_dataset_eval(rag, qa_pairs, dataset_docs_path, retrieval_mode, chunk_size, top_k, reranker_enabled, client):
    # Set settings
    settings.retrieval_mode = retrieval_mode
    settings.reranker_enabled = reranker_enabled
    settings.chunk_size = chunk_size
    settings.top_k = top_k

    # Reset and Re-Ingest documents
    reset_index(rag)
    rag.ingest([dataset_docs_path], chunk_size=chunk_size)

    results = []
    latencies = []

    for item in qa_pairs:
        question = item["question"]
        ref_answer = item["answer"]
        gold_sources = item.get("sources", [])

        # Measure latency end-to-end
        start_time = time.perf_counter()
        query_out = rag.query(question, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(elapsed_ms)

        pred_answer = query_out["answer"]
        hits = query_out["sources"]  # list of hit dicts
        
        # Build RetrievalHit list for local score calculations
        from app.pipeline.vector_store import RetrievalHit
        hit_objects = [
            RetrievalHit(score=h["score"], chunk_id=h["chunk_id"], text=h["snippet"], metadata={"source": h["source"]})
            for h in hits
        ]

        # Calculate metrics
        em = calculate_em(pred_answer, ref_answer)
        f1 = calculate_f1(pred_answer, ref_answer)
        recall5 = calculate_recall(hit_objects, gold_sources, 5)
        recall10 = calculate_recall(hit_objects, gold_sources, 10)
        
        contexts = [h.text for h in hit_objects]
        faith = check_faithfulness(question, pred_answer, contexts, client)

        # Hallucination checking: Recall@k is 0, but model produced a confident wrong answer
        is_hallucination = False
        recall_at_top = calculate_recall(hit_objects, gold_sources, top_k)
        if recall_at_top == 0.0 and not query_out["rejected"] and em == 0.0:
            is_hallucination = True

        results.append({
            "question": question,
            "prediction": pred_answer,
            "reference": ref_answer,
            "em": em,
            "f1": f1,
            "recall@5": recall5,
            "recall@10": recall10,
            "faithfulness": faith,
            "latency_ms": elapsed_ms,
            "is_hallucination": is_hallucination,
            "confident": query_out["confident"],
            "rejected": query_out["rejected"]
        })

    # Aggregate
    agg_em = np.mean([r["em"] for r in results])
    agg_f1 = np.mean([r["f1"] for r in results])
    agg_rec5 = np.mean([r["recall@5"] for r in results])
    agg_rec10 = np.mean([r["recall@10"] for r in results])
    agg_faith = np.mean([r["faithfulness"] for r in results])
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    
    # Hallucination rate: % of queries with recall=0 that produced a confident wrong answer
    recall_zero_count = sum(1 for r in results if calculate_recall([RetrievalHit(0.0, "", c, {}) for c in r["prediction"]], item.get("sources", []), top_k) == 0.0)
    # Actually, we can count total questions with recall = 0
    total_recall_zero = sum(1 for r in results if r["recall@5"] == 0.0) # recall@5 is 0
    hallucinations_in_zero = sum(1 for r in results if r["recall@5"] == 0.0 and r["is_hallucination"])
    hallucination_rate = (hallucinations_in_zero / total_recall_zero) if total_recall_zero > 0 else 0.0

    return {
        "em": float(agg_em),
        "f1": float(agg_f1),
        "recall@5": float(agg_rec5),
        "recall@10": float(agg_rec10),
        "faithfulness": float(agg_faith),
        "p50_latency_ms": float(p50),
        "p95_latency_ms": float(p95),
        "hallucination_rate": float(hallucination_rate),
        "total_recall_zero": int(total_recall_zero),
        "hallucinations_in_zero": int(hallucinations_in_zero),
        "raw_results": results
    }

def main():
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline.instance()

    # Load OpenAI client if available
    client = None
    if settings.openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            print("OpenAI Client successfully loaded. Real LLM faithfulness checks enabled.")
        except Exception:
            print("Failed to load OpenAI client. Using local evaluation metrics.")

    # Load evaluation subsets
    with open("data/eval/hotpotqa_subset.json", "r", encoding="utf-8") as f:
        hotpotqa_qa = json.load(f)
    with open("data/eval/nq_subset.json", "r", encoding="utf-8") as f:
        nq_qa = json.load(f)

    hotpot_docs_path = Path("data/docs/hotpotqa")
    nq_docs_path = Path("data/docs/nq")

    # Storage for all run summaries
    sweep_results = {}

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 1: Main Comparison (4 methods x 2 datasets)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Running Experiment 1: Main Comparison ---")
    methods = {
        "BM25": ("bm25", False),
        "Dense": ("dense", False),
        "Hybrid": ("hybrid", False),
        "Reranker": ("hybrid", True)
    }

    experiment1_data = {}
    for ds_name, (qa, docs) in [("hotpotqa", (hotpotqa_qa, hotpot_docs_path)), ("nq", (nq_qa, nq_docs_path))]:
        experiment1_data[ds_name] = {}
        for m_name, (mode, reranker) in methods.items():
            print(f"Running {m_name} on {ds_name}...")
            # Settings: chunk_size=256, top_k=5
            res = run_dataset_eval(rag, qa, docs, mode, chunk_size=256, top_k=5, reranker_enabled=reranker, client=client)
            experiment1_data[ds_name][m_name] = res
            print(f"  -> F1: {res['f1']:.4f} | EM: {res['em']:.4f} | Recall@5: {res['recall@5']:.4f} | Latency P50: {res['p50_latency_ms']:.1f}ms")

    sweep_results["main_comparison"] = experiment1_data

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 2: Ablation on Chunk Size (128 vs 256 vs 512, on HotpotQA)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Running Experiment 2: Chunk Size Ablation ---")
    chunk_sizes = [128, 256, 512]
    ablation_chunk = {}
    for cs in chunk_sizes:
        ablation_chunk[cs] = {}
        for m_name in ["Dense", "Hybrid"]:
            mode, reranker = methods[m_name]
            print(f"Running {m_name} with chunk_size={cs}...")
            res = run_dataset_eval(rag, hotpotqa_qa, hotpot_docs_path, mode, chunk_size=cs, top_k=5, reranker_enabled=reranker, client=client)
            ablation_chunk[cs][m_name] = res
            print(f"  -> F1: {res['f1']:.4f} | Recall@5: {res['recall@5']:.4f}")

    sweep_results["ablation_chunk_size"] = ablation_chunk

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 3: Ablation on Top-K (k = 1, 3, 5, 10, on HotpotQA)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Running Experiment 3: Top-K Ablation ---")
    top_ks = [1, 3, 5, 10]
    ablation_k = {}
    for tk in top_ks:
        print(f"Running Hybrid with top_k={tk}...")
        res = run_dataset_eval(rag, hotpotqa_qa, hotpot_docs_path, "hybrid", chunk_size=256, top_k=tk, reranker_enabled=False, client=client)
        ablation_k[tk] = res
        print(f"  -> F1: {res['f1']:.4f} | Recall@5 (or @k): {res['recall@5']:.4f} | Latency P50: {res['p50_latency_ms']:.1f}ms")

    sweep_results["ablation_top_k"] = ablation_k

    # Save summary json
    with open("storage/eval_results/experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    print("\nAll experiments run! Raw JSON summary dumped to storage/eval_results/experiments_summary.json")

    # ──────────────────────────────────────────────────────────────────────────
    # PLOTTING & FIGURES GENERATION
    # ──────────────────────────────────────────────────────────────────────────
    print("\nGenerating Plots...")

    # Plot 1: Main Retrieval Method Comparison (HotpotQA vs NQ)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics_to_plot = ["recall@5", "recall@10", "f1"]
    x = np.arange(len(metrics_to_plot))
    width = 0.2

    for i, (ds_name, title) in enumerate([("hotpotqa", "HotpotQA (Multi-hop)"), ("nq", "NQ-open (Single-hop)")]):
        ax = axes[i]
        for idx, m_name in enumerate(methods.keys()):
            data = experiment1_data[ds_name][m_name]
            y_vals = [data[m] * 100 for m in metrics_to_plot]
            ax.bar(x + (idx - 1.5) * width, y_vals, width, label=m_name)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(["Recall@5", "Recall@10", "Answer F1"])
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        if i == 0:
            ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "retrieval_comparison.png", dpi=150)
    plt.close()

    # Plot 2: Ablation on Chunk Size
    fig, ax = plt.subplots(figsize=(7, 5))
    for m_name in ["Dense", "Hybrid"]:
        f1_vals = [ablation_chunk[cs][m_name]["f1"] * 100 for cs in chunk_sizes]
        ax.plot(chunk_sizes, f1_vals, marker='o', linewidth=2, label=f"{m_name} F1")
        
        rec5_vals = [ablation_chunk[cs][m_name]["recall@5"] * 100 for cs in chunk_sizes]
        ax.plot(chunk_sizes, rec5_vals, marker='s', linestyle='--', linewidth=1.5, label=f"{m_name} Recall@5")

    ax.set_title("Impact of Chunk Size on Performance (HotpotQA)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Chunk Size (tokens)")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(chunk_sizes)
    ax.grid(linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_chunk_size.png", dpi=150)
    plt.close()

    # Plot 3: Ablation on Top-K (Recall vs k and F1 vs k)
    fig, ax1 = plt.subplots(figsize=(7, 5))

    f1_k = [ablation_k[tk]["f1"] * 100 for tk in top_ks]
    rec5_k = [ablation_k[tk]["recall@5"] * 100 for tk in top_ks] # note: recall@5 will be evaluated over the actual top_k

    color = 'tab:blue'
    ax1.set_xlabel('Top-k (k)')
    ax1.set_ylabel('Answer F1 (%)', color=color)
    ax1.plot(top_ks, f1_k, color=color, marker='o', linewidth=2, label="Answer F1")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Retrieval Recall@5 (%)', color=color)
    ax2.plot(top_ks, rec5_k, color=color, marker='s', linestyle='--', linewidth=2, label="Recall@5")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Diminishing Returns Curve: Top-k Sweep (HotpotQA)", fontsize=12, fontweight='bold')
    fig.tight_layout()  
    plt.savefig(FIGURES_DIR / "ablation_top_k.png", dpi=150)
    plt.close()

    # Plot 4: Latency vs Quality Tradeoff
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    # Compile points for various configurations
    points = []
    # Main comparison configs:
    for m_name in methods.keys():
        data = experiment1_data["hotpotqa"][m_name]
        points.append((data["p50_latency_ms"], data["f1"] * 100, f"{m_name} (k=5)"))
    
    # Top-K ablation configs (Hybrid):
    for tk in [1, 3, 10]:
        data = ablation_k[tk]
        points.append((data["p50_latency_ms"], data["f1"] * 100, f"Hybrid (k={tk})"))

    # Chunk size configs:
    for cs in [128, 512]:
        for m in ["Dense", "Hybrid"]:
            data = ablation_chunk[cs][m]
            points.append((data["p50_latency_ms"], data["f1"] * 100, f"{m} (cs={cs})"))

    x_lat = [p[0] for p in points]
    y_f1 = [p[1] for p in points]
    labels = [p[2] for p in points]

    scatter = ax.scatter(x_lat, y_f1, c=y_f1, cmap='viridis', s=100, edgecolors='black', alpha=0.85)
    
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x_lat[i], y_f1[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_title("Latency vs. Quality Pareto Frontier (HotpotQA)", fontsize=12, fontweight='bold')
    ax.set_xlabel("p50 Latency (ms)")
    ax.set_ylabel("Answer F1 (%)")
    ax.grid(linestyle='--', alpha=0.7)
    fig.colorbar(scatter, label='F1 score (%)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "latency_vs_quality.png", dpi=150)
    plt.close()

    # Plot 5: Hallucination Analysis
    # Compare confidence vs actual accuracy when retrieval succeeds (Recall@5 > 0) vs fails (Recall@5 = 0)
    # We will gather these statistics from the Main Comparison "Reranker" run on HotpotQA
    reranker_raw = experiment1_data["hotpotqa"]["Reranker"]["raw_results"]
    
    succeed_runs = [r for r in reranker_raw if r["recall@5"] > 0.0]
    failed_runs = [r for r in reranker_raw if r["recall@5"] == 0.0]

    # Calculate confidence rate: fraction of runs where confident=True (rejected=False)
    conf_succeed = sum(1 for r in succeed_runs if r["confident"]) / len(succeed_runs) if succeed_runs else 0.0
    conf_failed = sum(1 for r in failed_runs if r["confident"]) / len(failed_runs) if failed_runs else 0.0

    # Calculate actual accuracy (represented by F1)
    acc_succeed = np.mean([r["f1"] for r in succeed_runs]) if succeed_runs else 0.0
    acc_failed = np.mean([r["f1"] for r in failed_runs]) if failed_runs else 0.0

    labels = ['Retrieval Success\n(Recall > 0)', 'Retrieval Failure\n(Recall = 0)']
    confidence_rates = [conf_succeed * 100, conf_failed * 100]
    actual_f1s = [acc_succeed * 100, acc_failed * 100]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, confidence_rates, width, label='LLM Confident Rate (%)', color='royalblue')
    ax.bar(x + width/2, actual_f1s, width, label='Actual Answer F1 (%)', color='tomato')

    ax.set_title("Hallucination Vulnerability Analysis", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    
    # Add annotations
    for i in x:
        ax.annotate(f"{confidence_rates[i]:.1f}%", (i - width/2, confidence_rates[i] + 2), ha='center', fontsize=9, fontweight='bold')
        ax.annotate(f"{actual_f1s[i]:.1f}%", (i + width/2, actual_f1s[i] + 2), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hallucination_analysis.png", dpi=150)
    plt.close()

    print("All evaluation plots generated successfully in 'figures/' directory!")

if __name__ == "__main__":
    main()
