#!/usr/bin/env bash
# reproduce.sh - Replication script for RAG retrieval empirical study

set -e

echo "========================================================================="
echo "   REPRODUCE: Empirical Study of Retrieval Strategies for Multi-Hop QA   "
echo "========================================================================="

echo -e "\n[Step 1/5] Installing project dependencies..."
python -m pip install -r requirements.txt

echo -e "\n[Step 2/5] Running regression tests..."
python -m pytest tests -q

echo -e "\n[Step 3/5] Generating curated datasets & document corpuses..."
python generate_eval_data.py

echo -e "\n[Step 4/5] Running experiment sweeps and generating figures..."
python run_experiments.py

echo -e "\n[Step 5/5] Compiling final LaTeX report to PDF..."
python report/compile_report.py

echo -e "\n========================================================================="
echo "   REPRODUCTION COMPLETED SUCCESSFULLY!                                 "
echo "   - Evaluation metrics: storage/eval_results/experiments_summary.json  "
echo "   - Generated plots: figures/                                          "
echo "   - Academic manuscript: report/paper.tex and report/paper.pdf         "
echo "========================================================================="
