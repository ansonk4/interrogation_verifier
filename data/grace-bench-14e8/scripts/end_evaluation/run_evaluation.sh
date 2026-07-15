#!/bin/bash
set -euo pipefail

INPUT_DIR="/home/hoangpham/grace/resources/results/end_evaluation/infer_base_models_allsteps/gemini-3.1-pro-preview/hard/"
OUTPUT_DIR="temp"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GRACE — Evaluation"
echo "  Input: ${INPUT_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PYTHONPATH=. python3 -m src.end_evaluation.run_evaluation \
    --input "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "✓ Evaluation done → ${OUTPUT_DIR}"
