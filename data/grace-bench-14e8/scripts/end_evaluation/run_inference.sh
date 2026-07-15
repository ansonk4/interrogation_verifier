#!/bin/bash
set -euo pipefail

MODEL_NAME="Qwen3-8B"
VLLM_URLS="http://localhost:8000/v1"
INPUT_DIR="resources/final_data/grace_test/hard"
OUTPUT_DIR="resources/results/end_evaluation/inference/${MODEL_NAME,,}/hard"
MAX_CONCURRENT=64
TEMPERATURE=0.0
MAX_NEW_TOKENS=8192

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GRACE — Faithfulness Inference"
echo "  Model: ${MODEL_NAME}  |  Input: ${INPUT_DIR}"
echo "  Server: ${VLLM_URLS}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PYTHONPATH=. python3 -m src.end_evaluation.run_inference \
    --input "$INPUT_DIR" \
    --model-name "$MODEL_NAME" \
    --vllm-base-url $VLLM_URLS \
    --output-dir "$OUTPUT_DIR" \
    --max-concurrent "$MAX_CONCURRENT" \
    --temperature "$TEMPERATURE" \
    --max-new-tokens "$MAX_NEW_TOKENS"

echo "✓ Inference done → ${OUTPUT_DIR}"
