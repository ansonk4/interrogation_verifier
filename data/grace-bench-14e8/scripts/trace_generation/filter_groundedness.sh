#!/bin/bash

set -euo pipefail

DATASET="logiqa"
MODEL_NAME="Qwen3.5-27B-FP8"
OUTPUT_DIR="resources/results/curation/groundedness_filter/${DATASET}"

VLLM_URLS="http://localhost:8000/v1"

MAX_CONCURRENT=180
N_REPETITION=1
TEMPERATURE=0.7
TOP_P=0.85
TOP_K=20
MIN_P=0.0
PRESENCE_PENALTY=1.5
REPETITION_PENALTY=1.0
MAX_NEW_TOKENS=4096
ENABLE_THINKING=false

THINKING_FLAG=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_FLAG="--enable-thinking"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GRACE — LLM Groundedness Filter (DP-003)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Dataset     : ${DATASET}"
echo "  Model       : ${MODEL_NAME}"
echo "  Servers     : ${VLLM_URLS}"
echo "  Output      : ${OUTPUT_DIR}"
echo "  Concurrent  : ${MAX_CONCURRENT}"
echo "  Repetitions : ${N_REPETITION}"
echo "  Temperature : ${TEMPERATURE}"
echo "  Thinking    : ${ENABLE_THINKING}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

uv run python -m src.data_processing.stages.filter_groundedness_llm \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME" \
    --vllm-base-url-list $VLLM_URLS \
    --max-concurrent "$MAX_CONCURRENT" \
    --n-repetition "$N_REPETITION" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --min-p "$MIN_P" \
    --presence-penalty "$PRESENCE_PENALTY" \
    --repetition-penalty "$REPETITION_PENALTY" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    $THINKING_FLAG

echo ""
echo "✓ Done! Results saved to ${OUTPUT_DIR}/"
