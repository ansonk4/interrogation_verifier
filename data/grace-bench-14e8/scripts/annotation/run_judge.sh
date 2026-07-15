#!/bin/bash

set -euo pipefail

JUDGE_MODEL="Qwen3.5-35B-A3B-FP8"
JUDGE_ID="judge_qwen3.5_35b_a3b_fp8"

EVALUATOR_MODEL="qwen3.5-27b-fp8"

VLLM_URLS="http://localhost:8000/v1"

SAMPLE_CLEAN_PER_MODEL=2000

TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
MAX_TOKENS=2048
MAX_CONCURRENT=160
ENABLE_THINKING=false

FILTER_MODELS=""
FILTER_DATASETS=""

THINKING_FLAG=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_FLAG="--enable-thinking"
fi

MODEL_FILTER=""
if [ -n "$FILTER_MODELS" ]; then
    MODEL_FILTER="--filter-models $FILTER_MODELS"
fi

DATASET_FILTER=""
if [ -n "$FILTER_DATASETS" ]; then
    DATASET_FILTER="--filter-datasets $FILTER_DATASETS"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GRACE — Phase 3A: Per-Judge Annotation (from scratch)"
echo "  Judge model: ${JUDGE_MODEL}"
echo "  Judge ID   : ${JUDGE_ID}"
echo "  Server     : ${VLLM_URLS}"
echo "  Clean sample: ${SAMPLE_CLEAN_PER_MODEL}/model"
echo "  Thinking   : ${ENABLE_THINKING}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

uv run python -m src.llm_annotation.run_annotation \
    --phase judge \
    --judge-model "$JUDGE_MODEL" \
    --judge-id "$JUDGE_ID" \
    --evaluator-model "$EVALUATOR_MODEL" \
    --vllm-base-url $VLLM_URLS \
    --sample-clean-per-model "$SAMPLE_CLEAN_PER_MODEL" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --max-concurrent "$MAX_CONCURRENT" \
    $THINKING_FLAG \
    $MODEL_FILTER \
    $DATASET_FILTER

echo ""
echo "✓ Phase 3A complete for ${JUDGE_ID}"
echo "  Next: run again with different JUDGE_MODEL/JUDGE_ID, or aggregate:"
echo "  bash scripts/llm_annotation/run_aggregate.sh"
