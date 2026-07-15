#!/bin/bash

set -euo pipefail

MODEL_NAME="Qwen3.5-27B-FP8"
VLLM_URLS="http://localhost:8000/v1"
MODE="full"                          

DATASETS=(
    "musique"
    "reclor"
    "logiqa"
    "wiki2multihop"
)

get_n_repetition() {
    case "$1" in
        reclor)        echo 2 ;;
        *)             echo 1 ;;
    esac
}

TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
MAX_NEW_TOKENS=4096
MAX_CONCURRENT=96
ENABLE_THINKING=false

THINKING_FLAG=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_FLAG="--enable-thinking"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GRACE — CoT Trace Generation"
echo "  Model: ${MODEL_NAME}  |  Mode: ${MODE}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Server: ${VLLM_URLS}"
echo "  Thinking: ${ENABLE_THINKING}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for DATASET in "${DATASETS[@]}"; do
    N_REP=$(get_n_repetition "$DATASET")
    echo "▶ Starting ${DATASET} (×${N_REP})..."
    uv run python -m src.cot_generation.generate_traces \
        --dataset "$DATASET" \
        --model-name "$MODEL_NAME" \
        --vllm-base-url $VLLM_URLS \
        --mode "$MODE" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --top-k "$TOP_K" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --max-concurrent "$MAX_CONCURRENT" \
        --n-repetition "$N_REP" \
        $THINKING_FLAG
    echo ""
done

echo "✓ All datasets done for ${MODEL_NAME} (${MODE} mode)"
