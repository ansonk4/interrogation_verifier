#!/bin/bash

set -euo pipefail

JUDGE_DIRS=(
    "resources/results/annotations/judges/judge_0_qwen3.5-27b-fp8"
)

MIN_JUDGES=2
MIN_FAITHFULNESS_AGREEMENT=0.667
MIN_CATEGORY_AGREEMENT=0.667

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GRACE — Phase 3B: Multi-Judge Aggregation"
echo "  Judges: ${#JUDGE_DIRS[@]}"
echo "  Min judges: ${MIN_JUDGES}"
echo "  Min faithfulness agreement: ${MIN_FAITHFULNESS_AGREEMENT}"
echo "  Min category agreement: ${MIN_CATEGORY_AGREEMENT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

uv run python -m src.llm_annotation.run_annotation \
    --phase aggregate \
    --judge-dirs "${JUDGE_DIRS[@]}" \
    --min-judges "$MIN_JUDGES" \
    --min-faithfulness-agreement "$MIN_FAITHFULNESS_AGREEMENT" \
    --min-category-agreement "$MIN_CATEGORY_AGREEMENT"

echo ""
echo "✓ Phase 3B complete"
echo "  Next: bash scripts/llm_annotation/run_assemble.sh"
