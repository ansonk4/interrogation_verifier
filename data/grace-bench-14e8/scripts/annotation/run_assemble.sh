#!/bin/bash

set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GRACE — Phase 3C: Dataset Assembly"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

uv run python -m src.llm_annotation.run_annotation \
    --phase assemble

echo ""
echo "✓ Phase 3C complete"
echo "  Final dataset: resources/results/annotations/final/"
echo "  Tiers: gold/, silver/, bronze/, all/"
