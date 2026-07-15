#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

CALIBRATION_FRACTION="${CALIBRATION_FRACTION:-0.12}"
SEED="${SEED:-42}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Data Assembly — Full Pipeline                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Calibration:  ${CALIBRATION_FRACTION}"
echo "  Seed:         ${SEED}"
echo ""

uv run python -m src.data_assembly.run_assembly \
    --phase all \
    --calibration-fraction "${CALIBRATION_FRACTION}" \
    --seed "${SEED}"
