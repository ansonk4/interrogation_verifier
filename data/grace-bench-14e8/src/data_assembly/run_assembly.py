
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_V3_DIR = _PROJECT_ROOT / "resources" / "results" / "data_assembly_v3"

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3.5: Data Assembly — Trace Filtering & Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["all", "aggregate", "score", "select", "focus", "assemble"],
        default="all",
        help="Which sub-phase to run (default: all). 'focus' runs focused train selection.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets to process (default: all 5)",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help="Path to tokenizer model for token counting.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.12,
        help="Fraction of Gold pool to sample for calibration (default: 0.12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for calibration sampling (default: 42)",
    )

    args = parser.parse_args()

    phases = (
        ["aggregate", "score", "select", "focus", "assemble"]
        if args.phase == "all"
        else [args.phase]
    )

    for phase in phases:
        t0 = time.time()
        print(f"\n{'#'*70}")
        print(f"# Phase 3.5: {phase.upper()}")
        print(f"{'#'*70}")

        if phase == "aggregate":
            from src.data_assembly.aggregator import run_aggregation
            run_aggregation(
                datasets=args.datasets,
                output_dir=_V3_DIR / "aggregated",
            )

        elif phase == "score":
            from src.data_assembly.trace_scorer import run_scoring
            run_scoring(
                datasets=args.datasets,
                aggregated_dir=_V3_DIR / "aggregated",
                output_dir=_V3_DIR / "trace_scores",
            )

        elif phase == "select":
            from src.data_assembly.refined_selector import run_refined_selection

            results = run_refined_selection(
                base_dir=_V3_DIR,
                output_dir=_V3_DIR,
                seed=args.seed,
            )

            print("\n✅ Selection complete.")

        elif phase == "focus":
            from src.data_assembly.focused_selector import run_focused_selection

            run_focused_selection(
                v3_dir=_V3_DIR,
                output_dir=_V3_DIR,
                seed=args.seed,
            )

            print("\n✅ Focused train selection complete.")

        elif phase == "assemble":
            from src.data_assembly.assembler import run_assembly
            run_assembly(
                datasets=args.datasets,
                aggregated_dir=_V3_DIR / "aggregated",
                scores_dir=_V3_DIR / "trace_scores",
                output_dir=_V3_DIR,
            )

        elapsed = time.time() - t0
        print(f"\n  ⏱  {phase} completed in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
