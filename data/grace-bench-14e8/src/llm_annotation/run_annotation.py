
from __future__ import annotations

import asyncio
import json
import os
from argparse import ArgumentParser
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANNOTATIONS_DIR = _PROJECT_ROOT / "resources" / "results" / "annotations"

def run_judge(args) -> None:
    from src.llm_annotation.loader import load_traces_for_annotation
    from src.llm_annotation.prompts import load_grounding_taxonomy
    from src.llm_annotation.judge import annotate_traces

    taxonomy_path = Path(args.taxonomy_path) if args.taxonomy_path else None
    taxonomy = load_grounding_taxonomy(taxonomy_path)
    print(f"Taxonomy: {taxonomy.get('total_categories', '?')} categories")

    judge_dir_name = f"{args.judge_id}_{args.judge_model.lower().replace('/', '_')}"
    output_dir = Path(args.output_dir) if args.output_dir else (
        _ANNOTATIONS_DIR / "judges" / judge_dir_name
    )

    print("=" * 60)
    print("  GRACE: Per-Judge Annotation (all-steps-at-once)")
    print("=" * 60)
    print(f"  Judge model : {args.judge_model}")
    print(f"  Judge ID    : {args.judge_id}")
    print(f"  Output      : {output_dir}")
    print()

    filter_models = args.filter_models if args.filter_models else None
    filter_datasets = args.filter_datasets if args.filter_datasets else None
    traces_dir = Path(args.traces_dir) if args.traces_dir else None
    critique_dir = Path(args.critique_dir) if args.critique_dir else None

    traces = load_traces_for_annotation(
        traces_dir=traces_dir,
        critique_dir=critique_dir,
        evaluator_model=args.evaluator_model,
        filter_models=filter_models,
        filter_datasets=filter_datasets,
        sample_clean_per_model=args.sample_clean_per_model,
        include_all=args.include_all,
    )

    if not traces:
        print("No traces found — nothing to annotate.")
        return

    total_steps = sum(t.num_steps for t in traces)
    print(f"\nTraces to annotate: {len(traces):,} ({total_steps:,} steps)")

    asyncio.run(annotate_traces(
        traces=traces,
        judge_model=args.judge_model,
        judge_id=args.judge_id,
        taxonomy=taxonomy,
        vllm_base_urls=args.vllm_base_url,
        output_dir=output_dir,
        api_key=args.api_key,
        max_concurrent=args.max_concurrent,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
    ))

def run_aggregate(args) -> None:
    from src.llm_annotation.aggregator import aggregate_annotations

    judge_dirs = [Path(d) for d in args.judge_dirs]
    output_dir = Path(args.output_dir) if args.output_dir else None
    filter_datasets = args.filter_datasets if args.filter_datasets else None

    print("=" * 60)
    print("  GRACE Phase 3B: Multi-Judge Aggregation")
    print("=" * 60)

    for jd in judge_dirs:
        if not jd.exists():
            print(f"[ERROR] Judge directory not found: {jd}")
            return

    aggregate_annotations(
        judge_dirs=judge_dirs,
        output_dir=output_dir,
        datasets=filter_datasets,
        min_judges=args.min_judges,
        min_faithfulness_agreement=args.min_faithfulness_agreement,
        min_category_agreement=args.min_category_agreement,
    )

def run_assemble(args) -> None:
    from src.llm_annotation.assembler import assemble_dataset

    aggregated_dir = Path(args.aggregated_dir) if args.aggregated_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    filter_datasets = args.filter_datasets if args.filter_datasets else None

    print("=" * 60)
    print("  GRACE Phase 3C: Dataset Assembly")
    print("=" * 60)

    assemble_dataset(
        aggregated_dir=aggregated_dir,
        output_dir=output_dir,
        datasets=filter_datasets,
    )

def parse_args():
    parser = ArgumentParser(
        description="GRACE Phase 3: LLM-as-Annotator — multi-judge error taxonomy labeling.",
    )

    parser.add_argument(
        "--phase", type=str, required=True,
        choices=["judge", "aggregate", "assemble"],
        help="Which phase of the annotation pipeline to run.",
    )

    parser.add_argument(
        "--judge-model", type=str, default="Qwen3.5-27B-FP8",
        help="Model name for this judge.",
    )
    parser.add_argument(
        "--judge-id", type=str, default="judge_0",
        help="Unique judge identifier (e.g., judge_0, judge_1).",
    )
    parser.add_argument(
        "--evaluator-model", type=str, default="",
        help="(Optional) Evaluator model name used during trace selection.",
    )

    parser.add_argument(
        "--vllm-base-url", type=str, nargs="+",
        default=["http://localhost:8105/v1"],
        help="One or more vLLM OpenAI-compatible base URLs.",
    )
    parser.add_argument(
        "--api-key", type=str, default="unused",
        help="API key for the OpenAI-compatible server.",
    )

    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7).")
    parser.add_argument("--top-p", type=float, default=0.8,
                        help="Top-p (nucleus) sampling (default: 0.8).")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-k sampling (default: 20).")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max response tokens (default: 8192 for all-steps).")
    parser.add_argument("--max-concurrent", type=int, default=64,
                        help="Max concurrent async requests across all servers.")
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                        help="Enable model thinking mode.")

    parser.add_argument(
        "--sample-clean-per-model", type=int, default=2000,
        help="Number of clean traces (no Phase A errors) to sample per model "
             "for false-negative detection (default: 2000).",
    )
    parser.add_argument(
        "--include-all", action="store_true", default=True,
        help="Skip selection and annotate ALL traces (ignores Phase A priority).",
    )

    parser.add_argument(
        "--filter-models", type=str, nargs="+", default=None,
        help="Only annotate traces from these models.",
    )
    parser.add_argument(
        "--filter-datasets", type=str, nargs="+", default=None,
        help="Only annotate traces from these datasets.",
    )

    parser.add_argument(
        "--judge-dirs", type=str, nargs="+", default=None,
        help="Paths to judge output directories (for aggregation).",
    )
    parser.add_argument(
        "--min-judges", type=int, default=2,
        help="Minimum number of judges required for acceptance (default: 2).",
    )
    parser.add_argument(
        "--min-faithfulness-agreement", type=float, default=0.667,
        help="Minimum faithfulness agreement fraction (default: 0.667).",
    )
    parser.add_argument(
        "--min-category-agreement", type=float, default=0.667,
        help="Minimum category agreement fraction (default: 0.667).",
    )

    parser.add_argument(
        "--aggregated-dir", type=str, default=None,
        help="Override path to aggregated annotations directory.",
    )

    parser.add_argument(
        "--taxonomy-path", type=str, default=None,
        help="Override path to taxonomy JSON file.",
    )
    parser.add_argument(
        "--traces-dir", type=str, default=None,
        help="Override path to Phase 2a traces directory.",
    )
    parser.add_argument(
        "--critique-dir", type=str, default=None,
        help="(Not required) Path to open critique results for trace selection priority.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory for the current phase.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    print(json.dumps(
        {k: v for k, v in vars(args).items() if v is not None},
        indent=4, ensure_ascii=False,
    ))
    print()

    if args.phase == "judge":
        run_judge(args)
    elif args.phase == "aggregate":
        run_aggregate(args)
    elif args.phase == "assemble":
        run_assemble(args)

if __name__ == "__main__":
    main()
