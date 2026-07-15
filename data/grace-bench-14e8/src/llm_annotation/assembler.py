
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANNOTATIONS_DIR = _PROJECT_ROOT / "resources" / "results" / "annotations"
_TRACES_DIR = _PROJECT_ROOT / "resources" / "results" / "cot_traces" / "full"
_CRITIQUE_DIR = _PROJECT_ROOT / "resources" / "results" / "trace_analysis" / "open_critique"

DATASETS = ["logiqa", "musique", "reclor", "wiki2multihop"]
TIERS = ["gold", "silver", "bronze"]

def _build_trace_lookup(
    traces_dir: Path,
    model: str,
    dataset: str,
) -> dict[str, dict]:
    model_dir = model.lower().replace(" ", "-")
    jsonl_path = traces_dir / model_dir / f"{dataset}.jsonl"
    if not jsonl_path.exists():
        return {}

    lookup: dict[str, dict] = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = f"{record['id']}|{record.get('rep_index', 0)}"
                lookup[key] = record
            except (json.JSONDecodeError, KeyError):
                continue
    return lookup

def assemble_dataset(
    aggregated_dir: Path | None = None,
    traces_dir: Path | None = None,
    output_dir: Path | None = None,
    datasets: list[str] | None = None,
    tiers: list[str] | None = None,
) -> dict[str, int]:
    agg_dir = aggregated_dir or (_ANNOTATIONS_DIR / "aggregated")
    t_dir = traces_dir or _TRACES_DIR
    out_dir = output_dir or (_ANNOTATIONS_DIR / "final")
    target_datasets = datasets or DATASETS
    target_tiers = tiers or TIERS

    for tier in target_tiers:
        os.makedirs(out_dir / tier, exist_ok=True)

    os.makedirs(out_dir / "all", exist_ok=True)

    tier_counts: Counter = Counter()
    category_counts: Counter = Counter()
    trace_lookup_cache: dict[str, dict[str, dict]] = {}

    print(f"Assembling final dataset from {agg_dir}")
    print(f"Output: {out_dir}")
    print(f"Tiers: {target_tiers}\n")

    for dataset in target_datasets:
        agg_path = agg_dir / f"{dataset}.jsonl"
        if not agg_path.exists():
            print(f"  {dataset}: no aggregated file — skipping")
            continue

        annotations: list[dict] = []
        with open(agg_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    annotations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not annotations:
            continue

        tier_files = {}
        for tier in target_tiers:
            tier_files[tier] = open(out_dir / tier / f"{dataset}.jsonl", "w")
        all_file = open(out_dir / "all" / f"{dataset}.jsonl", "w")

        ds_counts = Counter()

        for ann in annotations:
            tier = ann.get("quality_tier", "rejected")
            if tier not in target_tiers:
                continue

            model = ann["model"]
            model_dir = model.lower().replace(" ", "-")
            cache_key = f"{model_dir}/{dataset}"
            if cache_key not in trace_lookup_cache:
                trace_lookup_cache[cache_key] = _build_trace_lookup(
                    t_dir, model, dataset
                )
            trace_lookup = trace_lookup_cache[cache_key]

            trace_key = f"{ann['trace_id']}|{ann.get('rep_index', 0)}"
            trace_record = trace_lookup.get(trace_key)

            step_text = ""
            previous_steps = []
            if trace_record:
                steps = trace_record.get("trace", {}).get("steps", [])
                for s in steps:
                    if s["id"] == ann["step_id"]:
                        step_text = s["text"]
                    elif s["id"] < ann["step_id"]:
                        previous_steps.append(s["text"])

            output_record = {
                "trace_id": ann["trace_id"],
                "rep_index": ann.get("rep_index", 0),
                "dataset": ann["dataset"],
                "model": ann["model"],
                "step_id": ann["step_id"],
                "step_text": step_text,
                "context": trace_record.get("context", "") if trace_record else "",
                "question": trace_record.get("question", "") if trace_record else "",
                "options": trace_record.get("options") if trace_record else None,
                "gold_answer": trace_record.get("gold_answer", "") if trace_record else "",
                "is_correct": trace_record.get("is_correct") if trace_record else None,
                "faithfulness": ann["faithfulness"],
                "error_category": ann["error_category"],
                "error_sub_flag": ann.get("error_sub_flag"),
                "severity": ann.get("severity"),
                "explanation": ann.get("explanation"),
                "quality_tier": tier,
                "annotation_meta": ann.get("annotation_meta", {}),
            }

            line_out = json.dumps(output_record, ensure_ascii=False) + "\n"
            tier_files[tier].write(line_out)
            all_file.write(line_out)

            tier_counts[tier] += 1
            ds_counts[tier] += 1
            if ann.get("error_category"):
                category_counts[ann["error_category"]] += 1

        for f in tier_files.values():
            f.close()
        all_file.close()

        total_ds = sum(ds_counts.values())
        print(f"  {dataset}: {total_ds:,} accepted "
              f"(gold={ds_counts.get('gold', 0)} "
              f"silver={ds_counts.get('silver', 0)} "
              f"bronze={ds_counts.get('bronze', 0)})")

    total = sum(tier_counts.values())
    print(f"\n{'━' * 60}")
    print(f"  ASSEMBLY SUMMARY")
    print(f"{'━' * 60}")
    print(f"  Total assembled : {total:,}")
    for tier in target_tiers:
        print(f"  {tier:16s}: {tier_counts.get(tier, 0):,}")
    print(f"  Category breakdown:")
    for cat, count in category_counts.most_common():
        print(f"    {cat:30s}: {count:,}")
    print(f"  Output          : {out_dir}")
    print(f"{'━' * 60}")

    return dict(tier_counts)
