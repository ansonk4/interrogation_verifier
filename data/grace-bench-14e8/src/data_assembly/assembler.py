
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.data_assembly.aggregator import DATASETS
from src.data_assembly.trace_scorer import TraceScore

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACES_DIR = _PROJECT_ROOT / "resources" / "results" / "cot_traces" / "full"
AGGREGATED_DIR = _PROJECT_ROOT / "resources" / "results" / "data_assembly" / "aggregated"
SCORES_DIR = _PROJECT_ROOT / "resources" / "results" / "data_assembly" / "trace_scores"
OUTPUT_DIR = _PROJECT_ROOT / "resources" / "results" / "data_assembly"

TRACE_MODELS_DIRS = [
    "llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3.5-27b-fp8",
    "qwen3.5-35b-a3b-fp8",
    "ministral-3-14b-instruct-2512",
    "gemma-4-31b-it-awq",
    "gpt-4o-mini",
    "gemini-3-flash-preview",
]

LOGIC_DATASETS = {"logiqa", "reclor"}
EVIDENCE_DATASETS = {"musique", "wiki2multihop"}

SPLIT_DIRS = {
    "train_gold":        "grace_train/curated/gold",
    "train_silver":      "grace_train/curated/silver",
    "train_focused":     "grace_train/focused",
    "test_hard":         "grace_test/hard",
    "test_calibration":  "grace_test/calibration",
}

def _get_track(dataset: str) -> str:
    if dataset in LOGIC_DATASETS:
        return "logic"
    return "evidence"

def _normalize_model(model_name: str) -> str:
    return model_name.lower().replace(" ", "-")

def _load_original_traces(
    dataset: str,
    traces_dir: Path | None = None,
) -> dict[str, dict]:
    base = traces_dir or TRACES_DIR
    traces: dict[str, dict] = {}

    for model_dir in TRACE_MODELS_DIRS:
        jsonl_path = base / model_dir / f"{dataset}.jsonl"
        if not jsonl_path.exists():
            continue
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = f"{d['id']}|{d.get('rep_index', 0)}|{_normalize_model(d['model'])}"
                traces[key] = d

    return traces

def _load_aggregated_by_trace(
    dataset: str,
    aggregated_dir: Path | None = None,
) -> dict[str, list[dict]]:
    base = aggregated_dir or AGGREGATED_DIR
    path = base / f"{dataset}.jsonl"

    groups: dict[str, list[dict]] = defaultdict(list)
    if not path.exists():
        return groups

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = f"{d['trace_id']}|{d['rep_index']}|{_normalize_model(d['model'])}"
            groups[key].append(d)

    for key in groups:
        groups[key].sort(key=lambda s: s["step_id"])

    return groups

def _load_trace_scores(
    dataset: str,
    scores_dir: Path | None = None,
) -> dict[str, dict]:
    base = scores_dir or SCORES_DIR
    path = base / f"{dataset}.jsonl"

    scores: dict[str, dict] = {}
    if not path.exists():
        return scores

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = f"{d['trace_id']}|{d['rep_index']}|{_normalize_model(d['model'])}"
            scores[key] = d

    return scores

def _load_split_trace_keys(
    dataset: str,
    split_name: str,
    output_dir: Path | None = None,
) -> set[str]:
    base = output_dir or OUTPUT_DIR
    split_subdir = SPLIT_DIRS[split_name]
    path = base / split_subdir / f"{dataset}.jsonl"

    keys: set[str] = set()
    if not path.exists():
        return keys

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = f"{d['trace_id']}|{d['rep_index']}|{_normalize_model(d['model'])}"
            keys.add(key)

    return keys

_TITLED_PASSAGE_DATASETS = {"musique", "wiki2multihop"}
_SINGLE_REF_DATASETS = {"logiqa", "reclor"}
_PARAGRAPH_DATASETS = set()

_MERGE_MIN_CHARS = 500

def _parse_context_to_passages(context: str, dataset: str) -> list[dict]:
    if not context or not context.strip():
        return []

    passages: list[dict] = []

    if dataset in _TITLED_PASSAGE_DATASETS:
        parts = re.split(r'(?=^\[)', context, flags=re.MULTILINE)
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            m = re.match(r'^\[([^\]]+)\]\s*(.*)', part, re.DOTALL)
            if m:
                passages.append({
                    "ref_id": f"ref_{len(passages) + 1}",
                    "title": m.group(1).strip(),
                    "text": m.group(2).strip(),
                })
            else:
                passages.append({
                    "ref_id": f"ref_{len(passages) + 1}",
                    "title": None,
                    "text": part,
                })

    elif dataset in _PARAGRAPH_DATASETS:
        raw_paras = [p.strip() for p in context.split("\n\n") if p.strip()]
        if not raw_paras:
            passages.append({"ref_id": "ref_1", "title": None, "text": context.strip()})
        else:
            chunks: list[str] = []
            current = raw_paras[0]
            for p in raw_paras[1:]:
                if len(current) < _MERGE_MIN_CHARS:
                    current = current + "\n\n" + p
                else:
                    chunks.append(current)
                    current = p
            chunks.append(current)
            for i, chunk in enumerate(chunks):
                passages.append({
                    "ref_id": f"ref_{i + 1}",
                    "title": None,
                    "text": chunk,
                })
    else:
        passages.append({
            "ref_id": "ref_1",
            "title": None,
            "text": context.strip(),
        })

    return passages

def _assemble_trace(
    original: dict,
    agg_steps: list[dict],
    score: dict | None,
    split_name: str,
    dataset: str,
) -> dict:
    trace_data = original.get("trace", {})
    original_steps = trace_data.get("steps", [])

    original_by_id = {s["id"]: s for s in original_steps}

    assembled_steps = []
    for agg in agg_steps:
        step_id = agg["step_id"]
        orig_step = original_by_id.get(step_id, {})

        step_tier = "unknown"
        if score and "step_summaries" in score:
            for ss in score["step_summaries"]:
                if ss["step_id"] == step_id:
                    step_tier = ss["step_tier"]
                    break

        raw_faith = agg.get("faithfulness")
        if raw_faith == "neutral":
            faithfulness = "faithful"
            error_category = None
            error_sub_flag = None
            severity = None
        else:
            faithfulness = raw_faith
            error_category = agg.get("error_category")
            error_sub_flag = agg.get("error_sub_flag")
            severity = agg.get("severity")

        assembled_steps.append({
            "step_id": step_id,
            "text": orig_step.get("text", agg.get("step_text", "")),
            "text_without_citations": orig_step.get("text_without_citations", ""),
            "citations": orig_step.get("citations", []),

            "faithfulness": faithfulness,
            "error_category": error_category,
            "error_sub_flag": error_sub_flag,
            "severity": severity,
            "explanation": agg.get("explanation"),
            "explanation_source": agg.get("explanation_source"),

            "n_judges": agg.get("n_judges", 0),
            "faithfulness_agreement": agg.get("faithfulness_agreement", 0.0),
            "category_agreement": agg.get("category_agreement"),
            "is_unanimous": agg.get("is_unanimous_faithfulness", False),
            "step_tier": step_tier,

            "per_judge": agg.get("per_judge", []),
        })

    record = {
        "trace_id": original["id"],
        "rep_index": original.get("rep_index", 0),
        "dataset": dataset,
        "track": _get_track(dataset),
        "model": original["model"],
        "split": split_name,
        "is_correct": original.get("is_correct"),

        "context": original.get("context", ""),
        "passages": _parse_context_to_passages(original.get("context", ""), dataset),
        "question": original.get("question", ""),
        "options": original.get("options"),
        "gold_answer": original.get("gold_answer", ""),

        "num_steps": len(assembled_steps),
        "final_answer": trace_data.get("final_answer", ""),

        "steps": assembled_steps,

        "trace_tier": score.get("trace_tier", "unknown") if score else "unknown",
        "disagreement_score": score.get("disagreement_score", 0.0) if score else 0.0,
        "num_faithful_steps": score.get("num_faithful_steps", 0) if score else 0,
        "num_unfaithful_steps": score.get("num_unfaithful_steps", 0) if score else 0,
        "error_categories_present": score.get("error_categories_present", []) if score else [],
    }

    return record

def run_assembly(
    datasets: list[str] | None = None,
    traces_dir: Path | None = None,
    aggregated_dir: Path | None = None,
    scores_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, dict[str, int]]:
    target_datasets = datasets or DATASETS
    out_dir = output_dir or OUTPUT_DIR

    stats: dict[str, dict[str, int]] = {}

    for ds in target_datasets:
        print(f"\n{'='*60}")
        print(f"Assembling final dataset: {ds}")
        print(f"{'='*60}")

        print("  Loading original traces...")
        originals = _load_original_traces(ds, traces_dir)
        print(f"    {len(originals):,} traces loaded")

        print("  Loading aggregated annotations...")
        agg_by_trace = _load_aggregated_by_trace(ds, aggregated_dir)
        print(f"    {len(agg_by_trace):,} annotated traces")

        print("  Loading trace scores...")
        scores = _load_trace_scores(ds, scores_dir)
        print(f"    {len(scores):,} scored traces")

        stats[ds] = {}

        for split_name, split_subdir in SPLIT_DIRS.items():
            split_keys = _load_split_trace_keys(ds, split_name, out_dir)
            if not split_keys:
                stats[ds][split_name] = 0
                continue

            assembled: list[dict] = []
            missing = 0

            for key in sorted(split_keys):
                original = originals.get(key)
                agg_steps = agg_by_trace.get(key, [])
                score = scores.get(key)

                if original is None:
                    missing += 1
                    continue

                record = _assemble_trace(original, agg_steps, score, split_name, ds)
                assembled.append(record)

            assembled.sort(key=lambda r: (r["model"], r["trace_id"], r["rep_index"]))

            final_dir = out_dir / split_subdir
            final_dir.mkdir(parents=True, exist_ok=True)
            out_path = final_dir / f"{ds}.jsonl"
            with open(out_path, "w") as f:
                for record in assembled:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats[ds][split_name] = len(assembled)
            if missing:
                print(f"    [WARN] {split_name}: {missing} traces missing from original data")
            print(f"    {split_name:20s}: {len(assembled):>6,} traces → {out_path}")

    print(f"\n{'='*60}")
    print("Assembly Summary")
    print(f"{'='*60}")
    print(f"  {'Dataset':<15s} {'Train Gold':>11s} {'Train Silver':>13s} {'Test Hard':>10s} {'Test Cal.':>10s}")
    total = {s: 0 for s in SPLIT_DIRS}
    for ds in target_datasets:
        row = stats.get(ds, {})
        g = row.get("train_gold", 0)
        s = row.get("train_silver", 0)
        h = row.get("test_hard", 0)
        c = row.get("test_calibration", 0)
        total["train_gold"] += g
        total["train_silver"] += s
        total["test_hard"] += h
        total["test_calibration"] += c
        print(f"  {ds:<15s} {g:>11,} {s:>13,} {h:>10,} {c:>10,}")
    print(f"  {'TOTAL':<15s} {total['train_gold']:>11,} {total['train_silver']:>13,} {total['test_hard']:>10,} {total['test_calibration']:>10,}")

    return stats
