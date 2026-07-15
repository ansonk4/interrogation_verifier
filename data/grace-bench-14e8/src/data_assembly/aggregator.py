
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

JUDGES_DIR = _PROJECT_ROOT / "resources" / "results" / "annotations" / "judges"
OUTPUT_DIR = _PROJECT_ROOT / "resources" / "results" / "data_assembly" / "aggregated"

DATASETS = ["logiqa", "musique", "reclor", "wiki2multihop"]

JUDGE_DIRS = {
    "gemma4": "judge_gemma4_31b_gemma-4-31b-it-awq",
    "qwen27b": "judge_qwen3.5_27b_fp8_qwen3.5-27b-fp8",
    "qwen35b": "judge_qwen3.5_35b_a3b_fp8_qwen3.5-35b-a3b-fp8",
}

JUDGE_PREFERENCE = ["gemma4", "qwen27b", "qwen35b"]

JUDGE_TIERS = {
    "gemma4": "A",
    "qwen27b": "A",
    "qwen35b": "B",
}

@dataclass
class JudgeAnnotation:

    judge_id: str
    judge_tier: str
    faithfulness: str | None
    error_category: str | None
    error_sub_flag: str | None
    severity: str | None
    confidence: float | None
    explanation: str | None
    parse_error: bool

@dataclass
class AggregatedStep:

    trace_id: str
    rep_index: int
    dataset: str
    model: str
    step_id: int
    step_text: str
    is_correct: bool | None

    faithfulness: str | None
    error_category: str | None
    error_sub_flag: str | None
    severity: str | None

    n_judges: int
    faithfulness_agreement: float
    category_agreement: float | None
    is_unanimous_faithfulness: bool
    is_unanimous_category: bool | None

    explanation: str | None
    explanation_source: str | None

    per_judge: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

def _majority_vote(values: list[str | None]) -> tuple[str | None, float]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None, 0.0
    counter = Counter(filtered)
    winner, count = counter.most_common(1)[0]
    return winner, count / len(filtered)

def _select_explanation(
    annotations: list[JudgeAnnotation],
    consensus_category: str | None,
) -> tuple[str | None, str | None]:
    for judge_id in JUDGE_PREFERENCE:
        for ann in annotations:
            if ann.judge_id == judge_id and ann.explanation:
                if consensus_category is None or ann.error_category == consensus_category:
                    return ann.explanation, judge_id

    for judge_id in JUDGE_PREFERENCE:
        for ann in annotations:
            if ann.judge_id == judge_id and ann.explanation:
                return ann.explanation, judge_id

    return None, None

def aggregate_step(annotations: list[JudgeAnnotation]) -> AggregatedStep | None:
    valid = [a for a in annotations if not a.parse_error and a.faithfulness is not None]
    if not valid:
        return None

    first = annotations[0]

    faith_labels = [a.faithfulness for a in valid]
    faith_winner, faith_agreement = _majority_vote(faith_labels)
    is_unanimous_faith = faith_agreement == 1.0

    cat_winner = None
    cat_agreement = None
    is_unanimous_cat = None
    sub_flag_winner = None

    if faith_winner == "unfaithful":
        unfaith_judges = [a for a in valid if a.faithfulness == "unfaithful"]
        if unfaith_judges:
            cat_labels = [a.error_category for a in unfaith_judges]
            cat_winner, cat_agreement = _majority_vote(cat_labels)
            is_unanimous_cat = cat_agreement == 1.0

            if is_unanimous_cat:
                sub_labels = [a.error_sub_flag for a in unfaith_judges]
                sub_flag_winner, _ = _majority_vote(sub_labels)

    sev_labels = [a.severity for a in valid if a.severity]
    sev_winner, _ = _majority_vote(sev_labels)

    explanation, explanation_source = _select_explanation(valid, cat_winner)

    per_judge = []
    for a in annotations:
        per_judge.append({
            "judge_id": a.judge_id,
            "judge_tier": a.judge_tier,
            "faithfulness": a.faithfulness,
            "error_category": a.error_category,
            "error_sub_flag": a.error_sub_flag,
            "severity": a.severity,
            "confidence": a.confidence,
            "explanation": a.explanation,
            "parse_error": a.parse_error,
        })

    return AggregatedStep(
        trace_id=first.trace_id if hasattr(first, "trace_id") else "",
        rep_index=first.rep_index if hasattr(first, "rep_index") else 0,
        dataset=first.dataset if hasattr(first, "dataset") else "",
        model=first.model if hasattr(first, "model") else "",
        step_id=first.step_id if hasattr(first, "step_id") else 0,
        step_text=first.step_text if hasattr(first, "step_text") else "",
        is_correct=first.is_correct if hasattr(first, "is_correct") else None,
        faithfulness=faith_winner,
        error_category=cat_winner,
        error_sub_flag=sub_flag_winner,
        severity=sev_winner,
        n_judges=len(valid),
        faithfulness_agreement=faith_agreement,
        category_agreement=cat_agreement,
        is_unanimous_faithfulness=is_unanimous_faith,
        is_unanimous_category=is_unanimous_cat,
        explanation=explanation,
        explanation_source=explanation_source,
        per_judge=per_judge,
    )

@dataclass
class _RawJudgeRecord:

    trace_id: str
    rep_index: int
    dataset: str
    model: str
    step_id: int
    step_text: str
    is_correct: bool | None
    judge_id: str
    judge_tier: str
    faithfulness: str | None
    error_category: str | None
    error_sub_flag: str | None
    severity: str | None
    confidence: float | None
    explanation: str | None
    parse_error: bool

def _load_judge_records(dataset: str, judges_dir: Path | None = None) -> dict[str, list[_RawJudgeRecord]]:
    base = judges_dir or JUDGES_DIR
    records_by_step: dict[str, list[_RawJudgeRecord]] = {}

    for judge_id, judge_dir_name in JUDGE_DIRS.items():
        judge_tier = JUDGE_TIERS[judge_id]
        jsonl_path = base / judge_dir_name / f"{dataset}.jsonl"

        if not jsonl_path.exists():
            print(f"  [WARN] Missing: {jsonl_path}")
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

                step_key = f"{d['trace_id']}|{d['rep_index']}|{d['model']}|{d['step_id']}"
                rec = _RawJudgeRecord(
                    trace_id=d["trace_id"],
                    rep_index=d["rep_index"],
                    dataset=d["dataset"],
                    model=d["model"],
                    step_id=d["step_id"],
                    step_text=d.get("step_text", ""),
                    is_correct=d.get("is_correct"),
                    judge_id=judge_id,
                    judge_tier=judge_tier,
                    faithfulness=d.get("faithfulness"),
                    error_category=d.get("error_category"),
                    error_sub_flag=d.get("error_sub_flag"),
                    severity=d.get("severity"),
                    confidence=d.get("confidence"),
                    explanation=d.get("explanation"),
                    parse_error=d.get("parse_error", False),
                )
                records_by_step.setdefault(step_key, []).append(rec)

    return records_by_step

def run_aggregation(
    datasets: list[str] | None = None,
    judges_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, list[AggregatedStep]]:
    target_datasets = datasets or DATASETS
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[AggregatedStep]] = {}

    for ds in target_datasets:
        print(f"\n{'='*60}")
        print(f"Aggregating: {ds}")
        print(f"{'='*60}")

        records_by_step = _load_judge_records(ds, judges_dir)
        print(f"  Loaded {len(records_by_step):,} unique steps across judges")

        aggregated: list[AggregatedStep] = []
        skipped = 0

        for step_key, raw_records in records_by_step.items():
            annotations = []
            for r in raw_records:
                annotations.append(JudgeAnnotation(
                    judge_id=r.judge_id,
                    judge_tier=r.judge_tier,
                    faithfulness=r.faithfulness,
                    error_category=r.error_category,
                    error_sub_flag=r.error_sub_flag,
                    severity=r.severity,
                    confidence=r.confidence,
                    explanation=r.explanation,
                    parse_error=r.parse_error,
                ))

            first = raw_records[0]
            result = aggregate_step(annotations)
            if result is None:
                skipped += 1
                continue

            result.trace_id = first.trace_id
            result.rep_index = first.rep_index
            result.dataset = first.dataset
            result.model = first.model
            result.step_id = first.step_id
            result.step_text = first.step_text
            result.is_correct = first.is_correct

            aggregated.append(result)

        aggregated.sort(key=lambda x: (x.trace_id, x.rep_index, x.model, x.step_id))

        out_path = out_dir / f"{ds}.jsonl"
        with open(out_path, "w") as f:
            for step in aggregated:
                f.write(json.dumps(step.to_dict(), ensure_ascii=False) + "\n")

        n_faith = sum(1 for s in aggregated if s.faithfulness == "faithful")
        n_unfaith = sum(1 for s in aggregated if s.faithfulness == "unfaithful")
        n_unanimous = sum(1 for s in aggregated if s.is_unanimous_faithfulness)
        print(f"  Aggregated: {len(aggregated):,} steps ({skipped} skipped)")
        print(f"  Faithful: {n_faith:,} | Unfaithful: {n_unfaith:,}")
        print(f"  Unanimous faithfulness: {n_unanimous:,} ({100*n_unanimous/max(len(aggregated),1):.1f}%)")
        print(f"  Output: {out_path}")

        all_results[ds] = aggregated

    return all_results
