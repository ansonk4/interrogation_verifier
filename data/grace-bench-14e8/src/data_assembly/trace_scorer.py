
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from src.data_assembly.aggregator import AggregatedStep, DATASETS

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

AGGREGATED_DIR = _PROJECT_ROOT / "resources" / "results" / "data_assembly" / "aggregated"
OUTPUT_DIR = _PROJECT_ROOT / "resources" / "results" / "data_assembly" / "trace_scores"

@dataclass
class StepSummary:

    step_id: int
    faithfulness: str | None
    error_category: str | None
    faithfulness_agreement: float
    category_agreement: float | None
    n_judges: int
    is_unanimous_faithfulness: bool
    is_unanimous_category: bool | None
    step_tier: str

@dataclass
class TraceScore:

    trace_id: str
    rep_index: int
    dataset: str
    model: str
    is_correct: bool | None

    num_steps: int
    num_faithful_steps: int
    num_unfaithful_steps: int
    num_neutral_steps: int
    num_ambiguous_steps: int

    trace_tier: str
    min_faithfulness_agreement: float
    mean_faithfulness_agreement: float
    has_disputed_steps: bool

    error_categories_present: list[str]
    category_agreement_on_errors: float | None

    disagreement_score: float

    detection_ambiguity: float
    subtle_error_bonus: float
    composite_difficulty: float

    step_summaries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _classify_step_tier(step: AggregatedStep) -> str:
    if step.faithfulness == "faithful" and step.is_unanimous_faithfulness:
        return "gold_faithful"

    if step.faithfulness == "unfaithful":
        if step.is_unanimous_faithfulness and step.is_unanimous_category:
            return "gold_unfaithful"
        if (step.faithfulness_agreement >= 2 / 3 and
                step.category_agreement is not None and
                step.category_agreement >= 2 / 3):
            return "silver"
        return "disputed"

    if step.faithfulness == "neutral" and step.is_unanimous_faithfulness:
        return "gold_faithful"

    if step.faithfulness_agreement >= 2 / 3:
        return "silver"

    return "disputed"

def _classify_trace_tier(step_tiers: list[str], step_summaries: list[StepSummary]) -> str:
    has_disputed = "disputed" in step_tiers

    if not has_disputed:
        if all(t.startswith("gold") for t in step_tiers):
            return "gold"
        return "silver"

    disputed_are_unfaithful = any(
        s.step_tier == "disputed" and s.faithfulness == "unfaithful"
        for s in step_summaries
    )

    if not disputed_are_unfaithful:
        return "mixed"

    return "disputed"

def score_trace(
    trace_id: str,
    rep_index: int,
    dataset: str,
    model: str,
    steps: list[AggregatedStep],
) -> TraceScore:
    steps = sorted(steps, key=lambda s: s.step_id)

    step_summaries = []
    for s in steps:
        tier = _classify_step_tier(s)
        step_summaries.append(StepSummary(
            step_id=s.step_id,
            faithfulness=s.faithfulness,
            error_category=s.error_category,
            faithfulness_agreement=s.faithfulness_agreement,
            category_agreement=s.category_agreement,
            n_judges=s.n_judges,
            is_unanimous_faithfulness=s.is_unanimous_faithfulness,
            is_unanimous_category=s.is_unanimous_category,
            step_tier=tier,
        ))

    step_tiers = [s.step_tier for s in step_summaries]

    n_faith = sum(1 for s in steps if s.faithfulness == "faithful")
    n_unfaith = sum(1 for s in steps if s.faithfulness == "unfaithful")
    n_neutral = sum(1 for s in steps if s.faithfulness == "neutral")
    n_ambiguous = sum(1 for s in steps if s.faithfulness is None)

    trace_tier = _classify_trace_tier(step_tiers, step_summaries)

    faith_agreements = [s.faithfulness_agreement for s in steps]
    min_fa = min(faith_agreements) if faith_agreements else 0.0
    mean_fa = sum(faith_agreements) / len(faith_agreements) if faith_agreements else 0.0

    error_cats = [s.error_category for s in steps if s.error_category]
    unique_cats = sorted(set(error_cats))

    cat_agreements = [s.category_agreement for s in steps
                      if s.faithfulness == "unfaithful" and s.category_agreement is not None]
    mean_cat_agr = sum(cat_agreements) / len(cat_agreements) if cat_agreements else None

    disagreement_scores = []
    for s in steps:
        if s.faithfulness == "unfaithful" and s.category_agreement is not None:
            disagreement_scores.append(1.0 - s.category_agreement)
    disagreement = max(disagreement_scores) if disagreement_scores else 0.0

    is_correct = steps[0].is_correct if steps else None

    unfaith_faith_agreements = [
        s.faithfulness_agreement for s in steps
        if s.faithfulness == "unfaithful"
    ]
    detection_ambiguity = (
        1.0 - min(unfaith_faith_agreements)
        if unfaith_faith_agreements else 0.0
    )

    subtle_error_bonus = (
        1.0 if is_correct and n_unfaith > 0 else 0.0
    )

    composite_difficulty = (
        0.4 * disagreement
        + 0.4 * detection_ambiguity
        + 0.2 * subtle_error_bonus
    )

    return TraceScore(
        trace_id=trace_id,
        rep_index=rep_index,
        dataset=dataset,
        model=model,
        is_correct=is_correct,
        num_steps=len(steps),
        num_faithful_steps=n_faith,
        num_unfaithful_steps=n_unfaith,
        num_neutral_steps=n_neutral,
        num_ambiguous_steps=n_ambiguous,
        trace_tier=trace_tier,
        min_faithfulness_agreement=min_fa,
        mean_faithfulness_agreement=mean_fa,
        has_disputed_steps="disputed" in step_tiers,
        error_categories_present=unique_cats,
        category_agreement_on_errors=mean_cat_agr,
        disagreement_score=disagreement,
        detection_ambiguity=detection_ambiguity,
        subtle_error_bonus=subtle_error_bonus,
        composite_difficulty=composite_difficulty,
        step_summaries=[asdict(s) for s in step_summaries],
    )

def _load_aggregated_steps(dataset: str, aggregated_dir: Path | None = None) -> list[AggregatedStep]:
    base = aggregated_dir or AGGREGATED_DIR
    path = base / f"{dataset}.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"Aggregated file not found: {path}")

    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            steps.append(AggregatedStep(**{k: d[k] for k in AggregatedStep.__dataclass_fields__ if k in d}))

    return steps

def _group_steps_by_trace(steps: list[AggregatedStep]) -> dict[str, list[AggregatedStep]]:
    groups: dict[str, list[AggregatedStep]] = defaultdict(list)
    for s in steps:
        key = f"{s.trace_id}|{s.rep_index}|{s.model}"
        groups[key].append(s)
    return dict(groups)

def run_scoring(
    datasets: list[str] | None = None,
    aggregated_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, list[TraceScore]]:
    target_datasets = datasets or DATASETS
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[TraceScore]] = {}

    for ds in target_datasets:
        print(f"\n{'='*60}")
        print(f"Scoring traces: {ds}")
        print(f"{'='*60}")

        steps = _load_aggregated_steps(ds, aggregated_dir)
        print(f"  Loaded {len(steps):,} aggregated steps")

        groups = _group_steps_by_trace(steps)
        print(f"  Found {len(groups):,} unique traces")

        scores: list[TraceScore] = []
        tier_counts: Counter = Counter()

        for trace_key, trace_steps in groups.items():
            first = trace_steps[0]
            ts = score_trace(
                trace_id=first.trace_id,
                rep_index=first.rep_index,
                dataset=first.dataset,
                model=first.model,
                steps=trace_steps,
            )
            scores.append(ts)
            tier_counts[ts.trace_tier] += 1

        scores.sort(key=lambda x: (x.trace_id, x.rep_index, x.model))

        out_path = out_dir / f"{ds}.jsonl"
        with open(out_path, "w") as f:
            for ts in scores:
                f.write(json.dumps(ts.to_dict(), ensure_ascii=False) + "\n")

        print(f"  Trace tiers:")
        for tier in ["gold", "silver", "mixed", "disputed"]:
            c = tier_counts.get(tier, 0)
            print(f"    {tier:10s}: {c:>6,} ({100*c/max(len(scores),1):.1f}%)")
        print(f"  Output: {out_path}")

        all_results[ds] = scores

    return all_results
