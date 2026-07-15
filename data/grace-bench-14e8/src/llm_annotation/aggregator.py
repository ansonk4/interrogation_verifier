
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANNOTATIONS_DIR = _PROJECT_ROOT / "resources" / "results" / "annotations"

DATASETS = ["logiqa", "musique", "reclor", "wiki2multihop"]

@dataclass
class AggregatedAnnotation:

    trace_id: str
    rep_index: int
    dataset: str
    model: str
    step_id: int
    n_judges: int
    faithfulness: str
    faithfulness_agreement: float
    error_category: str | None
    category_agreement: float
    is_unanimous: bool
    error_sub_flag: str | None
    severity: str | None
    explanation: str | None
    quality_tier: str
    per_judge: list[dict]
    judge_models: list[str]

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "rep_index": self.rep_index,
            "dataset": self.dataset,
            "model": self.model,
            "step_id": self.step_id,
            "faithfulness": self.faithfulness,
            "error_category": self.error_category,
            "error_sub_flag": self.error_sub_flag,
            "severity": self.severity,
            "explanation": self.explanation,
            "quality_tier": self.quality_tier,
            "annotation_meta": {
                "n_judges": self.n_judges,
                "faithfulness_agreement": self.faithfulness_agreement,
                "category_agreement": self.category_agreement,
                "is_unanimous": self.is_unanimous,
                "judge_models": self.judge_models,
                "per_judge": self.per_judge,
            },
        }

def _majority_vote(values: list[str | None]) -> tuple[str | None, float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None, 0.0

    counter = Counter(valid)
    winner, count = counter.most_common(1)[0]
    agreement = count / len(valid)
    return winner, agreement

def _pick_best_explanation(
    judges: list[dict],
    consensus_category: str | None,
) -> str | None:
    if not consensus_category:
        return None

    agreeing = [
        j for j in judges
        if j.get("error_category") == consensus_category
        and j.get("explanation")
    ]
    if not agreeing:
        return None

    agreeing.sort(key=lambda j: j.get("confidence", 0) or 0, reverse=True)
    return agreeing[0]["explanation"]

def _assign_quality_tier(
    n_judges: int,
    faithfulness_agreement: float,
    category_agreement: float,
    is_unanimous: bool,
    min_judges: int = 2,
    min_faithfulness_agreement: float = 0.667,
    min_category_agreement: float = 0.667,
) -> str:
    if n_judges < min_judges:
        return "rejected"

    if faithfulness_agreement < min_faithfulness_agreement:
        return "rejected"

    if category_agreement < min_category_agreement:
        return "rejected"

    if is_unanimous:
        return "gold"

    supermajority_threshold = (n_judges - 1) / n_judges
    if category_agreement >= supermajority_threshold:
        return "silver"

    return "bronze"

def _load_judge_annotations(
    judge_dir: Path,
    dataset: str,
) -> list[dict]:
    jsonl_path = judge_dir / f"{dataset}.jsonl"
    if not jsonl_path.exists():
        return []

    records: list[dict] = []
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not record.get("parse_error", False):
                    records.append(record)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARN] {jsonl_path}:{line_num} — skipped: {e}")
    return records

def _step_key(record: dict) -> str:
    return (
        f"{record['trace_id']}|{record.get('rep_index', 0)}"
        f"|{record['model']}|{record['step_id']}"
    )

def aggregate_dataset(
    judge_dirs: list[Path],
    dataset: str,
    min_judges: int = 2,
    min_faithfulness_agreement: float = 0.667,
    min_category_agreement: float = 0.667,
) -> list[AggregatedAnnotation]:
    all_records: list[dict] = []
    for jdir in judge_dirs:
        records = _load_judge_annotations(jdir, dataset)
        all_records.extend(records)

    if not all_records:
        return []

    by_step: dict[str, list[dict]] = {}
    for rec in all_records:
        key = _step_key(rec)
        by_step.setdefault(key, []).append(rec)

    aggregated: list[AggregatedAnnotation] = []
    for step_key, judges in by_step.items():
        n_judges = len(judges)

        faith_values = [j.get("faithfulness") for j in judges]
        faith_winner, faith_agree = _majority_vote(faith_values)

        unfaithful_judges = [
            j for j in judges if j.get("faithfulness") == "unfaithful"
        ]
        cat_values = [j.get("error_category") for j in unfaithful_judges]
        cat_winner, cat_agree = _majority_vote(cat_values)

        valid_cats = [c for c in cat_values if c is not None]
        is_unanimous = (
            len(set(valid_cats)) == 1 and len(valid_cats) == len(unfaithful_judges)
            if unfaithful_judges else False
        )

        if faith_winner in ("faithful", "neutral"):
            cat_winner = None
            cat_agree = 1.0
            is_unanimous = True

        sub_flag = None
        if cat_winner:
            agreeing_subs = [
                j.get("error_sub_flag")
                for j in judges
                if j.get("error_category") == cat_winner
            ]
            sub_flag, _ = _majority_vote(agreeing_subs)

        sev_values = [j.get("severity") for j in unfaithful_judges]
        severity, _ = _majority_vote(sev_values)

        if faith_winner == "unfaithful":
            explanation = _pick_best_explanation(judges, cat_winner)
        else:
            with_explanation = [
                j for j in judges if j.get("explanation")
            ]
            with_explanation.sort(
                key=lambda j: j.get("confidence", 0) or 0, reverse=True
            )
            explanation = with_explanation[0]["explanation"] if with_explanation else None

        tier = _assign_quality_tier(
            n_judges=n_judges,
            faithfulness_agreement=faith_agree,
            category_agreement=cat_agree,
            is_unanimous=is_unanimous,
            min_judges=min_judges,
            min_faithfulness_agreement=min_faithfulness_agreement,
            min_category_agreement=min_category_agreement,
        )

        first = judges[0]
        per_judge = [
            {
                "judge_id": j.get("judge_id"),
                "judge_model": j.get("judge_model"),
                "faithfulness": j.get("faithfulness"),
                "error_category": j.get("error_category"),
                "error_sub_flag": j.get("error_sub_flag"),
                "severity": j.get("severity"),
                "confidence": j.get("confidence"),
            }
            for j in judges
        ]

        aggregated.append(AggregatedAnnotation(
            trace_id=first["trace_id"],
            rep_index=first.get("rep_index", 0),
            dataset=first["dataset"],
            model=first["model"],
            step_id=first["step_id"],
            n_judges=n_judges,
            faithfulness=faith_winner or "unknown",
            faithfulness_agreement=faith_agree,
            error_category=cat_winner,
            category_agreement=cat_agree,
            is_unanimous=is_unanimous,
            error_sub_flag=sub_flag,
            severity=severity,
            explanation=explanation,
            quality_tier=tier,
            per_judge=per_judge,
            judge_models=list(set(
                j.get("judge_model", "unknown") for j in judges
            )),
        ))

    return aggregated

def aggregate_annotations(
    judge_dirs: list[Path],
    output_dir: Path | None = None,
    datasets: list[str] | None = None,
    min_judges: int = 2,
    min_faithfulness_agreement: float = 0.667,
    min_category_agreement: float = 0.667,
) -> dict[str, list[AggregatedAnnotation]]:
    out_dir = output_dir or (_ANNOTATIONS_DIR / "aggregated")
    os.makedirs(out_dir, exist_ok=True)
    target_datasets = datasets or DATASETS

    all_aggregated: dict[str, list[AggregatedAnnotation]] = {}
    total_stats = Counter()

    print(f"Aggregating from {len(judge_dirs)} judge(s):")
    for jd in judge_dirs:
        print(f"  - {jd}")
    print()

    for dataset in target_datasets:
        agg = aggregate_dataset(
            judge_dirs=judge_dirs,
            dataset=dataset,
            min_judges=min_judges,
            min_faithfulness_agreement=min_faithfulness_agreement,
            min_category_agreement=min_category_agreement,
        )

        if not agg:
            print(f"  {dataset}: no annotations found")
            continue

        out_path = out_dir / f"{dataset}.jsonl"
        with open(out_path, "w") as f:
            for a in agg:
                f.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")

        tier_counts = Counter(a.quality_tier for a in agg)
        cat_counts = Counter(a.error_category for a in agg if a.error_category)

        print(f"  {dataset}: {len(agg):,} steps aggregated")
        print(f"    Tiers: gold={tier_counts.get('gold', 0):,} "
              f"silver={tier_counts.get('silver', 0):,} "
              f"bronze={tier_counts.get('bronze', 0):,} "
              f"rejected={tier_counts.get('rejected', 0):,}")
        print(f"    Categories: {dict(cat_counts.most_common())}")

        all_aggregated[dataset] = agg
        for tier, count in tier_counts.items():
            total_stats[f"tier_{tier}"] += count
        total_stats["total"] += len(agg)

    print(f"\n{'━' * 60}")
    print(f"  AGGREGATION SUMMARY")
    print(f"{'━' * 60}")
    print(f"  Total steps       : {total_stats['total']:,}")
    print(f"  Gold (unanimous)  : {total_stats.get('tier_gold', 0):,}")
    print(f"  Silver (supermaj) : {total_stats.get('tier_silver', 0):,}")
    print(f"  Bronze (majority) : {total_stats.get('tier_bronze', 0):,}")
    print(f"  Rejected          : {total_stats.get('tier_rejected', 0):,}")
    accepted = (
        total_stats.get("tier_gold", 0)
        + total_stats.get("tier_silver", 0)
        + total_stats.get("tier_bronze", 0)
    )
    if total_stats["total"] > 0:
        print(f"  Acceptance rate   : {accepted / total_stats['total'] * 100:.1f}%")
    print(f"  Output            : {out_dir}")
    print(f"{'━' * 60}")

    return all_aggregated
