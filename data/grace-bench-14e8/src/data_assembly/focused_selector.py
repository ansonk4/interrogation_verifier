
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_V3_DIR = _PROJECT_ROOT / "resources" / "results" / "data_assembly_v3"

DATASETS = ["logiqa", "musique", "reclor", "wiki2multihop"]
LOGIC_DATASETS = {"logiqa", "reclor"}
EVIDENCE_DATASETS = {"musique", "wiki2multihop"}

ALL_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Ministral-3-14B-Instruct-2512",
    "Qwen2.5-7B-Instruct",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3.5-27B-FP8",
    "Qwen3.5-35B-A3B-FP8",
    "gemma-4-31B-it-AWQ",
    "gpt-4o-mini",
    "gemini-3-flash-preview",
]

LOGIC_CATEGORIES = [
    "reversed_reasoning", "wrong_argument_reading",
    "rule_violation", "overreaching_claim",
]
EVIDENCE_CATEGORIES = [
    "groundedness_violation", "contradiction",
    "confusion", "evidence_neglect",
]
ALL_CATEGORIES = LOGIC_CATEGORIES + EVIDENCE_CATEGORIES

TARGET_PER_CATEGORY = 500
TARGET_PER_MODEL_PER_CATEGORY = 50
FAITHFUL_RATIO = 1.0
MAX_FAITHFUL_PER_MODEL = 400
MAX_FAITHFUL_PER_DATASET = 1600

SILVER_MIN_CAT_AGREEMENT = 0.85
SILVER_RARE_CAT_AGREEMENT = 0.75

RARE_CATEGORIES = {"rule_violation", "reversed_reasoning"}

MIN_STEPS = 2
MAX_STEPS = 12

def _trace_key(r: dict) -> str:
    return f"{r['trace_id']}|{r['rep_index']}|{r['model']}"

def _length_ok(r: dict) -> bool:
    return MIN_STEPS <= r.get("num_steps", 0) <= MAX_STEPS

def _has_errors(r: dict) -> bool:
    return r.get("num_unfaithful_steps", 0) > 0

def _get_track(dataset: str) -> str:
    if dataset in LOGIC_DATASETS:
        return "logic"
    return "evidence"

def _get_valid_categories(dataset: str) -> list[str]:
    if dataset in LOGIC_DATASETS:
        return LOGIC_CATEGORIES
    return EVIDENCE_CATEGORIES

def _normalize_model(model: str) -> str:
    return model.lower().replace(" ", "-")

def _model_match(r: dict, target_model: str) -> bool:
    return _normalize_model(r.get("model", "")) == _normalize_model(target_model)

def _load_trace_scores(v3_dir: Path) -> dict[str, list[dict]]:
    scores_dir = v3_dir / "trace_scores"
    by_dataset: dict[str, list[dict]] = {}
    for ds in DATASETS:
        path = scores_dir / f"{ds}.jsonl"
        if not path.exists():
            print(f"  [WARN] Missing: {path}")
            continue
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        by_dataset[ds] = records
    return by_dataset

def _write_split(records: list[dict], out_dir: Path, split_subdir: str, dataset: str) -> Path:
    d = out_dir / split_subdir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{dataset}.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path

def _build_category_pools(
    all_scores: dict[str, list[dict]],
    test_keys: set[str],
) -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {cat: [] for cat in ALL_CATEGORIES}

    for ds in DATASETS:
        valid_cats = set(_get_valid_categories(ds))
        for r in all_scores.get(ds, []):
            if _trace_key(r) in test_keys or not _length_ok(r):
                continue
            if not _has_errors(r):
                continue

            tier = r.get("trace_tier", "")
            cat_agr = r.get("category_agreement_on_errors")

            if tier == "gold":
                quality_score = 2.0 + (cat_agr or 1.0)
            elif tier in ("silver", "mixed"):
                if cat_agr is None:
                    continue
                trace_cats = set(r.get("error_categories_present", []))
                is_rare = bool(trace_cats & RARE_CATEGORIES)
                min_agr = SILVER_RARE_CAT_AGREEMENT if is_rare else SILVER_MIN_CAT_AGREEMENT
                if cat_agr < min_agr:
                    continue
                quality_score = 1.0 + cat_agr
            else:
                continue

            r_with_score = {**r, "_quality_score": quality_score}
            for cat in r.get("error_categories_present", []):
                if cat in valid_cats:
                    pools[cat].append(r_with_score)

    for cat in pools:
        pools[cat].sort(key=lambda r: -r["_quality_score"])

    return pools

def _select_unfaithful(
    category_pools: dict[str, list[dict]],
    seed: int = 42,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    rng = random.Random(seed)
    selected_keys: set[str] = set()
    selected_traces: list[dict] = []

    stats: dict[str, dict[str, int]] = {}

    for cat in ALL_CATEGORIES:
        pool = category_pools.get(cat, [])
        rng.shuffle(pool)
        pool.sort(key=lambda r: -r["_quality_score"])

        by_model: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            by_model[r["model"]].append(r)

        model_quotas: dict[str, int] = {m: TARGET_PER_MODEL_PER_CATEGORY for m in ALL_MODELS}
        model_selected: dict[str, list[dict]] = {m: [] for m in ALL_MODELS}

        remaining_target = TARGET_PER_CATEGORY
        for model in ALL_MODELS:
            model_pool = by_model.get(model, [])
            quota = model_quotas[model]
            taken = 0
            for r in model_pool:
                if taken >= quota:
                    break
                key = _trace_key(r)
                if key not in selected_keys:
                    selected_keys.add(key)
                    clean = {k: v for k, v in r.items() if not k.startswith("_")}
                    selected_traces.append(clean)
                taken += 1
            model_selected[model] = model_pool[:taken]
            remaining_target -= taken

        if remaining_target > 0:
            models_with_supply = [
                (m, by_model.get(m, []))
                for m in ALL_MODELS
                if len(by_model.get(m, [])) > len(model_selected.get(m, []))
            ]
            models_with_supply.sort(key=lambda x: -len(x[1]))

            for model, model_pool in models_with_supply:
                if remaining_target <= 0:
                    break
                already_taken = len(model_selected.get(model, []))
                for r in model_pool[already_taken:]:
                    if remaining_target <= 0:
                        break
                    key = _trace_key(r)
                    if key not in selected_keys:
                        selected_keys.add(key)
                        clean = {k: v for k, v in r.items() if not k.startswith("_")}
                        selected_traces.append(clean)
                    remaining_target -= 1

        cat_model_counts: dict[str, int] = {}
        for model in ALL_MODELS:
            cat_model_counts[model] = len(model_selected.get(model, []))
        stats[cat] = cat_model_counts

    return selected_traces, stats

def _select_faithful(
    all_scores: dict[str, list[dict]],
    test_keys: set[str],
    unfaithful_keys: set[str],
    target_total: int,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed + 1)

    pool: list[dict] = []
    for ds in DATASETS:
        ds_count = 0
        for r in all_scores.get(ds, []):
            key = _trace_key(r)
            if key in test_keys or key in unfaithful_keys:
                continue
            if not _length_ok(r):
                continue
            if _has_errors(r):
                continue
            if r.get("trace_tier") != "gold":
                continue
            if ds_count >= MAX_FAITHFUL_PER_DATASET:
                continue
            pool.append(r)
            ds_count += 1

    rng.shuffle(pool)
    pool.sort(key=lambda r: -r.get("num_steps", 0))

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_model[r["model"]].append(r)

    quota = MAX_FAITHFUL_PER_MODEL
    remaining = target_total
    selected: list[dict] = []

    for model in ALL_MODELS:
        model_pool = by_model.get(model, [])
        take = min(quota, len(model_pool), remaining)
        selected.extend(model_pool[:take])
        remaining -= take
        if remaining <= 0:
            break

    if remaining > 0:
        already_keys = set(_trace_key(r) for r in selected)
        leftover = [r for r in pool if _trace_key(r) not in already_keys]
        for r in leftover:
            if remaining <= 0:
                break
            selected.append(r)
            remaining -= 1

    return selected

def _print_report(
    unfaithful: list[dict],
    faithful: list[dict],
    unfaithful_stats: dict[str, dict[str, int]],
):
    all_traces = unfaithful + faithful
    n_total = len(all_traces)
    n_unf = len(unfaithful)
    n_faith = len(faithful)

    print(f"\n{'='*70}")
    print("  train_focused — Selection Report")
    print(f"{'='*70}")

    print(f"\n  Total traces:    {n_total:>6,}")
    print(f"  Unfaithful:      {n_unf:>6,}  ({100*n_unf/max(n_total,1):.1f}%)")
    print(f"  Faithful:        {n_faith:>6,}  ({100*n_faith/max(n_total,1):.1f}%)")
    print(f"  Ratio:           1:{n_faith/max(n_unf,1):.1f}")

    print(f"\n  {'Category':<35s} {'Traces':>7s} {'Target':>7s} {'Fill%':>6s}")
    print(f"  {'-'*60}")
    for cat in ALL_CATEGORIES:
        count = sum(
            1 for r in unfaithful
            if cat in r.get("error_categories_present", [])
        )
        target = TARGET_PER_CATEGORY
        fill_pct = 100 * count / target if target else 0
        flag = "✓" if fill_pct >= 90 else "△" if fill_pct >= 70 else "✗"
        print(f"  {flag} {cat:<33s} {count:>7,} / {target:>5} ({fill_pct:>5.1f}%)")

    model_counts = Counter(r["model"] for r in all_traces)
    print(f"\n  {'Model':<40s} {'Total':>6s} {'Unf':>5s} {'Faith':>5s}")
    print(f"  {'-'*60}")
    unf_by_model = Counter(r["model"] for r in unfaithful)
    faith_by_model = Counter(r["model"] for r in faithful)
    for model in ALL_MODELS:
        n = model_counts.get(model, 0)
        u = unf_by_model.get(model, 0)
        f_count = faith_by_model.get(model, 0)
        print(f"  {model:<40s} {n:>6,} {u:>5,} {f_count:>5,}")

    ds_counts = Counter(r.get("dataset", "?") for r in all_traces)
    print(f"\n  {'Dataset':<20s} {'Total':>6s}")
    print(f"  {'-'*30}")
    for ds in DATASETS:
        print(f"  {ds:<20s} {ds_counts.get(ds, 0):>6,}")

    n_logic = sum(1 for r in all_traces if r.get("dataset") in LOGIC_DATASETS)
    n_evidence = sum(1 for r in all_traces if r.get("dataset") in EVIDENCE_DATASETS)
    print(f"\n  Track split: Logic {n_logic:,} ({100*n_logic/max(n_total,1):.1f}%)"
          f"  | Evidence {n_evidence:,} ({100*n_evidence/max(n_total,1):.1f}%)")

    tier_counts = Counter(r.get("trace_tier", "?") for r in unfaithful)
    print(f"\n  Unfaithful tier composition (internal only):")
    for tier in ["gold", "silver", "mixed"]:
        c = tier_counts.get(tier, 0)
        print(f"    {tier:<12s}: {c:>5,} ({100*c/max(n_unf,1):.1f}%)")

    print(f"{'='*70}")

def run_focused_selection(
    v3_dir: Path | None = None,
    output_dir: Path | None = None,
    test_keys: set[str] | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, list[dict]]:
    v3 = v3_dir or _V3_DIR
    out = output_dir or v3

    print("=" * 70)
    print("  Focused Train-Set Selection (train_focused)")
    print("=" * 70)

    print("\n1. Loading trace scores...")
    all_scores = _load_trace_scores(v3)
    total = sum(len(v) for v in all_scores.values())
    print(f"   Loaded {total:,} traces across {len(all_scores)} datasets")

    if test_keys is None:
        print("\n2. Loading test keys to exclude...")
        test_keys = set()
        for split_dir in ["grace_test/hard", "grace_test/calibration"]:
            for ds in DATASETS:
                path = v3 / split_dir / f"{ds}.jsonl"
                if path.exists():
                    with open(path) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                r = json.loads(line)
                                test_keys.add(_trace_key(r))
        print(f"   Excluded {len(test_keys):,} test traces")
    else:
        print(f"\n2. Using {len(test_keys):,} provided test keys")

    print("\n3. Building category pools...")
    category_pools = _build_category_pools(all_scores, test_keys)
    for cat in ALL_CATEGORIES:
        n = len(category_pools[cat])
        print(f"   {cat:<35s}: {n:>6,} qualified traces")

    print(f"\n4. Selecting unfaithful traces (target: {TARGET_PER_CATEGORY} × {len(ALL_CATEGORIES)} categories = {TARGET_PER_CATEGORY * len(ALL_CATEGORIES):,})...")
    unfaithful, unf_stats = _select_unfaithful(category_pools, seed)
    unfaithful_keys = set(_trace_key(r) for r in unfaithful)
    print(f"   Selected {len(unfaithful):,} unique unfaithful traces")

    target_faithful = int(len(unfaithful) * FAITHFUL_RATIO)
    print(f"\n5. Selecting faithful traces (target: {target_faithful:,}, ratio 1:{FAITHFUL_RATIO:.0f})...")
    faithful = _select_faithful(all_scores, test_keys, unfaithful_keys, target_faithful, seed)
    print(f"   Selected {len(faithful):,} faithful traces")

    _print_report(unfaithful, faithful, unf_stats)

    all_selected = unfaithful + faithful
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in all_selected:
        r_out = {k: v for k, v in r.items() if not k.startswith("_")}
        r_out["split"] = "train_focused"
        by_dataset[r_out["dataset"]].append(r_out)

    if not dry_run:
        print("\n6. Writing output files...")
        for ds in DATASETS:
            records = by_dataset.get(ds, [])
            if records:
                path = _write_split(records, out, "grace_train/focused", ds)
                print(f"   {ds:<20s}: {len(records):>6,} traces → {path}")
        print("\n✅ train_focused selection complete.")
    else:
        print("\n[DRY RUN] — no files written")

    return dict(by_dataset)

def main():
    parser = argparse.ArgumentParser(
        description="Focused train-set selection: compact, category-balanced, model-diverse."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report without writing files")
    parser.add_argument("--v3-dir", type=str, default=None,
                        help="Override data_assembly_v3 directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    run_focused_selection(
        v3_dir=Path(args.v3_dir) if args.v3_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        seed=args.seed,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
