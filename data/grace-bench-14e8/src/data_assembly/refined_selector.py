
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BASE_DIR = _PROJECT_ROOT / "resources" / "results" / "data_assembly"
_OUTPUT_DIR = _PROJECT_ROOT / "resources" / "results" / "data_assembly_v3"

DATASETS = ["logiqa", "musique", "reclor", "wiki2multihop"]
LOGIC_DATASETS = {"logiqa", "reclor"}
EVIDENCE_DATASETS = {"musique", "wiki2multihop"}

MODEL_FAMILIES = {
    "Llama-3.1-8B-Instruct": "llama",
    "Ministral-3-14B-Instruct-2512": "mistral",
    "Qwen2.5-7B-Instruct": "qwen",
    "Qwen3-8B": "qwen",
    "Qwen3-14B": "qwen",
    "Qwen3.5-27B-FP8": "qwen",
    "Qwen3.5-35B-A3B-FP8": "qwen",
    "gemma-4-31B-it-AWQ": "gemma",
    "gpt-4o-mini": "openai",
    "gemini-3-flash-preview": "google",
}

def _get_family(model: str) -> str:
    if model in MODEL_FAMILIES:
        return MODEL_FAMILIES[model]
    for k, v in MODEL_FAMILIES.items():
        if k.lower() == model.lower():
            return v
    return "unknown"

TEST_HARD_TARGETS = {
    "logiqa": 300,
    "reclor": 300,
    "musique": 350,
    "wiki2multihop": 250,
}

TEST_CAL_ERROR_TARGETS = {
    "logiqa": 90,
    "reclor": 90,
    "musique": 140,
    "wiki2multihop": 170,
}
TEST_CAL_CLEAN_TARGETS = {
    "logiqa": 40,
    "reclor": 40,
    "musique": 60,
    "wiki2multihop": 100,
}

CATEGORY_MAX_FRAC = 0.28
CATEGORY_MIN_FRAC = 0.03

MODEL_MIN_FRAC_OF_FAMILY = 0.12

MIN_STEPS = 2
MAX_STEPS = 12

TRAIN_ALL_FAITHFUL_CAP_PER_DATASET = 3000
TRAIN_SILVER_MIN_CAT_AGREEMENT = 0.67
TRAIN_FAMILY_CAP_PER_TIER = 5000

def _load_trace_scores(base_dir: Path) -> dict[str, list[dict]]:
    scores_dir = base_dir / "trace_scores"
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

def _length_ok(r: dict) -> bool:
    return MIN_STEPS <= r.get("num_steps", 0) <= MAX_STEPS

def _has_errors(r: dict) -> bool:
    return r.get("num_unfaithful_steps", 0) > 0

def _family_balanced_sample(
    pool: list[dict],
    target: int,
    rng: random.Random,
    sort_key=None,
    model_floor: bool = False,
) -> list[dict]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_family[_get_family(r["model"])].append(r)

    for fam in by_family:
        rng.shuffle(by_family[fam])
        if sort_key:
            by_family[fam].sort(key=sort_key)

    families = sorted(by_family.keys())
    remaining = target
    allocations: dict[str, int] = {}

    unallocated = list(families)
    while remaining > 0 and unallocated:
        per_fam = remaining // len(unallocated)
        extra = remaining % len(unallocated)
        new_unallocated = []
        for i, fam in enumerate(unallocated):
            fam_target = per_fam + (1 if i < extra else 0)
            available = len(by_family[fam]) - allocations.get(fam, 0)
            alloc = min(fam_target, available)
            allocations[fam] = allocations.get(fam, 0) + alloc
            remaining -= alloc
            if alloc < fam_target:
                pass
            else:
                new_unallocated.append(fam)
        if not new_unallocated or remaining == 0:
            break
        unallocated = new_unallocated

    result = []
    for fam in families:
        n = allocations.get(fam, 0)
        fam_traces = by_family[fam]

        if not model_floor or n == 0:
            result.extend(fam_traces[:n])
            continue

        by_model: dict[str, list[dict]] = defaultdict(list)
        for r in fam_traces:
            by_model[r["model"]].append(r)

        models = sorted(by_model.keys())
        n_models = len(models)
        model_min = max(1, int(n * MODEL_MIN_FRAC_OF_FAMILY))

        model_selected: dict[str, list[dict]] = {}
        total_taken = 0
        for m in models:
            take = min(model_min, len(by_model[m]))
            model_selected[m] = by_model[m][:take]
            total_taken += take

        already = set(id(r) for m_sel in model_selected.values() for r in m_sel)
        leftover = [r for r in fam_traces if id(r) not in already]
        if sort_key:
            leftover.sort(key=sort_key)
        fill = n - total_taken
        if fill > 0:
            for r in leftover[:fill]:
                model_selected.setdefault(r["model"], []).append(r)

        for m in models:
            result.extend(model_selected.get(m, []))

    return result

def _get_category_distribution(traces: list[dict]) -> Counter:
    cats: Counter = Counter()
    for r in traces:
        for cat in r.get("error_categories_present", []):
            cats[cat] += 1
    return cats

def _check_category_caps(selected: list[dict]) -> bool:
    cats = _get_category_distribution(selected)
    total = sum(cats.values())
    if total == 0:
        return True
    for cat, count in cats.items():
        frac = count / total
        if frac > CATEGORY_MAX_FRAC:
            return False
    return True

def _model_balanced_sample(
    pool: list[dict],
    target: int,
    rng: random.Random,
    sort_key=None,
) -> list[dict]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_model[r["model"]].append(r)

    for model in by_model:
        rng.shuffle(by_model[model])
        if sort_key:
            by_model[model].sort(key=sort_key)

    models = sorted(by_model.keys())
    remaining = target
    allocations: dict[str, int] = {}

    unallocated = list(models)
    while remaining > 0 and unallocated:
        per_model = remaining // len(unallocated)
        extra = remaining % len(unallocated)
        new_unallocated = []
        for i, model in enumerate(unallocated):
            model_target = per_model + (1 if i < extra else 0)
            available = len(by_model[model]) - allocations.get(model, 0)
            alloc = min(model_target, available)
            allocations[model] = allocations.get(model, 0) + alloc
            remaining -= alloc
            if alloc < model_target:
                pass
            else:
                new_unallocated.append(model)
        if not new_unallocated or remaining == 0:
            break
        unallocated = new_unallocated

    result = []
    for model in models:
        n = allocations.get(model, 0)
        result.extend(by_model[model][:n])
    return result

def _select_test_hard(
    all_scores: dict[str, list[dict]],
    seed: int = 42,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    rng = random.Random(seed)
    selected: dict[str, list[dict]] = {}
    rejected: dict[str, list[dict]] = {}

    for ds in DATASETS:
        traces = all_scores.get(ds, [])
        target = TEST_HARD_TARGETS.get(ds, 0)

        qualified = [r for r in traces if _has_errors(r) and _length_ok(r)]

        best_per_qm: dict[str, dict] = {}
        for r in qualified:
            qm_key = f"{r['trace_id']}|{r['model']}"
            existing = best_per_qm.get(qm_key)
            if existing is None or r.get("composite_difficulty", 0) > existing.get("composite_difficulty", 0):
                best_per_qm[qm_key] = r
        qualified = list(best_per_qm.values())

        rng.shuffle(qualified)
        qualified.sort(key=lambda r: -r.get("composite_difficulty", 0.0))

        take = _model_balanced_sample(
            qualified, target, rng,
            sort_key=lambda r: -r.get("composite_difficulty", 0.0),
        )

        take_ids = set(id(r) for r in take)
        rej = [r for r in traces if id(r) not in take_ids]

        selected[ds] = take
        rejected[ds] = rej

    TRACKS = {
        "logic": sorted(LOGIC_DATASETS & set(DATASETS)),
        "evidence": sorted(EVIDENCE_DATASETS & set(DATASETS)),
    }

    MAX_SWAP_ROUNDS = 5
    for track_name, track_datasets in TRACKS.items():
        for swap_round in range(MAX_SWAP_ROUNDS):
            track_selected = [
                r for ds in track_datasets for r in selected.get(ds, [])
            ]
            cats = _get_category_distribution(track_selected)
            total_cats = sum(cats.values())
            if total_cats == 0:
                break

            over_cats = {
                cat for cat, cnt in cats.items()
                if cnt / total_cats > CATEGORY_MAX_FRAC
            }
            if not over_cats:
                break

            swaps_made = 0
            for ds in track_datasets:
                take = selected[ds]
                reserve = rejected[ds]

                selected_qm_keys = set(
                    f"{r['trace_id']}|{r['model']}" for r in take
                )

                reserve_by_model: dict[str, list[dict]] = defaultdict(list)
                for r in reserve:
                    if _has_errors(r) and _length_ok(r):
                        qm_key = f"{r['trace_id']}|{r['model']}"
                        if qm_key not in selected_qm_keys:
                            reserve_by_model[r["model"]].append(r)
                for model in reserve_by_model:
                    reserve_by_model[model].sort(
                        key=lambda r: -r.get("composite_difficulty", 0.0)
                    )

                new_take = []
                for r in take:
                    r_cats = set(r.get("error_categories_present", []))
                    if r_cats and r_cats.issubset(over_cats):
                        model = r["model"]
                        replacement = None
                        for j, alt in enumerate(reserve_by_model.get(model, [])):
                            alt_cats = set(alt.get("error_categories_present", []))
                            if alt_cats and not alt_cats.issubset(over_cats):
                                replacement = alt
                                reserve_by_model[model].pop(j)
                                break
                        if replacement:
                            new_take.append(replacement)
                            swaps_made += 1
                        else:
                            new_take.append(r)
                    else:
                        new_take.append(r)

                selected[ds] = new_take
                new_take_ids = set(id(r) for r in new_take)
                rejected[ds] = [r for r in all_scores.get(ds, []) if id(r) not in new_take_ids]

            if swaps_made > 0:
                print(f"   [{track_name}] Swap round {swap_round + 1}: {swaps_made} traces swapped")
            if swaps_made == 0:
                break

    total_deduped = 0
    for ds in DATASETS:
        take = selected[ds]
        reserve = rejected[ds]

        seen_qm: dict[str, dict] = {}
        deduped = []
        for r in take:
            qm_key = f"{r['trace_id']}|{r['model']}"
            if qm_key in seen_qm:
                existing = seen_qm[qm_key]
                if r.get("composite_difficulty", 0) > existing.get("composite_difficulty", 0):
                    deduped.remove(existing)
                    deduped.append(r)
                    seen_qm[qm_key] = r
                    reserve.append(existing)
                else:
                    reserve.append(r)
                total_deduped += 1
            else:
                seen_qm[qm_key] = r
                deduped.append(r)

        if len(deduped) < len(take):
            deficit = len(take) - len(deduped)
            fill_candidates = [
                r for r in reserve
                if _has_errors(r) and _length_ok(r)
                and f"{r['trace_id']}|{r['model']}" not in seen_qm
            ]
            fill_candidates.sort(key=lambda r: -r.get("composite_difficulty", 0.0))

            model_counts = Counter(r["model"] for r in deduped)
            target_per_model = len(take) // len(MODEL_FAMILIES)
            for r in fill_candidates:
                if deficit <= 0:
                    break
                if model_counts.get(r["model"], 0) < target_per_model:
                    qm_key = f"{r['trace_id']}|{r['model']}"
                    if qm_key not in seen_qm:
                        deduped.append(r)
                        seen_qm[qm_key] = r
                        model_counts[r["model"]] += 1
                        deficit -= 1

            if deficit > 0:
                for r in fill_candidates:
                    if deficit <= 0:
                        break
                    qm_key = f"{r['trace_id']}|{r['model']}"
                    if qm_key not in seen_qm:
                        deduped.append(r)
                        seen_qm[qm_key] = r
                        deficit -= 1

        selected[ds] = deduped
        rejected[ds] = [r for r in all_scores.get(ds, []) if id(r) not in set(id(x) for x in deduped)]

    if total_deduped > 0:
        print(f"   Deduped {total_deduped} rep_index duplicates across all datasets")

    all_selected = [r for ds_traces in selected.values() for r in ds_traces]

    for track_name, track_datasets in TRACKS.items():
        track_traces = [r for ds in track_datasets for r in selected.get(ds, [])]
        cats = _get_category_distribution(track_traces)
        total_cats = sum(cats.values())
        if total_cats > 0:
            print(f"   Category distribution [{track_name} track] ({len(track_traces)} traces):")
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                frac = count / total_cats
                flag = "⚠️" if frac > CATEGORY_MAX_FRAC or frac < CATEGORY_MIN_FRAC else "✓"
                print(f"   {flag} {cat:30s}: {count:>5} ({frac*100:.1f}%)")

    cats = _get_category_distribution(all_selected)
    total_cats = sum(cats.values())
    if total_cats > 0:
        print(f"   Category distribution [overall] ({len(all_selected)} traces):")
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            frac = count / total_cats
            print(f"     {cat:30s}: {count:>5} ({frac*100:.1f}%)")

    model_counts = Counter(r["model"] for r in all_selected)
    print("   Model distribution in test_hard:")
    for model, count in model_counts.most_common():
        fam = _get_family(model)
        print(f"     {model:35s} [{fam:8s}]: {count:>4}")

    return selected, rejected

def _select_test_calibration(
    gold_pool: dict[str, list[dict]],
    seed: int = 42,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    selected: dict[str, list[dict]] = {}

    for ds in DATASETS:
        traces = gold_pool.get(ds, [])
        n_error = TEST_CAL_ERROR_TARGETS.get(ds, 0)
        n_clean = TEST_CAL_CLEAN_TARGETS.get(ds, 0)

        valid = [r for r in traces if _length_ok(r)]
        error_pool = [r for r in valid if _has_errors(r)]
        clean_pool = [r for r in valid if not _has_errors(r)]

        error_take = _family_balanced_sample(
            error_pool, n_error, rng,
            sort_key=lambda r: (
                -r.get("num_unfaithful_steps", 0) / max(r.get("num_steps", 1), 1),
                -r.get("num_steps", 0),
            ),
        )
        clean_take = _family_balanced_sample(
            clean_pool, n_clean, rng,
            sort_key=lambda r: -r.get("num_steps", 0),
        )

        selected[ds] = error_take + clean_take

    return selected

def _build_train_curated(
    all_scores: dict[str, list[dict]],
    test_keys: set[str],
    seed: int = 42,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    rng = random.Random(seed)
    train_gold: dict[str, list[dict]] = {}
    train_silver: dict[str, list[dict]] = {}

    for ds in DATASETS:
        traces = all_scores.get(ds, [])

        available = [
            r for r in traces
            if _trace_key(r) not in test_keys and _length_ok(r)
        ]

        gold = [r for r in available if r.get("trace_tier") == "gold"]
        silver = [r for r in available if r.get("trace_tier") in ("silver", "mixed")]

        gold_with_errors = [r for r in gold if _has_errors(r)]
        gold_all_faithful = [r for r in gold if not _has_errors(r)]
        rng.shuffle(gold_all_faithful)
        gold_all_faithful_capped = gold_all_faithful[:TRAIN_ALL_FAITHFUL_CAP_PER_DATASET]
        gold_filtered = gold_with_errors + gold_all_faithful_capped

        silver_filtered = []
        for r in silver:
            cat_agr = r.get("category_agreement_on_errors")
            if r.get("num_unfaithful_steps", 0) == 0:
                silver_filtered.append(r)
            elif cat_agr is not None and cat_agr >= TRAIN_SILVER_MIN_CAT_AGREEMENT:
                silver_filtered.append(r)

        gold_filtered = _cap_by_family(gold_filtered, TRAIN_FAMILY_CAP_PER_TIER, rng)
        silver_filtered = _cap_by_family(silver_filtered, TRAIN_FAMILY_CAP_PER_TIER, rng)

        train_gold[ds] = gold_filtered
        train_silver[ds] = silver_filtered

    return train_gold, train_silver

def _build_train_full(
    all_scores: dict[str, list[dict]],
    test_keys: set[str],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    train_gold: dict[str, list[dict]] = {}
    train_silver: dict[str, list[dict]] = {}

    for ds in DATASETS:
        traces = all_scores.get(ds, [])
        available = [
            r for r in traces
            if _trace_key(r) not in test_keys and _length_ok(r)
        ]
        train_gold[ds] = [r for r in available if r.get("trace_tier") == "gold"]
        train_silver[ds] = [r for r in available if r.get("trace_tier") in ("silver", "mixed")]

    return train_gold, train_silver

def _cap_by_family(traces: list[dict], cap: int, rng: random.Random) -> list[dict]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in traces:
        by_family[_get_family(r["model"])].append(r)

    result = []
    for fam in sorted(by_family.keys()):
        pool = by_family[fam]
        if len(pool) > cap:
            rng.shuffle(pool)
            pool = pool[:cap]
        result.extend(pool)
    return result

def _trace_key(r: dict) -> str:
    return f"{r['trace_id']}|{r['rep_index']}|{r['model']}"

def run_refined_selection(
    base_dir: Path | None = None,
    output_dir: Path | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> dict:
    base = base_dir or _BASE_DIR
    out = output_dir or _OUTPUT_DIR

    print("=" * 70)
    print("Phase 3.5 v3: Refined Trace Selection")
    print("=" * 70)

    print("\n1. Loading trace scores...")
    all_scores = _load_trace_scores(base)
    for ds in DATASETS:
        n = len(all_scores.get(ds, []))
        print(f"   {ds:15s}: {n:>7,} traces")
    total = sum(len(v) for v in all_scores.values())
    print(f"   {'TOTAL':15s}: {total:>7,} traces")

    print(f"\n2. Selecting test_hard (target: {sum(TEST_HARD_TARGETS.values())} traces)...")
    print("   Category distribution in selected set:")
    hard_selected, hard_rejected = _select_test_hard(all_scores, seed)

    for ds in DATASETS:
        n_sel = len(hard_selected.get(ds, []))
        target = TEST_HARD_TARGETS.get(ds, 0)
        families = Counter(_get_family(r["model"]) for r in hard_selected.get(ds, []))
        fam_str = ", ".join(f"{f}:{c}" for f, c in sorted(families.items()))
        print(f"   {ds:15s}: {n_sel:>5}/{target} selected | families: {fam_str}")

    print(f"\n3. Selecting test_calibration (target: {sum(TEST_CAL_ERROR_TARGETS.values()) + sum(TEST_CAL_CLEAN_TARGETS.values())} traces)...")
    hard_keys: set[str] = set()
    for ds in DATASETS:
        for r in hard_selected.get(ds, []):
            hard_keys.add(_trace_key(r))

    gold_pool: dict[str, list[dict]] = {}
    for ds in DATASETS:
        gold_pool[ds] = [
            r for r in all_scores.get(ds, [])
            if r.get("trace_tier") == "gold" and _trace_key(r) not in hard_keys
        ]

    cal_selected = _select_test_calibration(gold_pool, seed)

    for ds in DATASETS:
        n_sel = len(cal_selected.get(ds, []))
        n_err = sum(1 for r in cal_selected.get(ds, []) if _has_errors(r))
        n_clean = n_sel - n_err
        print(f"   {ds:15s}: {n_err:>4} error + {n_clean:>4} clean = {n_sel:>4}")

    test_keys: set[str] = set()
    for ds in DATASETS:
        for r in hard_selected.get(ds, []):
            test_keys.add(_trace_key(r))
        for r in cal_selected.get(ds, []):
            test_keys.add(_trace_key(r))

    print("\n4. Building train sets...")
    print("   Curated (quality-filtered):")
    train_gold_cur, train_silver_cur = _build_train_curated(all_scores, test_keys, seed)
    for ds in DATASETS:
        ng = len(train_gold_cur.get(ds, []))
        ns = len(train_silver_cur.get(ds, []))
        print(f"   {ds:15s}: gold {ng:>6,} | silver {ns:>6,}")

    print("   Full (length-filtered only):")
    train_gold_full, train_silver_full = _build_train_full(all_scores, test_keys)
    for ds in DATASETS:
        ng = len(train_gold_full.get(ds, []))
        ns = len(train_silver_full.get(ds, []))
        print(f"   {ds:15s}: gold {ng:>6,} | silver {ns:>6,}")

    all_assigned_keys = set(test_keys)
    for ds in DATASETS:
        for r in train_gold_full.get(ds, []):
            all_assigned_keys.add(_trace_key(r))
        for r in train_silver_full.get(ds, []):
            all_assigned_keys.add(_trace_key(r))

    reserve: dict[str, list[dict]] = {}
    for ds in DATASETS:
        reserve[ds] = [
            r for r in all_scores.get(ds, [])
            if _trace_key(r) not in all_assigned_keys
        ]

    _print_summary(hard_selected, cal_selected, train_gold_cur, train_silver_cur, reserve)

    if not dry_run:
        print("\n6. Writing output files...")
        output_map = {
            "grace_test/hard": hard_selected,
            "grace_test/calibration": cal_selected,
            "grace_train/curated/gold": train_gold_cur,
            "grace_train/curated/silver": train_silver_cur,
            "grace_train/full/gold": train_gold_full,
            "grace_train/full/silver": train_silver_full,
            "reserve": reserve,
        }
        for split_subdir, data in output_map.items():
            for ds in DATASETS:
                records = data.get(ds, [])
                if records:
                    for r in records:
                        r["split"] = split_subdir.replace("/", "_")
                    path = _write_split(records, out, split_subdir, ds)
                    print(f"   {split_subdir:35s} / {ds:15s}: {len(records):>6,} → {path}")
    else:
        print("\n[DRY RUN] — no files written")

    return {
        "test_hard": hard_selected,
        "test_calibration": cal_selected,
        "train_gold_curated": train_gold_cur,
        "train_silver_curated": train_silver_cur,
        "train_gold_full": train_gold_full,
        "train_silver_full": train_silver_full,
        "reserve": reserve,
    }

def _print_summary(
    hard: dict[str, list[dict]],
    cal: dict[str, list[dict]],
    train_gold: dict[str, list[dict]],
    train_silver: dict[str, list[dict]],
    reserve: dict[str, list[dict]],
):
    print(f"\n{'=' * 70}")
    print("v3 Refined Selection Summary")
    print(f"{'=' * 70}")

    splits = {
        "train_gold_curated": train_gold,
        "train_silver_curated": train_silver,
        "test_hard": hard,
        "test_calibration": cal,
        "reserve": reserve,
    }

    header = f"  {'Split':<25s} {'Traces':>8s} {'Steps':>8s} {'Unfaith%':>9s} {'Logic':>7s} {'Evid':>7s} {'Families':>8s}"
    print(f"\n{header}")
    print("  " + "-" * 85)

    for split_name, data in splits.items():
        all_records = [r for ds in DATASETS for r in data.get(ds, [])]
        n_traces = len(all_records)
        n_steps = sum(r.get("num_steps", 0) for r in all_records)
        n_unf = sum(r.get("num_unfaithful_steps", 0) for r in all_records)
        unf_pct = f"{100 * n_unf / n_steps:.1f}%" if n_steps else "—"
        n_logic = sum(1 for r in all_records if r.get("dataset") in LOGIC_DATASETS)
        n_evid = sum(1 for r in all_records if r.get("dataset") in EVIDENCE_DATASETS)
        n_families = len(set(_get_family(r["model"]) for r in all_records)) if all_records else 0
        print(f"  {split_name:<25s} {n_traces:>8,} {n_steps:>8,} {unf_pct:>9s} {n_logic:>7,} {n_evid:>7,} {n_families:>8}")

    for split_name in ("test_hard", "test_calibration"):
        data = splits[split_name]
        all_records = [r for ds in DATASETS for r in data.get(ds, [])]
        fam_counts = Counter(_get_family(r["model"]) for r in all_records)
        print(f"\n  Family distribution — {split_name}:")
        for fam, count in sorted(fam_counts.items()):
            pct = 100 * count / len(all_records) if all_records else 0
            bar = "█" * int(pct / 2)
            print(f"    {fam:15s}: {count:>5} ({pct:>5.1f}%) {bar}")

def main():
    parser = argparse.ArgumentParser(description="Phase 3.5 v3: Refined Trace Selection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_refined_selection(
        base_dir=Path(args.base_dir) if args.base_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        seed=args.seed,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
