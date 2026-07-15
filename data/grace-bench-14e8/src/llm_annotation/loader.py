
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.trace_loader import (
    TraceRecord,
    TraceStep,
    load_all_traces,
    MODELS,
    DATASETS,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CRITIQUE_DIR = _PROJECT_ROOT / "resources" / "results" / "trace_analysis" / "open_critique"
_TRACES_DIR = _PROJECT_ROOT / "resources" / "results" / "cot_traces" / "full"

def _load_traces_with_unfaithful_steps(
    critique_dir: Path,
    evaluator_model: str = "qwen3.5-27b-fp8",
) -> set[str]:
    crit_dir = critique_dir / evaluator_model
    if not crit_dir.exists():
        print(f"[WARN] No critique results at {crit_dir} — all traces will be treated equally")
        return set()

    traces_with_errors: set[str] = set()

    for jsonl_path in sorted(crit_dir.glob("*.jsonl")):
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("judgment") == "unfaithful":
                        model = record.get("model", "")
                        dataset = record.get("dataset", "")
                        trace_id = record.get("trace_id", "")
                        rep_index = record.get("rep_index", 0)
                        key = f"{model.lower()}/{dataset}/{trace_id}/{rep_index}"
                        traces_with_errors.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue

    return traces_with_errors

def select_traces_for_annotation(
    all_traces: list[TraceRecord],
    traces_with_errors: set[str],
    sample_clean_per_model: int = 2000,
    seed: int = 42,
) -> list[TraceRecord]:
    error_traces: list[TraceRecord] = []
    clean_traces: list[TraceRecord] = []

    for t in all_traces:
        normalized_key = f"{t.model.lower()}/{t.dataset}/{t.trace_id}/{t.rep_index}"
        if normalized_key in traces_with_errors:
            error_traces.append(t)
        else:
            clean_traces.append(t)

    rng = random.Random(seed)
    clean_by_model: dict[str, list[TraceRecord]] = {}
    for t in clean_traces:
        clean_by_model.setdefault(t.model, []).append(t)

    sampled_clean: list[TraceRecord] = []
    for model, model_traces in sorted(clean_by_model.items()):
        n = min(sample_clean_per_model, len(model_traces))
        sampled_clean.extend(rng.sample(model_traces, n))

    selected = error_traces + sampled_clean

    print(f"  Traces with errors (Phase A): {len(error_traces):,}")
    print(f"  Clean traces sampled:         {len(sampled_clean):,} "
          f"({sample_clean_per_model}/model × {len(clean_by_model)} models)")
    print(f"  Total selected:               {len(selected):,}")

    return selected

def load_traces_for_annotation(
    traces_dir: Path | None = None,
    critique_dir: Path | None = None,
    evaluator_model: str = "qwen3.5-27b-fp8",
    filter_models: list[str] | None = None,
    filter_datasets: list[str] | None = None,
    sample_clean_per_model: int = 2000,
    include_all: bool = False,
    seed: int = 42,
) -> list[TraceRecord]:
    t_dir = traces_dir or None
    crit_dir = critique_dir or _CRITIQUE_DIR

    print("Loading traces from Phase 2a...")
    all_traces = load_all_traces(
        traces_dir=t_dir,
        models=filter_models,
        datasets=filter_datasets,
        valid_only=True,
    )
    total_steps = sum(t.num_steps for t in all_traces)
    print(f"  Total traces: {len(all_traces):,} ({total_steps:,} steps)")

    if include_all:
        print("  Selection: ALL (--include-all)")
        return all_traces

    print("\nLoading Phase A results for selection priority...")
    traces_with_errors = _load_traces_with_unfaithful_steps(
        crit_dir, evaluator_model
    )
    print(f"  Traces with ≥1 unfaithful step: {len(traces_with_errors):,}")

    print("\nSelecting traces for annotation...")
    selected = select_traces_for_annotation(
        all_traces=all_traces,
        traces_with_errors=traces_with_errors,
        sample_clean_per_model=sample_clean_per_model,
        seed=seed,
    )

    selected_steps = sum(t.num_steps for t in selected)
    print(f"  Selected steps: {selected_steps:,}")

    return selected
