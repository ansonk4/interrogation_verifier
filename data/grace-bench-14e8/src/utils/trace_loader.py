
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRACES_DIR = _PROJECT_ROOT / "resources" / "results" / "cot_traces" / "full"

MODELS = [
    "llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3.5-27b-fp8",
    "ministral-3-14b-instruct-2512",
    "gemma-4-31b-it-awq",
]

DATASETS = ["logiqa", "musique", "reclor", "wiki2multihop"]

@dataclass
class TraceStep:

    step_id: int
    text: str
    text_without_citations: str
    citations: list[str]

@dataclass
class TraceRecord:

    trace_id: str
    rep_index: int
    dataset: str
    context: str
    question: str
    options: list[str] | None
    gold_answer: str
    model: str
    steps: list[TraceStep]
    final_answer: str
    num_steps: int
    is_valid: bool
    is_correct: bool | None
    raw_text: str

    @property
    def composite_key(self) -> str:
        return f"{self.model}/{self.dataset}/{self.trace_id}/{self.rep_index}"

def _parse_record(record: dict) -> TraceRecord:
    trace = record["trace"]
    steps = [
        TraceStep(
            step_id=s["id"],
            text=s["text"],
            text_without_citations=s.get("text_without_citations", s["text"]),
            citations=s.get("citations", []),
        )
        for s in trace.get("steps", [])
    ]

    return TraceRecord(
        trace_id=record["id"],
        rep_index=record.get("rep_index", 0),
        dataset=record["dataset"],
        context=record["context"],
        question=record["question"],
        options=record.get("options"),
        gold_answer=record["gold_answer"],
        model=record["model"],
        steps=steps,
        final_answer=trace.get("final_answer", ""),
        num_steps=trace.get("num_steps", len(steps)),
        is_valid=trace.get("is_valid", False),
        is_correct=record.get("is_correct"),
        raw_text=trace.get("raw_text", ""),
    )

def load_traces(
    model: str,
    dataset: str,
    traces_dir: Path | None = None,
) -> list[TraceRecord]:
    base = traces_dir or _TRACES_DIR
    jsonl_path = base / model / f"{dataset}.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Trace file not found: {jsonl_path}")

    records: list[TraceRecord] = []
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(_parse_record(record))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARN] {jsonl_path}:{line_num} — skipped: {e}")

    return records

def load_all_traces(
    traces_dir: Path | None = None,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    valid_only: bool = True,
    correct_only: bool = False,
    incorrect_only: bool = False,
) -> list[TraceRecord]:
    target_models = models or MODELS
    target_datasets = datasets or DATASETS
    all_records: list[TraceRecord] = []

    for model in target_models:
        for dataset in target_datasets:
            try:
                records = load_traces(model, dataset, traces_dir)
            except FileNotFoundError:
                print(f"[WARN] No traces for {model}/{dataset} — skipping")
                continue

            for r in records:
                if valid_only and not r.is_valid:
                    continue
                if correct_only and r.is_correct is not True:
                    continue
                if incorrect_only and r.is_correct is not False:
                    continue
                all_records.append(r)

    return all_records
