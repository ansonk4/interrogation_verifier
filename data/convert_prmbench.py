"""Convert paired PRMBench processes to graph-verifier MVP cases."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from graph_verifier.utils.llm import LLMError, complete, parse_json_object


ROOT = Path(__file__).parent
SOURCE = ROOT / "PRMBench_Preview" / "prmbench_preview.jsonl"
OUTPUT = ROOT / "prmbench_cases.jsonl"
DATASET = "hitsmy/PRMBench_Preview"
REVISION = "5cc7683d0ae5797f84d7aeac0607966f277c39e1"
EXTRACTOR_MODEL_CONFIG = "model/openrouter/deepseek/deepseek-v4-flash-high.json"
ERROR_CLASSES = (
    "circular",
    "confidence",
    "counterfactual",
    "deception",
    "domain_inconsistency",
    "missing_condition",
    "redundency",
    "step_contradiction",
)
QUESTION_LIMIT = 100
CANDIDATES_PER_CLASS = 20
EXTRACTION_BATCH_SIZE = 10
TAIL_STEPS = 5


def main() -> int:
    rows = read_source()
    candidates = select_rows(rows)
    answers = extract_answers(candidates)
    selected = select_convertible_rows(candidates, answers)
    cases = build_cases(selected, answers)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"{OUTPUT}: {len(cases)} cases from {len(selected)} distinct questions")
    return 0


def read_source() -> list[tuple[int, dict[str, Any]]]:
    with SOURCE.open(encoding="utf-8") as handle:
        return [(line_number, json.loads(line)) for line_number, line in enumerate(handle, 1) if line.strip()]


def select_rows(rows: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, dict[str, Any]]]:
    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for line_number, row in rows:
        classification = row["classification"]
        if (
            classification in ERROR_CLASSES
            and row["original_question"] == row["modified_question"]
            and row["original_process"]
            and row["modified_process"]
        ):
            buckets[classification].append((line_number, row))

    selected: list[tuple[int, dict[str, Any]]] = []
    seen_questions: set[str] = set()
    for classification in ERROR_CLASSES:
        count = 0
        for item in buckets[classification]:
            question = item[1]["original_question"]
            if question in seen_questions:
                continue
            selected.append(item)
            seen_questions.add(question)
            count += 1
            if count == CANDIDATES_PER_CLASS:
                break
        else:
            raise RuntimeError(f"not enough distinct questions for {classification}")

    expected = CANDIDATES_PER_CLASS * len(ERROR_CLASSES)
    if len(selected) != expected:
        raise RuntimeError(f"selected {len(selected)} candidates, expected {expected}")
    return selected


def extract_answers(
    selected: list[tuple[int, dict[str, Any]]],
) -> dict[str, tuple[str, dict[str, Any]]]:
    answers: dict[str, tuple[str, dict[str, Any]]] = {}
    for start in range(0, len(selected), EXTRACTION_BATCH_SIZE):
        batch = selected[start : start + EXTRACTION_BATCH_SIZE]
        payload = []
        tails: dict[str, str] = {}
        for line_number, row in batch:
            for version, field in (("correct", "original_process"), ("incorrect", "modified_process")):
                key = f"{line_number}:{version}"
                tail = "\n\n".join(row[field][-TAIL_STEPS:])
                tails[key] = tail
                payload.append({"key": key, "response_tail": tail})
        extracted = call_extractor(payload)
        if set(extracted) != set(tails):
            raise RuntimeError("answer extractor returned the wrong case set")
        for key, answer in extracted.items():
            if not answer or answer not in tails[key]:
                continue
            answers[key] = (
                answer,
                {
                    "method": "llm_exact_substring",
                    "model_config": EXTRACTOR_MODEL_CONFIG,
                    "source_window": f"last_{TAIL_STEPS}_steps",
                },
            )
    return answers


def call_extractor(payload: list[dict[str, str]]) -> dict[str, str | None]:
    prompt = """Extract the final answer explicitly claimed by each RESPONSE TAIL.
Do not solve, correct, evaluate, rewrite, or infer an unstated answer.
For each item, copy the shortest exact substring that unambiguously gives its final answer.
Return only JSON in this shape: {"cases": [{"key": "...", "agent_answer": "exact substring or null"}]}.
Include every key exactly once. An answer must occur verbatim in its response_tail. Use null when the tail does not explicitly commit to a final answer, including when it says another approach is needed.

ITEMS:
""" + json.dumps(payload, ensure_ascii=False)
    error = ""
    for _ in range(2):
        try:
            response = parse_json_object(complete(prompt, EXTRACTOR_MODEL_CONFIG))
            cases = response.get("cases")
            if not isinstance(cases, list):
                raise ValueError("missing cases list")
            extracted: dict[str, str | None] = {}
            for case in cases:
                key = str(case["key"])
                raw_answer = case["agent_answer"]
                answer = str(raw_answer).strip() if raw_answer is not None else None
                if key in extracted:
                    raise ValueError(f"duplicate key {key}")
                extracted[key] = answer
            return extracted
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            error = str(exc)
    raise RuntimeError(f"answer extraction failed: {error}")


def select_convertible_rows(
    candidates: list[tuple[int, dict[str, Any]]],
    answers: dict[str, tuple[str, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    targets = {
        classification: QUESTION_LIMIT // len(ERROR_CLASSES)
        + (index < QUESTION_LIMIT % len(ERROR_CLASSES))
        for index, classification in enumerate(ERROR_CLASSES)
    }
    selected: list[tuple[int, dict[str, Any]]] = []
    counts = defaultdict(int)
    for item in candidates:
        line_number, row = item
        classification = row["classification"]
        if counts[classification] == targets[classification]:
            continue
        if f"{line_number}:correct" not in answers or f"{line_number}:incorrect" not in answers:
            continue
        selected.append(item)
        counts[classification] += 1
    missing = {
        classification: targets[classification] - counts[classification]
        for classification in ERROR_CLASSES
        if counts[classification] < targets[classification]
    }
    if missing:
        raise RuntimeError(f"not enough cases with explicit source answers: {missing}")
    return selected


def build_cases(
    selected: list[tuple[int, dict[str, Any]]],
    answers: dict[str, tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    cases = []
    for number, (line_number, row) in enumerate(selected, 1):
        correct_answer, correct_extraction = answers[f"{line_number}:correct"]
        incorrect_answer, incorrect_extraction = answers[f"{line_number}:incorrect"]
        shared_source = {
            "dataset": DATASET,
            "revision": REVISION,
            "line": line_number,
            "idx": row["idx"],
            "classification": row["classification"],
            "modified_steps": row["modified_steps"],
            "error_steps": row["error_steps"],
            "error_reason": row["reason"],
        }
        cases.extend(
            [
                {
                    "id": f"prmbench_{number:03d}_correct",
                    "question": row["original_question"],
                    "agent_answer": correct_answer,
                    "agent_reasoning": "\n\n".join(row["original_process"]),
                    "expected_answer": correct_answer,
                    "agent_model_config": None,
                    "source": {
                        **shared_source,
                        "version": "correct",
                        "agent_reasoning_source": "original_process",
                        "agent_answer_extraction": correct_extraction,
                    },
                    "notes": "Published PRMBench correct process; expected_answer is its extracted answer for offline evaluation.",
                },
                {
                    "id": f"prmbench_{number:03d}_incorrect",
                    "question": row["modified_question"],
                    "agent_answer": incorrect_answer,
                    "agent_reasoning": "\n\n".join(row["modified_process"]),
                    "expected_answer": correct_answer,
                    "agent_model_config": None,
                    "source": {
                        **shared_source,
                        "version": "incorrect",
                        "agent_reasoning_source": "modified_process",
                        "agent_answer_extraction": incorrect_extraction,
                    },
                    "notes": "Published PRMBench error-injected process; expected_answer comes from the paired correct process.",
                },
            ]
        )
    return cases


if __name__ == "__main__":
    raise SystemExit(main())
