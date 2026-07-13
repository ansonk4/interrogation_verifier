"""Convert DeltaBench's published responses to graph-verifier cases."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from graph_verifier.utils.llm import complete


ROOT = Path(__file__).parent
SOURCE = ROOT / "DeltaBench" / "data" / "Deltabench_v1.jsonl"
OUTPUT = ROOT / "deltabench_cases.jsonl"
EXTRACTOR_MODEL_CONFIG = "model/openrouter/deepseek/deepseek-v4-flash-high.json"
REPORTED_GENERATORS = ["QwQ-32B-Preview", "DeepSeek-R1", "Gemini 2.0 Flash Thinking"]
CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
DOUBLE_BRACKET = re.compile(r"\[\[([^\[\]]+)\]\]")


def parse_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("answer extractor did not return JSON") from None
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("answer extractor JSON must be an object")
    return data


def call_json(prompt: str) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(3):
        try:
            return parse_json(complete(prompt, EXTRACTOR_MODEL_CONFIG))
        except Exception as exc:
            error = exc
            time.sleep(attempt + 1)
    raise RuntimeError(f"answer extraction failed: {error}")


def boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    marker = "\\boxed{"
    start = 0
    while (found := text.find(marker, start)) != -1:
        pos, depth = found + len(marker), 1
        answer: list[str] = []
        while pos < len(text) and depth:
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            answer.append(char)
            pos += 1
        if depth == 0:
            answers.append("".join(answer).strip())
        start = max(pos + 1, found + len(marker))
    return answers


def extract_unstructured(response: str) -> tuple[str, dict[str, Any]]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", response) if part.strip()]
    prompt = """Select the minimum paragraph range that states the final answer explicitly claimed by MODEL RESPONSE.
Do not solve, correct, evaluate, summarize, or infer missing content.
Return only JSON: {"paragraph_indices": [integer, ...], "agent_answer": string|null}.
Use source order. Include multiple paragraphs only when the answer spans them.
Exclude confidence-only follow-ups such as "I think that is correct" when an earlier paragraph states the actual answer.
When possible, agent_answer should copy only the final value, choice, statement, or submitted answer verbatim from the selected paragraphs.
If the response never commits to an answer, return an empty index list and null agent_answer.

NUMBERED PARAGRAPHS:
""" + json.dumps(
        [{"index": index, "content": paragraph} for index, paragraph in enumerate(paragraphs)],
        ensure_ascii=False,
    )
    data = call_json(prompt)
    indices = data.get("paragraph_indices")
    if indices == []:
        return "", {"method": "no_explicit_answer", "model_config": EXTRACTOR_MODEL_CONFIG}
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        raise ValueError("answer extractor returned invalid paragraph indices")
    indices = list(dict.fromkeys(indices))
    if indices != sorted(indices) or any(index < 0 or index >= len(paragraphs) for index in indices):
        raise ValueError("answer extractor returned out-of-range paragraph indices")
    selected = "\n\n".join(paragraphs[index] for index in indices)
    answer = data.get("agent_answer")
    if isinstance(answer, str) and answer.strip() and answer.strip() in selected:
        return answer.strip(), {
            "method": "llm_exact_substring",
            "paragraph_indices": indices,
            "model_config": EXTRACTOR_MODEL_CONFIG,
        }
    return selected, {
        "method": "llm_final_paragraphs",
        "paragraph_indices": indices,
        "model_config": EXTRACTOR_MODEL_CONFIG,
    }


def extract_code(response: str, blocks: list[str]) -> tuple[str, dict[str, Any]]:
    if len(blocks) == 1:
        indices = [0]
    else:
        prompt = """Select the fenced code block or blocks that constitute the model's final submitted solution.
Do not solve, correct, or rewrite any code. Omit examples, sample output, and abandoned attempts.
Return only JSON: {"indices": [integer, ...]}. Use source order. Return an empty list if no block is submitted as the answer.

MODEL RESPONSE AND NUMBERED CODE BLOCKS:
""" + json.dumps(
            {
                "model_response": response,
                "code_blocks": [{"index": index, "content": block} for index, block in enumerate(blocks)],
            },
            ensure_ascii=False,
        )
        indices = call_json(prompt).get("indices")
        if not isinstance(indices, list) or any(type(index) is not int for index in indices):
            raise ValueError("answer extractor returned invalid code-block indices")
        indices = list(dict.fromkeys(indices))
        if indices != sorted(indices) or any(index < 0 or index >= len(blocks) for index in indices):
            raise ValueError("answer extractor returned out-of-range code-block indices")
    if not indices:
        return "", {"method": "no_explicit_answer", "model_config": EXTRACTOR_MODEL_CONFIG}
    return "\n\n".join(blocks[index].strip() for index in indices), {
        "method": "code_blocks",
        "indices": indices,
        "model_config": EXTRACTOR_MODEL_CONFIG if len(blocks) > 1 else None,
    }


def extract_agent_answer(item: dict[str, Any], response: str) -> tuple[str, dict[str, Any]]:
    if item["task_l1"] == "code":
        blocks = CODE_BLOCK.findall(response)
        if blocks:
            return extract_code(response, blocks)
    boxes = boxed_answers(response)
    if boxes:
        return boxes[-1], {"method": "last_boxed", "model_config": None}
    brackets = DOUBLE_BRACKET.findall(response)
    if brackets:
        return brackets[-1].strip(), {"method": "last_double_bracket", "model_config": None}
    return extract_unstructured(response)


def published_reasoning(item: dict[str, Any]) -> tuple[str, str]:
    if item["long_cot"]:
        return item["long_cot"], "long_cot"
    response = "\n\n".join(section["content"] for section in item["sections"])
    if not response:
        raise ValueError(f"{item['id']} has no published response")
    return response, "sections_content_recovery"


def convert(index_item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    row, item = index_item
    try:
        reasoning, reasoning_source = published_reasoning(item)
        agent_answer, extraction = extract_agent_answer(item, reasoning)
    except Exception as exc:
        raise RuntimeError(f"row {row} ({item['id']}): {exc}") from exc
    return {
        "id": f"deltabench_{item['id']}",
        "question": item["question"],
        "agent_answer": agent_answer,
        "agent_reasoning": reasoning,
        "expected_answer": item["answer"] or None,
        "agent_model_config": None,
        "source": {
            "dataset": "OpenStellarTeam/DeltaBench",
            "row": row,
            "origin": item["origin"],
            "task_l1": item["task_l1"],
            "task_l2": item["task_l2"],
            "final_correct": item["final_correct"],
            "agent_reasoning_source": reasoning_source,
            "agent_answer_extraction": extraction,
            "generator": {
                "model": None,
                "model_config": None,
                "per_example_attribution_available": False,
                "reported_model_pool": REPORTED_GENERATORS,
                "sampling": "random; parameters not reported",
            },
        },
    }


def main() -> None:
    with SOURCE.open(encoding="utf-8") as source:
        items = [json.loads(line) for line in source if line.strip()]
    with ThreadPoolExecutor(max_workers=6) as executor:
        cases = [
            case
            for case in executor.map(convert, enumerate(items))
            if case["agent_answer"].strip()
        ]
    with OUTPUT.open("w", encoding="utf-8") as output:
        for case in cases:
            output.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"{OUTPUT}: {len(cases)} cases; skipped {len(items) - len(cases)} without explicit answers")


if __name__ == "__main__":
    main()
