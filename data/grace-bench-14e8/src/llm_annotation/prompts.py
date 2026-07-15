
from __future__ import annotations

import json
import re
from pathlib import Path

from src.utils.prompt_helpers import (
    reconstruct_references,
    build_options_block,
    build_previous_steps_block,
)
from src.utils.trace_loader import TraceRecord

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TAXONOMY_DIR = _PROJECT_ROOT / "taxonomy"
_GROUNDING_TAXONOMY_PATH = _TAXONOMY_DIR / "taxonomy_grounding.json"
_INFERENCE_TAXONOMY_PATH = _TAXONOMY_DIR / "taxonomy_inference.json"

INFERENCE_DATASETS = {"logiqa", "reclor"}
GROUNDING_DATASETS = {"musique", "wiki2multihop"}

def get_track(dataset: str) -> str:
    ds = dataset.strip().lower()
    if ds in INFERENCE_DATASETS:
        return "inference"
    elif ds in GROUNDING_DATASETS:
        return "grounding"
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Expected one of: {INFERENCE_DATASETS | GROUNDING_DATASETS}"
        )

def load_grounding_taxonomy(taxonomy_path: Path | None = None) -> dict:
    path = taxonomy_path or _GROUNDING_TAXONOMY_PATH
    with open(path, "r") as f:
        return json.load(f)

def load_inference_taxonomy(taxonomy_path: Path | None = None) -> dict:
    path = taxonomy_path or _INFERENCE_TAXONOMY_PATH
    with open(path, "r") as f:
        return json.load(f)

def get_track_categories(track: str, grounding_taxonomy: dict | None = None, inference_taxonomy: dict | None = None) -> list[dict]:
    if track == "inference":
        if inference_taxonomy is None:
            inference_taxonomy = load_inference_taxonomy()
        return inference_taxonomy["categories"]
    else:
        if grounding_taxonomy is None:
            grounding_taxonomy = load_grounding_taxonomy()
        return grounding_taxonomy["categories"]

def build_inference_taxonomy_block(inference_taxonomy: dict) -> str:
    lines = []
    for cat in inference_taxonomy["categories"]:
        lines.append(f"### {cat['name']} (`{cat['id']}`)")
        lines.append(f"**Question:** \"{cat['annotator_question']}\"")
        lines.append(f"**Definition:** {cat['definition']}")
        lines.append("")

    return "\n".join(lines)

def build_grounding_taxonomy_block(grounding_taxonomy: dict) -> str:
    lines = []
    for cat in grounding_taxonomy["categories"]:
        lines.append(f"### {cat['name']} (`{cat['id']}`)")
        lines.append(f"**Question:** \"{cat['annotator_question']}\"")
        lines.append(f"**Definition:** {cat['definition']}")

        if cat.get("sub_flags"):
            lines.append("**Sub-flags (optional fine-grained label):**")
            for sf in cat["sub_flags"]:
                lines.append(f"  - `{sf['id']}`: {sf['name']}")
        lines.append("")

    return "\n".join(lines)

def build_taxonomy_block(track: str = "grounding", grounding_taxonomy: dict | None = None, inference_taxonomy: dict | None = None) -> str:
    if track == "inference":
        if inference_taxonomy is None:
            inference_taxonomy = load_inference_taxonomy()
        return build_inference_taxonomy_block(inference_taxonomy)
    else:
        if grounding_taxonomy is None:
            grounding_taxonomy = load_grounding_taxonomy()
        return build_grounding_taxonomy_block(grounding_taxonomy)

ANNOTATION_SYSTEM = (
    "You are a professional annotator for the GRACE benchmark, which evaluates "
    "step-level faithfulness of LLM reasoning traces. Your job is to read "
    "every reasoning step, compare each against the provided context, and "
    "determine whether each is faithful. If it is not faithful, you classify "
    "the error using the taxonomy provided."
)

CATEGORY_EXAMPLES = {
    "inference": "reversed_reasoning, wrong_argument_reading, rule_violation, overreaching_claim",
    "grounding": "groundedness_violation, contradiction, confusion, evidence_neglect",
}

ANNOTATION_PROMPT = """\
You are a faithfulness evaluator for the GRACE benchmark, which assesses \
step-level faithfulness of LLM reasoning traces. Your job is to read EVERY \
reasoning step, compare each against the provided context, and determine \
whether each step is faithful.

For EACH step in the reasoning trace below, determine whether it is \
FAITHFUL or UNFAITHFUL based on the reference context.
If a step is UNFAITHFUL, classify the error into exactly ONE category \
from the taxonomy.

- **Faithful**: All claims in the step are supported by or logically \
derivable from the provided context.
- **Unfaithful**: The step contains information that conflicts with, \
goes beyond, or misuses the provided context.

{taxonomy_block}

{context}

{question}
{options_block}

{trace_steps}

Evaluate ALL {total_steps} steps above. For EACH step, respond in \
exactly this format:

<step id="1">
<explanation>Compare the step's claims against the context. 2-4 sentences.</explanation>
<faithfulness>faithful | unfaithful</faithfulness>
<error_category>Exact category name (e.g., {category_examples}). \
Only if unfaithful; write "none" if faithful.</error_category>
</step>
<step id="2">
...
</step>

You MUST produce one <step> block for each of the {total_steps} steps, \
in order from step 1 to step {total_steps}."""

_DATASET_DISPLAY = {
    "logiqa": "LogiQA",
    "reclor": "ReClor",
    "musique": "MuSiQue",
    "wiki2multihop": "2WikiMultiHopQA",
}

_VALID_CATEGORIES_INFERENCE = {
    "reversed_reasoning",
    "wrong_argument_reading",
    "rule_violation",
    "overreaching_claim",
}

_VALID_CATEGORIES_GROUNDING = {
    "groundedness_violation",
    "contradiction",
    "confusion",
    "evidence_neglect",
}

_VALID_CATEGORIES_ALL = _VALID_CATEGORIES_INFERENCE | _VALID_CATEGORIES_GROUNDING

_VALID_FAITHFULNESS = {"faithful", "unfaithful"}

def get_valid_categories(track: str) -> set[str]:
    if track == "inference":
        return _VALID_CATEGORIES_INFERENCE
    elif track == "grounding":
        return _VALID_CATEGORIES_GROUNDING
    else:
        return _VALID_CATEGORIES_ALL

def validate_faithfulness(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    return v if v in _VALID_FAITHFULNESS else None

def validate_category(category: str | None, dataset: str | None = None) -> str | None:
    if not category:
        return None
    cat = category.strip().lower()
    if cat in ("null", "none", "n/a", "na", "-"):
        return None

    if dataset:
        track = get_track(dataset)
        valid = get_valid_categories(track)
    else:
        valid = _VALID_CATEGORIES_ALL

    return cat if cat in valid else None

def build_all_trace_steps_block(steps: list[dict]) -> str:
    lines = []
    for step in steps:
        sid = step.get("step_id", step.get("id", "?"))
        text = step.get("text", "")
        lines.append(f"Step {sid}: {text}")
    return "\n".join(lines)

def format_all_steps_prompt(
    trace: TraceRecord,
    grounding_taxonomy: dict | None = None,
    inference_taxonomy: dict | None = None,
) -> str:
    dataset = trace.dataset
    question = trace.question
    options = trace.options
    total_steps = trace.num_steps

    formatted_context = reconstruct_references(trace.context, dataset)

    track = get_track(dataset)

    if track == "inference":
        if inference_taxonomy is None:
            inference_taxonomy = load_inference_taxonomy()
        taxonomy_block = build_inference_taxonomy_block(inference_taxonomy)
    else:
        if grounding_taxonomy is None:
            grounding_taxonomy = load_grounding_taxonomy()
        taxonomy_block = build_grounding_taxonomy_block(grounding_taxonomy)

    options_block = build_options_block(options)
    trace_steps = build_all_trace_steps_block(
        [{"step_id": s.step_id, "text": s.text} for s in trace.steps]
    )
    category_examples = CATEGORY_EXAMPLES[track]

    return ANNOTATION_PROMPT.format(
        taxonomy_block=taxonomy_block,
        context=formatted_context,
        question=question,
        options_block=options_block,
        trace_steps=trace_steps,
        total_steps=total_steps,
        category_examples=category_examples,
    )

def parse_all_steps_response(
    raw_text: str,
    expected_steps: int,
    dataset: str | None = None,
) -> list[dict]:
    text = raw_text
    think_match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if think_match:
        text = think_match.group(1).strip()

    step_blocks = re.findall(
        r'<step\s+id=["\']?(\d+)["\']?\s*>(.*?)</step>',
        text,
        re.DOTALL,
    )

    parsed_by_id: dict[int, dict] = {}

    for step_id_str, block_content in step_blocks:
        step_id = int(step_id_str)

        faithfulness = None
        m = re.search(r'<faithfulness>\s*(.*?)\s*</faithfulness>', block_content, re.DOTALL)
        if m:
            faithfulness = validate_faithfulness(m.group(1))

        error_category = None
        m = re.search(r'<error_category>\s*(.*?)\s*</error_category>', block_content, re.DOTALL)
        if m:
            error_category = validate_category(m.group(1), dataset=dataset)

        explanation = None
        m = re.search(r'<explanation>\s*(.*?)\s*</explanation>', block_content, re.DOTALL)
        if m:
            explanation = m.group(1).strip()

        parsed_by_id[step_id] = {
            "step_id": step_id,
            "faithfulness": faithfulness,
            "error_category": error_category,
            "explanation": explanation,
            "raw_response": block_content.strip(),
            "parse_error": faithfulness is None,
        }

    results = []
    for sid in range(1, expected_steps + 1):
        if sid in parsed_by_id:
            results.append(parsed_by_id[sid])
        else:
            results.append({
                "step_id": sid,
                "faithfulness": None,
                "error_category": None,
                "explanation": None,
                "raw_response": "",
                "parse_error": True,
            })

    return results
