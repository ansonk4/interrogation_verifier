

from __future__ import annotations

import json
import re
from pathlib import Path

LOGIC_DATASETS = {"logiqa", "reclor"}
EVIDENCE_DATASETS = {"musique", "wiki2multihop"}


def get_track(dataset: str) -> str:
    ds = dataset.strip().lower()
    if ds in LOGIC_DATASETS:
        return "logic"
    elif ds in EVIDENCE_DATASETS:
        return "evidence"
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

TAXONOMY_LOGIC = """\
### Error Categories (GRACE-Logic)

A logic error means the premises are correctly read from the context, but the logical operation applied is invalid. If the step is unfaithful, classify into exactly ONE category. Check in order — take the FIRST match.

1. **Reversed Reasoning** (`reversed_reasoning`)
   The step reverses or confuses the direction of a logical relationship. This includes swapping cause and effect, treating a sufficient condition as necessary (or vice versa), affirming the consequent, denying the antecedent, or inverting if-then implications.
   *Quick check: Did the step get the direction of a relationship backwards?*

2. **Wrong Argument Reading** (`wrong_argument_reading`)
   The step misidentifies the logical structure, roles, or components of an argument. This includes confusing premises with conclusions, mislabeling the type of flaw in an argument, misidentifying the point of disagreement between speakers, or treating distinct concepts as semantically equivalent without textual support.
   *Quick check: Did the step misidentify what the argument is saying or doing?*

3. **Rule Violation** (`rule_violation`)
   The step violates an explicit rule, constraint, or boundary stated in the context. This includes breaking one-to-one mappings, ignoring mutual exclusivity, incorrect set operations, invalid elimination of candidates, violating disjunction logic, or failing at constraint satisfaction.
   *Quick check: Did the step ignore or break a constraint stated in the text?*

4. **Overreaching Claim** (`overreaching_claim`)
   The step extends a conclusion beyond what the premises logically support. This includes overgeneralizing from specific cases, misapplying quantifier scope, confusing relative with absolute claims, making unjustified causal attributions, treating absence of evidence as evidence of absence or introduce unsupported assumptions.
   *Quick check: Did the step claim more than the evidence actually supports?*"""

TAXONOMY_EVIDENCE = """\
### Error Categories (GRACE-Evidence)

An evidence error means the step makes a factual claim that conflicts with, goes beyond, or ignores the provided context. If the step is unfaithful, classify into exactly ONE category. Check in order — take the FIRST match.

1. **Groundedness Violation** (`groundedness_violation`)
   The step makes a claim — whether a fabricated detail, a true-but-unsourced fact from training data, or a plausible-but-ungrounded inference — that is not supported by the provided context.
   *Quick check: Does the step claim something not supported by the context?*

2. **Contradiction** (`contradiction`)
   The step directly and unambiguously opposes an explicit statement in the context. The context says X; the step says not-X.
   *Quick check: Does the context explicitly say the opposite?*

3. **Confusion** (`confusion`)
   The step uses information that IS in the context, but attaches it to the wrong entity, merges distinct entities, reverses a relationship direction, or confuses properties between entities.
   *Quick check: Did it mix up entities, facts, or relationships?*

4. **Evidence Neglect** (`evidence_neglect`)
   The step claims information is missing or unavailable when it IS present in the context, or fails to track entity state changes across a narrative.
   *Quick check: Does it ignore or deny available information?*"""

TRACK_TAXONOMY = {
    "logic": TAXONOMY_LOGIC,
    "evidence": TAXONOMY_EVIDENCE,
}

CATEGORY_EXAMPLES = {
    "logic": "reversed_reasoning, wrong_argument_reading, rule_violation, overreaching_claim",
    "evidence": "groundedness_violation, contradiction, confusion, evidence_neglect",
}


_CATEGORY_ALIASES: dict[str, str] = {
    "e1": "groundedness_violation", "e2": "contradiction",
    "e3": "confusion", "e4": "evidence_neglect",
    "l1": "reversed_reasoning", "l2": "wrong_argument_reading",
    "l3": "rule_violation", "l4": "overreaching_claim",
}


def _normalize_category(raw: str) -> str:

    val = raw.strip().lower()
    if val in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[val]
    prefix = val.split()[0].rstrip("—-:.")
    if prefix in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[prefix]
    return val

def format_passages_as_references(passages: list[dict]) -> str:

    if not passages:
        return "(No context provided)"

    lines = []
    for passage in passages:
        ref_id = passage.get("ref_id", "ref_?")
        title = passage.get("title")
        text = passage.get("text", "")

        if title:
            lines.append(f"---\n[{ref_id}] {title}\n{text}\n")
        else:
            lines.append(f"---\n[{ref_id}]\n{text}\n")
    lines.append("---")
    return "\n".join(lines)


def build_options_block(options: list[str] | None) -> str:

    if not options:
        return ""
    labels = "ABCDEFGHIJ"
    lines = ["Options:"]
    for i, opt in enumerate(options):
        label = labels[i] if i < len(labels) else str(i)
        lines.append(f"  {label}) {opt}")
    return "\n".join(lines)


def build_all_trace_steps_block(steps: list[dict]) -> str:

    lines = []
    for step in steps:
        sid = step["step_id"]
        text = step.get("text", step.get("text_without_citations", ""))
        lines.append(f"Step {sid}: {text}")
    return "\n".join(lines)


ALLSTEPS_PROMPT = """\
You are a faithfulness evaluator for the GRACE benchmark, which assesses step-level faithfulness of LLM reasoning traces. Your job is to read EVERY reasoning step, compare each against the provided context, and determine whether each step is faithful.

## Task
For EACH step in the reasoning trace below, determine whether it is FAITHFUL or UNFAITHFUL based on the reference context.
If a step is UNFAITHFUL, classify the error into exactly ONE category from the taxonomy.

## Definitions
- **Faithful**: All claims in the step are supported by or logically derivable from the provided context.
- **Unfaithful**: The step contains information that conflicts with, goes beyond, or misuses the provided context.

{taxonomy_block}

## Context
{context}

## Question
{question}
{options_block}

## Reasoning Trace ({total_steps} steps)
{trace_steps}

## Instructions
Evaluate ALL {total_steps} steps above. For EACH step, respond in exactly this format:

<step id="1">
<explanation>Compare the step's claims against the context. 2-4 sentences.</explanation>
<faithfulness>faithful | unfaithful</faithfulness>
<error_category>Exact category name (e.g., {category_examples}). Only if unfaithful; write "none" if faithful.</error_category>
</step>
<step id="2">
...
</step>

You MUST produce one <step> block for each of the {total_steps} steps, in order from step 1 to step {total_steps}."""


def build_all_steps_prompt(trace: dict) -> str:

    dataset = trace["dataset"]
    question = trace.get("question", "")
    options = trace.get("options")
    steps = trace.get("steps", [])
    total_steps = len(steps)
    passages = trace.get("passages", [])

    track = get_track(dataset)
    taxonomy_block = TRACK_TAXONOMY[track]

    if passages:
        formatted_context = format_passages_as_references(passages)
    else:
        from src.utils.prompt_helpers import reconstruct_references
        formatted_context = reconstruct_references(trace.get("context", ""), dataset)

    options_block = build_options_block(options)
    trace_steps = build_all_trace_steps_block(steps)
    category_examples = CATEGORY_EXAMPLES[track]

    return ALLSTEPS_PROMPT.format(
        taxonomy_block=taxonomy_block,
        context=formatted_context,
        question=question,
        options_block=options_block,
        trace_steps=trace_steps,
        total_steps=total_steps,
        category_examples=category_examples,
    )




def parse_all_steps_response(raw_text: str, expected_steps: int) -> list[dict]:

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
            val = m.group(1).strip().lower()
            if val in ("faithful", "unfaithful"):
                faithfulness = val

        error_category = None
        m = re.search(r'<error_category>\s*(.*?)\s*</error_category>', block_content, re.DOTALL)
        if m:
            val = m.group(1).strip().lower()
            if val and val not in ("null", "none", "n/a", "na", "-"):
                error_category = _normalize_category(val)

        explanation = None
        m = re.search(r'<explanation>\s*(.*?)\s*</explanation>', block_content, re.DOTALL)
        if m:
            explanation = m.group(1).strip()

        parsed_by_id[step_id] = {
            "step_id": step_id,
            "pred_faithfulness": faithfulness,
            "pred_error_category": error_category,
            "pred_explanation": explanation,
            "raw_response": block_content.strip(),
        }

    results = []
    for sid in range(1, expected_steps + 1):
        if sid in parsed_by_id:
            results.append(parsed_by_id[sid])
        else:
            results.append({
                "step_id": sid,
                "pred_faithfulness": None,
                "pred_error_category": None,
                "pred_explanation": None,
                "raw_response": "",
            })

    return results




def build_trace_prediction(
    trace: dict,
    step_predictions: list[dict],
    model_name: str,
) -> dict:

    steps = trace.get("steps", [])
    gold_by_id = {s["step_id"]: s for s in steps}

    enriched_preds = []
    for pred in step_predictions:
        sid = pred["step_id"]
        gold_step = gold_by_id.get(sid, {})
        enriched_preds.append({
            "step_id": sid,
            "step_text": gold_step.get("text", gold_step.get("text_without_citations", "")),
            "pred_faithfulness": pred.get("pred_faithfulness"),
            "pred_error_category": pred.get("pred_error_category"),
            "pred_explanation": pred.get("pred_explanation"),
            "raw_response": pred.get("raw_response", ""),
            "gold_faithfulness": gold_step.get("gold_faithfulness") or gold_step.get("faithfulness"),
            "gold_error_category": gold_step.get("gold_error_category") or gold_step.get("error_category"),
        })

    return {
        "trace_id": trace["trace_id"],
        "dataset": trace["dataset"],
        "track": trace.get("track", ""),
        "model": model_name,
        "generator_model": trace.get("model", ""),
        "split": trace.get("split", ""),
        "context": trace.get("context", ""),
        "question": trace.get("question", ""),
        "options": trace.get("options"),
        "gold_answer": trace.get("gold_answer", ""),
        "num_steps": len(steps),
        "step_predictions": enriched_preds,
    }


def load_predictions(path: Path) -> list[dict]:

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
