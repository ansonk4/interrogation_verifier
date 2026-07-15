
from __future__ import annotations

_SINGLE_REF_DATASETS = {"reclor", "logiqa"}

_TITLED_PARA_DATASETS = {"musique", "wiki2multihop"}

_MERGE_MIN_CHARS = 500

def reconstruct_references(context: str, dataset: str) -> str:
    import re as _re

    if dataset in _SINGLE_REF_DATASETS:
        refs = [context.strip()]

    elif dataset in _TITLED_PARA_DATASETS:
        refs = [p.strip() for p in context.split("\n\n") if p.strip()]

    else:
        refs = [context.strip()]

    lines = []
    for i, ref in enumerate(refs, 1):
        clean_ref = _re.sub(r"\n{2,}", "\n", ref.strip())
        lines.append(f"---\n[ref_{i}]\n{clean_ref}\n")
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

def build_previous_steps_block(steps_texts: list[str]) -> str:
    if not steps_texts:
        return "(This is the first step — no previous steps)"
    lines = []
    for i, text in enumerate(steps_texts, 1):
        lines.append(f"Step {i}: {text}")
    return "\n".join(lines)
