
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, overload

from datasets import load_from_disk

ACTIVE_DATASETS = ("musique", "reclor", "logiqa", "wiki2multihop")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CURATED_DIR = _PROJECT_ROOT / "resources" / "datasets" / "curated"

MERGE_MIN_CHARS = 500

MUSIQUE_MAX_REFS = 10

_DATASET_META = {
    "musique": {
        "domain": "Wikipedia multi-hop",
        "answer_type": "free_text",
        "description": "Multi-hop QA with gold decomposition steps and distractor paragraphs",
    },
    "reclor": {
        "domain": "GMAT/LSAT exams",
        "answer_type": "mcq",
        "description": "Formal logical reasoning (context-grounded subset)",
    },
    "logiqa": {
        "domain": "Chinese civil service exams",
        "answer_type": "mcq",
        "description": "Applied logical reasoning (grounded, context ≥ 200 chars)",
    },
    "wiki2multihop": {
        "domain": "Wikipedia comparison & bridge",
        "answer_type": "free_text",
        "description": "Multi-hop comparison & bridge reasoning with structured evidence chains (≥ 4 hops)",
    },
}

_PROMPT_FREE_TEXT = """\
Answer the question below by detailed reasoning step-by-step. Base your reasoning \
only on the provided references.

- Each reasoning step must be grounded in specific information from the references.
- At the end of each step, cite the references you used in brackets (e.g., [ref_1, ref_2]).
- If a step is a logical deduction from previous steps rather than directly from \
a reference, you may omit the citation.
- Do NOT introduce any information that is not present in the references.

{references_text}

{question}

Step 1: <reasoning based on references> [ref_N, ...]
Step 2: <reasoning based on references> [ref_N, ...]
...
Final Answer: <your answer>"""

_PROMPT_MCQ = """\
Answer the question below by detailed reasoning step-by-step. Base your reasoning \
only on the provided references, then select the best option.

- Each reasoning step must be grounded in specific information from the references.
- At the end of each step, cite the references you used in brackets (e.g., [ref_1, ref_2]).
- If a step is a logical deduction from previous steps rather than directly from \
a reference, you may omit the citation.
- Do NOT introduce any information that is not present in the references.

{references_text}

{question}

{options_text}

Step 1: <reasoning based on references> [ref_N, ...]
Step 2: <reasoning based on references> [ref_N, ...]
...
Final Answer: <selected option, e.g., A) ...>"""

_OPTION_LABELS = "ABCDEFGHIJ"

def merge_short_paragraphs(text: str, min_chars: int = MERGE_MIN_CHARS) -> list[str]:
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paras:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = raw_paras[0]
    for p in raw_paras[1:]:
        if len(current) < min_chars:
            current = current + "\n\n" + p
        else:
            chunks.append(current)
            current = p
    chunks.append(current)
    return chunks

def _build_musique_references(paragraphs: list[dict]) -> list[str]:
    refs = []
    for p in paragraphs:
        title = p.get("title", "")
        text = p.get("paragraph_text", "")
        if title:
            refs.append(f"[{title}]\n{text}")
        else:
            refs.append(text)
    return refs

def _build_text_references(text: str, min_chars: int = MERGE_MIN_CHARS) -> list[str]:
    return merge_short_paragraphs(text, min_chars=min_chars)

def _build_single_reference(text: str) -> list[str]:
    return [text.strip()]

def format_references(refs: list[str]) -> str:
    import re
    lines = []
    for i, ref in enumerate(refs, 1):
        clean_ref = re.sub(r"\n{2,}", "\n", ref.strip())
        lines.append(f"---\n[ref_{i}]\n{clean_ref}\n")
    lines.append("---")
    return "\n".join(lines)

@dataclass
class GraceExample:

    dataset_name: str
    index: int
    references: list[str]
    context: str
    question: str
    options: list[str] | None
    gold_answer: str
    gold_answer_index: int | None
    answer_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        references_text = format_references(self.references)
        if self.answer_type == "free_text":
            return _PROMPT_FREE_TEXT.format(
                references_text=references_text,
                question=self.question,
            )
        else:
            options_text = self._format_options()
            return _PROMPT_MCQ.format(
                references_text=references_text,
                question=self.question,
                options_text=options_text,
            )

    @property
    def num_references(self) -> int:
        return len(self.references)

    @property
    def context_length(self) -> int:
        return len(self.context)

    @property
    def id(self) -> str:
        raw_id = self.metadata.get("id") or self.metadata.get("id_string")
        if raw_id:
            return f"{self.dataset_name}_{raw_id}"
        return f"{self.dataset_name}_{self.index}"

    def _format_options(self) -> str:
        if not self.options:
            return ""
        lines = []
        for i, opt in enumerate(self.options):
            label = _OPTION_LABELS[i] if i < len(_OPTION_LABELS) else str(i)
            lines.append(f"{label}) {opt}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        q_preview = self.question[:80] + "..." if len(self.question) > 80 else self.question
        return (
            f"GraceExample(\n"
            f"  dataset={self.dataset_name!r},\n"
            f"  index={self.index},\n"
            f"  id={self.id!r},\n"
            f"  question={q_preview!r},\n"
            f"  answer_type={self.answer_type!r},\n"
            f"  gold_answer={self.gold_answer!r},\n"
            f"  num_references={self.num_references},\n"
            f"  context_length={self.context_length:,},\n"
            f"  num_options={len(self.options) if self.options else 0}\n"
            f")"
        )

    def show(self) -> None:
        print(f"{'─' * 70}")
        print(f"Dataset: {self.dataset_name}  |  Index: {self.index}  |  ID: {self.id}")
        print(f"Answer type: {self.answer_type}  |  Context: {self.context_length:,} chars  |  Refs: {self.num_references}")
        print(f"{'─' * 70}")
        print(f"\n📚 REFERENCES ({self.num_references} total):")
        for i, ref in enumerate(self.references, 1):
            preview = ref[:200] + "..." if len(ref) > 200 else ref
            print(f"  [ref_{i}] ({len(ref)} chars) {preview}")
        print(f"\n❓ QUESTION:\n{self.question}")
        if self.options:
            print(f"\n📋 OPTIONS:")
            for i, opt in enumerate(self.options):
                label = _OPTION_LABELS[i] if i < len(_OPTION_LABELS) else str(i)
                marker = " ✓" if i == self.gold_answer_index else ""
                print(f"  {label}) {opt}{marker}")
        print(f"\n✅ GOLD ANSWER: {self.gold_answer}")
        if self.metadata:
            print(f"\n🏷️  METADATA:")
            for k, v in self.metadata.items():
                v_str = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                print(f"  {k}: {v_str}")
        print(f"{'─' * 70}")

def _normalize_musique(row: dict, index: int) -> GraceExample:
    import random as _random

    paragraphs = row["paragraphs"]

    supporting = [p for p in paragraphs if p.get("is_supporting")]
    distractors = [p for p in paragraphs if not p.get("is_supporting")]

    n_distractors_needed = max(0, MUSIQUE_MAX_REFS - len(supporting))
    raw_id = row.get("id", "")
    rng = _random.Random(hash(raw_id) if raw_id else index)
    sampled_distractors = rng.sample(
        distractors, min(n_distractors_needed, len(distractors))
    )

    selected = supporting + sampled_distractors
    rng.shuffle(selected)

    references = _build_musique_references(selected)
    context = "\n\n".join(references)

    hop_count = None
    if raw_id:
        hop_str = raw_id.split("__")[0]
        digits = "".join(c for c in hop_str if c.isdigit())
        if digits:
            hop_count = int(digits[0])

    return GraceExample(
        dataset_name="musique",
        index=index,
        references=references,
        context=context,
        question=row["question"],
        options=None,
        gold_answer=row["answer"],
        gold_answer_index=None,
        answer_type="free_text",
        metadata={
            "id": raw_id,
            "paragraphs": paragraphs,
            "question_decomposition": row.get("question_decomposition"),
            "answer_aliases": row.get("answer_aliases", []),
            "hop_count": hop_count,
            "answerable": row.get("answerable"),
        },
    )

def _normalize_quality(row: dict, index: int) -> GraceExample:
    article = row["article"]
    options = row["options"]
    answer_idx = row["answer"]

    references = _build_text_references(article, min_chars=MERGE_MIN_CHARS)

    return GraceExample(
        dataset_name="quality",
        index=index,
        references=references,
        context=article,
        question=row["question"],
        options=options,
        gold_answer=options[answer_idx],
        gold_answer_index=answer_idx,
        answer_type="mcq",
        metadata={
            "article_length": row.get("article_length", len(article)),
        },
    )


def _normalize_reclor(row: dict, index: int) -> GraceExample:
    context = row["context"]
    answers = row["answers"]
    label = row["label"]

    references = _build_single_reference(context)

    return GraceExample(
        dataset_name="reclor",
        index=index,
        references=references,
        context=context,
        question=row["question"],
        options=answers,
        gold_answer=answers[label],
        gold_answer_index=label,
        answer_type="mcq",
        metadata={
            "id_string": row.get("id_string", ""),
            "context_length": row.get("context_length", len(context)),
            "label": label,
        },
    )

def _normalize_logiqa(row: dict, index: int) -> GraceExample:
    context = row["context"]
    options = row["options"]
    correct_idx = row["correct_option"]

    references = _build_single_reference(context)

    return GraceExample(
        dataset_name="logiqa",
        index=index,
        references=references,
        context=context,
        question=row["query"],
        options=options,
        gold_answer=options[correct_idx],
        gold_answer_index=correct_idx,
        answer_type="mcq",
        metadata={
            "context_length": row.get("context_length", len(context)),
            "correct_option": correct_idx,
        },
    )

def _normalize_wiki2multihop(row: dict, index: int) -> GraceExample:
    import random as _random

    paragraphs = row["paragraphs"]

    supporting = [p for p in paragraphs if p.get("is_supporting")]
    distractors = [p for p in paragraphs if not p.get("is_supporting")]

    n_distractors_needed = max(0, MUSIQUE_MAX_REFS - len(supporting))
    raw_id = row.get("id", "")
    rng = _random.Random(hash(raw_id) if raw_id else index)
    sampled_distractors = rng.sample(
        distractors, min(n_distractors_needed, len(distractors))
    )

    selected = supporting + sampled_distractors
    rng.shuffle(selected)

    references = _build_musique_references(selected)
    context = "\n\n".join(references)

    return GraceExample(
        dataset_name="wiki2multihop",
        index=index,
        references=references,
        context=context,
        question=row["question"],
        options=None,
        gold_answer=row["answer"],
        gold_answer_index=None,
        answer_type="free_text",
        metadata={
            "id": raw_id,
            "type": row.get("type", ""),
            "paragraphs": paragraphs,
            "evidences": row.get("evidences", []),
            "supporting_facts": row.get("supporting_facts"),
            "num_hops": row.get("num_hops"),
        },
    )

_NORMALIZERS = {
    "musique": _normalize_musique,
    "quality": _normalize_quality,
    "reclor": _normalize_reclor,
    "logiqa": _normalize_logiqa,
    "wiki2multihop": _normalize_wiki2multihop,
}

class GraceDataset:

    def __init__(self, name: str, data_dir: str | Path | None = None) -> None:
        name = name.lower().strip()
        self.name = name
        self._meta = _DATASET_META[name]
        self._normalizer = _NORMALIZERS[name]

        curated_dir = Path(data_dir) if data_dir else _CURATED_DIR
        dataset_path = curated_dir / name

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Curated dataset not found at {dataset_path}. "
                f"Run 'uv run python src/data_processing/stages/curate_samples.py' first."
            )

        self._raw_dataset = load_from_disk(str(dataset_path))

    @property
    def domain(self) -> str:
        return self._meta["domain"]

    @property
    def answer_type(self) -> str:
        return self._meta["answer_type"]

    @property
    def description(self) -> str:
        return self._meta["description"]

    @property
    def columns(self) -> list[str]:
        return list(self._raw_dataset.column_names)

    @property
    def schema(self) -> dict[str, str]:
        features = self._raw_dataset.features
        result = {}
        for col_name, feat in features.items():
            result[col_name] = str(feat)
        return result

    @overload
    def __getitem__(self, idx: int) -> GraceExample: ...
    @overload
    def __getitem__(self, idx: slice) -> list[GraceExample]: ...

    def __getitem__(self, idx: int | slice) -> GraceExample | list[GraceExample]:
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self._normalizer(self._raw_dataset[i], i) for i in indices]
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        return self._normalizer(self._raw_dataset[idx], idx)

    def __len__(self) -> int:
        return len(self._raw_dataset)

    def __iter__(self) -> Iterator[GraceExample]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return (
            f"GraceDataset(name={self.name!r}, size={len(self)}, "
            f"answer_type={self.answer_type!r}, domain={self.domain!r})"
        )

    def raw(self, idx: int) -> dict:
        return self._raw_dataset[idx]

    def info(self) -> None:
        print(f"\n{'═' * 70}")
        print(f"  GRACE Dataset: {self.name.upper()}")
        print(f"{'═' * 70}")
        print(f"  Domain:       {self.domain}")
        print(f"  Description:  {self.description}")
        print(f"  Answer type:  {self.answer_type}")
        print(f"  Size:         {len(self):,} examples")
        print(f"{'─' * 70}")
        print(f"  Raw columns:  {', '.join(self.columns)}")
        print(f"{'─' * 70}")
        print(f"  Schema:")
        for col, dtype in self.schema.items():
            print(f"    {col}: {dtype}")
        print(f"{'─' * 70}")

        ctx_lengths = self._get_context_lengths()
        ctx_lengths.sort()
        avg_len = sum(ctx_lengths) / len(ctx_lengths)
        median_len = ctx_lengths[len(ctx_lengths) // 2]
        print(f"  Context length (all {len(ctx_lengths):,} examples):")
        print(f"    Min:    {ctx_lengths[0]:>8,} chars")
        print(f"    Median: {median_len:>8,} chars")
        print(f"    Mean:   {avg_len:>8,.0f} chars")
        print(f"    Max:    {ctx_lengths[-1]:>8,} chars")

        sample_size = min(len(self), 200)
        ref_counts = []
        ref_lengths = []
        for i in range(sample_size):
            ex = self[i]
            ref_counts.append(ex.num_references)
            ref_lengths.extend(len(r) for r in ex.references)
        ref_counts.sort()
        ref_lengths.sort()
        print(f"{'─' * 70}")
        print(f"  References (sample of {sample_size}):")
        print(f"    Refs/example: median={ref_counts[len(ref_counts)//2]}, "
              f"mean={sum(ref_counts)/len(ref_counts):.0f}, "
              f"range=[{ref_counts[0]}, {ref_counts[-1]}]")
        print(f"    Chars/ref:    median={ref_lengths[len(ref_lengths)//2]}, "
              f"mean={sum(ref_lengths)/len(ref_lengths):.0f}")
        print(f"{'═' * 70}\n")

    def _get_context_lengths(self) -> list[int]:
        if self.name == "musique":
            lengths = []
            for i in range(len(self)):
                row = self._raw_dataset[i]
                total = sum(
                    len(p.get("title", "")) + len(p.get("paragraph_text", "")) + 5
                    for p in row["paragraphs"]
                )
                lengths.append(total)
            return lengths
        elif self.name == "quality":
            return [self._raw_dataset[i]["article_length"] for i in range(len(self))]
        elif self.name in ("reclor", "logiqa"):
            return [self._raw_dataset[i]["context_length"] for i in range(len(self))]
        elif self.name == "wiki2multihop":
            lengths = []
            for i in range(len(self)):
                row = self._raw_dataset[i]
                total = sum(
                    len(p.get("title", "")) + len(p.get("paragraph_text", "")) + 5
                    for p in row["paragraphs"]
                )
                lengths.append(total)
            return lengths
        else:
            return [self[i].context_length for i in range(min(len(self), 100))]

    def sample(self, n: int = 1, seed: int = 42) -> list[GraceExample]:
        import random as _random

        rng = _random.Random(seed)
        indices = rng.sample(range(len(self)), min(n, len(self)))
        return [self[i] for i in indices]

def load_all_datasets(
    data_dir: str | Path | None = None,
    include_inactive: bool = False,
) -> dict[str, GraceDataset]:
    names = ACTIVE_DATASETS
    result = {}
    for name in names:
        try:
            result[name] = GraceDataset(name, data_dir=data_dir)
        except FileNotFoundError:
            print(f"⚠️  Skipping {name}: curated data not found")
    return result
