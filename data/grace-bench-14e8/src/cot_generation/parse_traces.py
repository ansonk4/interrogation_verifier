
from __future__ import annotations

import re
from dataclasses import dataclass, field

_STEP_RE = re.compile(
    r"^[Ss]tep\s+(\d+)\s*:\s*(.*)",
    re.MULTILINE,
)

_FINAL_ANSWER_RE = re.compile(
    r"^[Ff]inal\s+[Aa]nswer\s*:\s*(.*)",
    re.MULTILINE,
)

_CITATION_RE = re.compile(
    r"\[ref_(\d+)(?:\s*,\s*ref_(\d+))*\]"
)

_INDIVIDUAL_REF_RE = re.compile(r"ref_(\d+)")

@dataclass
class ParsedStep:
    id: int
    text: str
    text_without_citations: str = ""
    citations: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.citations = []
        matches = _CITATION_RE.findall(self.text)
        for match_tuple in _CITATION_RE.finditer(self.text):
            refs = _INDIVIDUAL_REF_RE.findall(match_tuple.group())
            self.citations.extend(f"ref_{r}" for r in refs)

        self.text_without_citations = _CITATION_RE.sub("", self.text).strip()

        seen = set()
        unique_citations = []
        for c in self.citations:
            if c not in seen:
                seen.add(c)
                unique_citations.append(c)
        self.citations = unique_citations

@dataclass
class ParsedTrace:
    raw_text: str
    steps: list[ParsedStep] = field(default_factory=list)
    final_answer: str = ""
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.steps) >= 1 and bool(self.final_answer.strip())

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "steps": [
                {
                    "id": s.id,
                    "text": s.text,
                    "text_without_citations": s.text_without_citations,
                    "citations": s.citations,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "num_steps": self.num_steps,
            "is_valid": self.is_valid,
            "parse_warnings": self.parse_warnings,
        }

def parse_trace(raw_text: str) -> ParsedTrace:
    if not raw_text or not raw_text.strip():
        return ParsedTrace(
            raw_text=raw_text or "",
            parse_warnings=["Empty model output"],
        )

    warnings: list[str] = []
    text = raw_text.strip()

    step_matches = list(_STEP_RE.finditer(text))
    final_match = _FINAL_ANSWER_RE.search(text)

    boundaries = []
    for m in step_matches:
        boundaries.append(("step", m.start(), m))
    if final_match:
        boundaries.append(("final", final_match.start(), final_match))

    boundaries.sort(key=lambda x: x[1])

    steps: list[ParsedStep] = []
    for i, (kind, _pos, match) in enumerate(boundaries):
        if kind != "step":
            continue

        step_num = int(match.group(1))
        first_line = match.group(2).strip()

        content_start = match.end()
        if i + 1 < len(boundaries):
            content_end = boundaries[i + 1][1]
        else:
            content_end = len(text)

        remaining = text[content_start:content_end].strip()

        if remaining:
            full_text = first_line + "\n" + remaining if first_line else remaining
        else:
            full_text = first_line

        full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

        steps.append(ParsedStep(id=step_num, text=full_text))

    final_answer = ""
    if final_match:
        first_line = final_match.group(1).strip()
        content_after = text[final_match.end():].strip()
        if content_after:
            final_answer = first_line + "\n" + content_after if first_line else content_after
        else:
            final_answer = first_line
        final_answer = final_answer.strip()

    if not steps:
        warnings.append("No Step N: lines found")

    if not final_answer:
        warnings.append("No Final Answer: line found")

    if steps:
        expected = list(range(1, len(steps) + 1))
        actual = [s.id for s in steps]
        if actual != expected:
            warnings.append(
                f"Step numbering mismatch: expected {expected}, got {actual}"
            )

    if len(steps) > 15:
        warnings.append(f"Excessive steps: {len(steps)} (>15)")

    if 0 < len(steps) < 2:
        warnings.append(f"Very few steps: {len(steps)} (<2)")

    return ParsedTrace(
        raw_text=raw_text,
        steps=steps,
        final_answer=final_answer,
        parse_warnings=warnings,
    )
