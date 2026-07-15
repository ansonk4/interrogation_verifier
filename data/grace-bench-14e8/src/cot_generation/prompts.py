
from __future__ import annotations

from src.dataset.grace_dataset import GraceExample

SYSTEM_MESSAGE = (
    "You are a careful reasoning assistant. Follow the output format exactly."
)

def build_cot_prompt(example: GraceExample) -> str:
    return example.prompt
