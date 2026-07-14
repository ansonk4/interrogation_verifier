from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any


VALID = "valid"
DEBT = "debt"
REFUTED = "refuted"
QUERY_TARGET = "query_target"


@dataclass
class Verification:
    status: str = DEBT
    reason: str = ""
    evidence: Any | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Verification:
        if not data:
            return cls()
        return cls(
            status=str(data.get("status", DEBT)),
            reason=str(data.get("reason", "")),
            evidence=data.get("evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "reason": self.reason}
        if self.evidence is not None:
            out["evidence"] = self.evidence
        return out


@dataclass
class Case:
    id: str
    question: str
    agent_answer: str
    agent_reasoning: str
    expected_answer: str | None = None
    source: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    graph: dict[str, Any] | None = None
    agent_model_config: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Case:
        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            agent_answer=str(data["agent_answer"]),
            agent_reasoning=str(data.get("agent_reasoning", "")),
            expected_answer=data.get("expected_answer"),
            source=dict(data.get("source", {})),
            notes=str(data.get("notes", "")),
            graph=data.get("graph"),
            agent_model_config=(
                str(data["agent_model_config"])
                if data.get("agent_model_config") is not None
                else None
            ),
        )


@dataclass
class Node:
    id: str
    claim: str
    kind: str = "fact"
    sources: list[str] = field(default_factory=list)
    decisive: bool = False
    verification: Verification = field(default_factory=Verification)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        return cls(
            id=str(data["id"]),
            claim=str(data["claim"]),
            kind=str(data.get("kind", "fact")),
            sources=[str(source) for source in data.get("sources", [])],
            verification=Verification.from_dict(data.get("verification")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "kind": self.kind,
            "sources": self.sources,
            "decisive": self.decisive,
            "verification": self.verification.to_dict(),
        }


@dataclass
class Edge:
    id: str
    premise_node_ids: list[str]
    target_node_id: str
    claim: str = ""
    decisive: bool = False
    verification: Verification = field(default_factory=Verification)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        if "conclusion" in data:
            raise ValueError("edge uses deprecated conclusion field")
        return cls(
            id=str(data["id"]),
            premise_node_ids=[str(node_id) for node_id in data.get("premise_node_ids", [])],
            target_node_id=str(data["target_node_id"]),
            claim=str(data.get("claim", "")),
            verification=Verification.from_dict(data.get("verification")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "premise_node_ids": self.premise_node_ids,
            "target_node_id": self.target_node_id,
            "claim": self.claim,
            "decisive": self.decisive,
            "verification": self.verification.to_dict(),
        }


@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    coverage_claim: str = ""
    coverage_decisive: bool = True
    coverage_verification: Verification = field(default_factory=Verification)
    tool_debt: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Graph:
        if not isinstance(data, dict):
            raise TypeError("graph must be an object")
        coverage = data.get("coverage", {})
        coverage_claim = data.get("coverage_claim", "")
        coverage_verification = data.get("coverage_verification")
        if isinstance(coverage, dict):
            coverage_claim = coverage.get("claim", coverage_claim)
            coverage_verification = coverage.get("verification", coverage_verification)
        graph = cls(
            nodes=[Node.from_dict(node) for node in data.get("nodes", [])],
            edges=[Edge.from_dict(edge) for edge in data.get("edges", [])],
            coverage_claim=str(coverage_claim),
            coverage_verification=Verification.from_dict(coverage_verification),
            tool_debt=[str(item) for item in data.get("tool_debt", [])],
        )
        error = graph_id_error(graph)
        if error:
            raise ValueError(error)
        return graph

    @classmethod
    def failed(cls, reason: str) -> Graph:
        return cls(
            nodes=[Node(id="graph_error", claim=reason, kind="tool_failure")],
            coverage_claim="graph extraction failed",
            tool_debt=[reason],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "coverage_claim": self.coverage_claim,
            "coverage_decisive": self.coverage_decisive,
            "coverage_verification": self.coverage_verification.to_dict(),
            "tool_debt": self.tool_debt,
        }


@dataclass
class StatusResult:
    status: str
    reason: str


def graph_id_error(graph: Graph) -> str | None:
    node_ids = [node.id for node in graph.nodes]
    edge_ids = [edge.id for edge in graph.edges]
    if len(node_ids) != len(set(node_ids)):
        return "duplicate node id"
    if len(edge_ids) != len(set(edge_ids)):
        return "duplicate edge id"
    if set(node_ids).intersection(edge_ids):
        return "node and edge ids collide"
    return None


_ANSWER_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_ANSWER_PREFIXES = ("the answer is ", "answer is ", "answer ")


def answer_claim_matches(claim: str, answer: str) -> bool:
    payload = answer_claim_payload(claim)
    return bool(normalize_answer_text(answer)) and (
        answer_values_equal(claim, answer) or answer_values_equal(payload, answer)
    )


def answer_claim_payload(claim: str) -> str:
    text = _strip_terminal_punctuation(" ".join(claim.split()))
    lowered = text.casefold()
    for prefix in _ANSWER_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def answer_claim_value(claim: str) -> object | None:
    return parse_answer_value(answer_claim_payload(claim))


def answer_values_equal(left: str, right: str) -> bool:
    left_value = parse_answer_value(left)
    right_value = parse_answer_value(right)
    if left_value is not None and right_value is not None:
        return left_value == right_value
    return normalize_answer_text(left) == normalize_answer_text(right)


def parse_answer_value(text: str) -> object | None:
    text = _normalize_answer_surface(text)
    if not text:
        return None

    if text.startswith("(") and text.endswith(")"):
        parts = _split_top_level(text[1:-1], ",")
        if len(parts) > 1:
            values = tuple(parse_answer_value(part) for part in parts)
            if all(value is not None for value in values):
                return values

    latex_fraction = re.fullmatch(
        r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        text,
    )
    if latex_fraction:
        numerator = parse_answer_value(latex_fraction.group(1))
        denominator = parse_answer_value(latex_fraction.group(2))
        if isinstance(numerator, Fraction) and isinstance(denominator, Fraction):
            if denominator:
                return numerator / denominator
            return None

    fraction = re.fullmatch(rf"({_ANSWER_NUMBER})\s*/\s*({_ANSWER_NUMBER})", text)
    if fraction:
        denominator = Fraction(fraction.group(2))
        return Fraction(fraction.group(1)) / denominator if denominator else None
    if re.fullmatch(_ANSWER_NUMBER, text):
        return Fraction(text)
    return None


def normalize_answer_text(text: str) -> str:
    return " ".join(_normalize_answer_surface(text).casefold().split())


def _normalize_answer_surface(text: str) -> str:
    text = _strip_terminal_punctuation(text.strip())
    text = text.replace("$", "").replace("\\(", "").replace("\\)", "")
    text = re.sub(r"\\(?:left|right|displaystyle)\b", "", text)
    text = text.replace("\\,", " ").replace("−", "-")
    return " ".join(text.split())


def _strip_terminal_punctuation(text: str) -> str:
    text = text.rstrip()
    text = text.rstrip("!?").rstrip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    return text


def _split_top_level(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == separator and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts
