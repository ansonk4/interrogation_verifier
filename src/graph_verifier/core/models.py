from __future__ import annotations

from dataclasses import dataclass, field
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


def answer_claim_matches(claim: str, answer: str) -> bool:
    claim = " ".join(claim.lower().split()).rstrip(".!?")
    answer = " ".join(answer.lower().split()).rstrip(".!?")
    return bool(answer) and claim in {
        answer,
        f"answer {answer}",
        f"answer is {answer}",
        f"the answer is {answer}",
    }
