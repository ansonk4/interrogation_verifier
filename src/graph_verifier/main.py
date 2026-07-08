from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from typing import Any

from graph_verifier.core.aggregate import final_status
from graph_verifier.core.graph import build_graph, interrogate, mark_decisive
from graph_verifier.core.models import Case, Graph
from graph_verifier.core.verify import verify_graph
from graph_verifier.utils.jsonl import read_jsonl
from graph_verifier.utils.llm import LLMError, complete_json


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases")
    parser.add_argument("--mode", choices=["direct", "one-shot-graph", "interrogation"], default="interrogation")
    args = parser.parse_args(list(argv) if argv is not None else None)

    for row in read_jsonl(args.cases):
        case = Case.from_dict(row)
        if args.mode == "direct":
            output = run_direct(case)
        else:
            graph = build_graph(case)
            if args.mode == "interrogation":
                graph = interrogate(case, graph)
            mark_decisive(case, graph)
            verify_graph(case, graph)
            result = final_status(graph)
            output = compact_output(case, args.mode, graph, result.status)
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


def run_direct(case: Case) -> dict[str, Any]:
    try:
        data = complete_json(
            "direct.md",
            {
                "question": case.question,
                "agent_answer": case.agent_answer,
                "agent_reasoning": case.agent_reasoning,
            },
        )
        status = str(data.get("status", "verification_debt"))
    except LLMError as exc:
        status = "verification_debt"
        data = {"reason": str(exc)}
    return {
        "id": case.id,
        "mode": "direct",
        "status": status,
        "reason": str(data.get("reason", ""))[:120],
    }


def compact_output(case: Case, mode: str, graph: Graph, status: str) -> dict[str, Any]:
    decisive_nodes = [node for node in graph.nodes if node.decisive]
    decisive_edges = [edge for edge in graph.edges if edge.decisive]
    items = [*graph.nodes, *graph.edges]
    if graph.coverage_decisive:
        coverage_status = graph.coverage_verification.status
    else:
        coverage_status = "debt"
    valid = sum(1 for item in items if item.verification.status == "valid")
    debt = sum(1 for item in items if item.verification.status == "debt")
    refuted = sum(1 for item in items if item.verification.status == "refuted")
    if coverage_status == "valid":
        valid += 1
    elif coverage_status == "refuted":
        refuted += 1
    else:
        debt += 1
    return {
        "id": case.id,
        "mode": mode,
        "status": status,
        "decisive": {
            "nodes": len(decisive_nodes),
            "edges": len(decisive_edges),
            "coverage": graph.coverage_decisive,
        },
        "valid": valid,
        "debt": debt,
        "refuted": refuted,
    }


if __name__ == "__main__":
    raise SystemExit(main())
