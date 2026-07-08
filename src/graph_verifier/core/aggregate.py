from __future__ import annotations

from graph_verifier.core.models import DEBT, REFUTED, Graph, StatusResult


def final_status(graph: Graph) -> StatusResult:
    refuted = [
        item.id
        for item in [*graph.nodes, *graph.edges]
        if item.decisive and item.verification.status == REFUTED
    ]
    if graph.coverage_decisive and graph.coverage_verification.status == REFUTED:
        refuted.append("coverage")
    if refuted:
        return StatusResult("answer_refuted", "refuted: " + ",".join(refuted))

    if graph.tool_debt:
        return StatusResult("verification_debt", "tool: " + graph.tool_debt[0])

    if graph.coverage_decisive and graph.coverage_verification.status == DEBT:
        return StatusResult("coverage_debt", "coverage")

    node_debt = [
        node.id for node in graph.nodes if node.decisive and node.verification.status == DEBT
    ]
    if node_debt:
        return StatusResult("node_debt", "nodes: " + ",".join(node_debt))

    edge_debt = [
        edge.id for edge in graph.edges if edge.decisive and edge.verification.status == DEBT
    ]
    if edge_debt:
        return StatusResult("verification_debt", "edges: " + ",".join(edge_debt))

    return StatusResult("verified_reliable", "all decisive items valid")
