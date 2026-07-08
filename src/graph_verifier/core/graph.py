from __future__ import annotations

from graph_verifier.core.models import Case, Edge, Graph, Node, Verification
from graph_verifier.utils.llm import LLMError, complete_json


def build_graph(case: Case) -> Graph:
    if case.graph:
        try:
            graph = Graph.from_dict(case.graph)
        except (KeyError, TypeError, ValueError) as exc:
            return Graph.failed(f"case graph malformed: {exc}")
        if not graph.nodes:
            graph.tool_debt.append("graph has no nodes")
        if not graph.coverage_claim:
            graph.coverage_claim = "missing coverage claim"
            graph.tool_debt.append("graph has no coverage claim")
        return graph

    try:
        data = complete_json(
            "graph_extract.md",
            {
                "question": case.question,
                "agent_answer": case.agent_answer,
                "agent_reasoning": case.agent_reasoning,
            },
        )
        return Graph.from_dict(data)
    except (LLMError, KeyError, TypeError, ValueError) as exc:
        return Graph.failed(f"graph extraction failed: {exc}")


def interrogate(case: Case, graph: Graph, max_rounds: int = 3) -> Graph:
    if graph.tool_debt:
        return graph
    for _ in range(max_rounds):
        vague_nodes = [node for node in graph.nodes if is_vague(node.claim)]
        weak_edges = [edge for edge in graph.edges if not edge.premise_node_ids]
        if not vague_nodes and not weak_edges:
            break
        try:
            update = complete_json(
                "interrogate.md",
                {
                    "question": case.question,
                    "agent_answer": case.agent_answer,
                    "agent_reasoning": case.agent_reasoning,
                    "graph": graph.to_dict(),
                    "vague_nodes": [node.id for node in vague_nodes],
                    "weak_edges": [edge.id for edge in weak_edges],
                },
            )
        except LLMError as exc:
            graph.tool_debt.append(f"interrogation failed: {exc}")
            break
        for node_data in update.get("new_nodes", []):
            try:
                node = Node.from_dict(node_data)
            except (KeyError, TypeError, ValueError):
                continue
            node.sources = [*node.sources, "interrogation"]
            graph.nodes.append(node)
        for edge_data in update.get("new_edges", []):
            try:
                graph.edges.append(Edge.from_dict(edge_data))
            except (KeyError, TypeError, ValueError):
                continue
        for item in update.get("debt", []):
            item_id = str(item)
            for node in graph.nodes:
                if node.id == item_id:
                    node.verification = Verification("debt", "interrogation could not ground")
            for edge in graph.edges:
                if edge.id == item_id:
                    edge.verification = Verification("debt", "interrogation could not ground")
    return graph


def mark_decisive(case: Case, graph: Graph) -> Graph:
    for node in graph.nodes:
        node.decisive = False
    for edge in graph.edges:
        edge.decisive = False
    graph.coverage_decisive = True

    try:
        review = complete_json(
            "decisiveness.md",
            {
                "question": case.question,
                "agent_answer": case.agent_answer,
                "agent_reasoning": case.agent_reasoning,
                "graph": {
                    "nodes": [{"id": node.id, "claim": node.claim} for node in graph.nodes],
                    "edges": [
                        {
                            "id": edge.id,
                            "premise_node_ids": edge.premise_node_ids,
                            "claim": edge.claim,
                            "conclusion": edge.conclusion,
                        }
                        for edge in graph.edges
                    ],
                    "coverage_claim": graph.coverage_claim,
                },
            },
        )
        node_flags = bool_map(review, "nodes")
        edge_flags = bool_map(review, "edges")
        coverage = review.get("coverage", True)
        if not isinstance(coverage, bool):
            raise TypeError("coverage must be boolean")
    except (LLMError, KeyError, TypeError, ValueError) as exc:
        graph.tool_debt.append(f"decisiveness failed: {exc}")
        return graph

    nodes_by_id = {node.id: node for node in graph.nodes}
    for node_id, decisive in node_flags.items():
        if decisive and node_id in nodes_by_id:
            nodes_by_id[node_id].decisive = True
    edges_by_id = {edge.id: edge for edge in graph.edges}
    for edge_id, decisive in edge_flags.items():
        if decisive and edge_id in edges_by_id:
            edges_by_id[edge_id].decisive = True
    graph.coverage_decisive = coverage
    return graph


def bool_map(data: dict[str, object], key: str) -> dict[str, bool]:
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    out: dict[str, bool] = {}
    for item_id, decisive in value.items():
        if not isinstance(decisive, bool):
            raise TypeError(f"{key}.{item_id} must be boolean")
        out[str(item_id)] = decisive
    return out


def is_vague(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("clearly", "obvious", "some", "about", "roughly"))
