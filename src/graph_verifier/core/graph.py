from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import NamedTuple

from graph_verifier.core.models import (
    QUERY_TARGET,
    Case,
    Edge,
    Graph,
    Node,
    Verification,
    answer_claim_matches,
)
from graph_verifier.core.verify import check_closed_calculation, check_grounding
from graph_verifier.utils.artifacts import append_jsonl, case_name, write_json
from graph_verifier.utils.llm import LLMError, complete_agent_json, complete_json


class InterrogationUpdateResult(NamedTuple):
    accepted: bool
    rejection_reason: str

    def __bool__(self) -> bool:
        return self.accepted


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
        canonicalize_answer_node(case, graph)
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
        graph = Graph.from_dict(data)
        canonicalize_answer_node(case, graph)
        return graph
    except (LLMError, KeyError, TypeError, ValueError) as exc:
        return Graph.failed(f"graph extraction failed: {exc}")


def interrogate(case: Case, graph: Graph, artifact_dir: Path | None = None, max_rounds: int = 20) -> Graph:
    if graph.tool_debt:
        return graph
    handled: set[tuple[str, str, object]] = set()
    for round_number in range(1, max_rounds + 1):
        try:
            target_nodes, target_edges, target_coverage, review = find_interrogation_targets(case, graph)
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            graph.tool_debt.append(f"interrogation target selection failed: {exc}")
            save_interrogation_event(
                artifact_dir,
                case.id,
                {
                    "round": round_number,
                    "event": "target_selection_error",
                    "prompt": "interrogation_targets.md",
                    "error": str(exc),
                },
            )
            break
        target = select_interrogation_target(graph, target_nodes, target_edges, target_coverage, handled)
        if target is not None:
            reasons = review.get("reasons", {})
            if isinstance(reasons, dict):
                target["target_reason"] = str(
                    reasons.get(target["target_id"], reasons.get("coverage", ""))
                )
        save_interrogation_event(
            artifact_dir,
            case.id,
            {
                "round": round_number,
                "event": "target_selection",
                "prompt": "interrogation_targets.md",
                "selected": target,
                "response": review,
            },
        )
        if target is None:
            break
        target_type = str(target["target_type"])
        target_id = str(target["target_id"])
        target_key = (target_type, target_id, target_signature(graph, target_type, target_id))
        if not case.agent_model_config:
            mark_interrogation_debt(graph, target_type, target_id, "original agent unavailable")
            handled.add(target_key)
            save_interrogation_event(
                artifact_dir,
                case.id,
                {
                    "round": round_number,
                    "event": "interrogate_unavailable",
                    "target": target,
                    "reason": "case has no agent_model_config",
                },
            )
            save_interrogation_graph(artifact_dir, case.id, graph)
            continue
        rejection_reason = ""
        interrogation_error = False
        for attempt in range(1, 3):
            try:
                update = complete_agent_json(
                    "interrogate.md",
                    {
                        "question": case.question,
                        "agent_answer": case.agent_answer,
                        "agent_reasoning": case.agent_reasoning,
                        "graph": graph_prompt_data(graph),
                        "rejection_reason": rejection_reason,
                        **target,
                    },
                    case.agent_model_config,
                )
            except (LLMError, KeyError, TypeError, ValueError) as exc:
                graph.tool_debt.append(f"interrogation failed: {exc}")
                save_interrogation_event(
                    artifact_dir,
                    case.id,
                    {
                        "round": round_number,
                        "attempt": attempt,
                        "event": "interrogate_error",
                        "prompt": "interrogate.md",
                        "target": target,
                        "error": str(exc),
                    },
                )
                interrogation_error = True
                break
            accepted, rejection_reason = apply_interrogation_update(
                graph,
                update,
                question=case.question,
                target_type=target_type,
                target_id=target_id,
                answer_node_ids=matching_answer_ids(case, graph),
            )
            save_interrogation_event(
                artifact_dir,
                case.id,
                {
                    "round": round_number,
                    "attempt": attempt,
                    "event": "interrogate",
                    "prompt": "interrogate.md",
                    "target": target,
                    "accepted": accepted,
                    "rejection_reason": rejection_reason,
                    "agent_model_config": case.agent_model_config,
                    "response": update,
                },
            )
            if accepted:
                handled.add(target_key)
                break
        else:
            mark_interrogation_debt(
                graph,
                target_type,
                target_id,
                f"interrogation rejected twice: {rejection_reason}",
            )
            handled.add(target_key)
        save_interrogation_graph(artifact_dir, case.id, graph)
        if interrogation_error:
            break
    else:
        final_review: dict[str, object] | None = None
        try:
            nodes, edges, coverage, final_review = find_interrogation_targets(case, graph)
            remaining = select_interrogation_target(graph, nodes, edges, coverage, handled)
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            remaining = {"error": str(exc)}
        if remaining is not None:
            graph.tool_debt.append(f"interrogation reached max rounds: {max_rounds}")
        save_interrogation_event(
            artifact_dir,
            case.id,
            {
                "round": max_rounds,
                "event": "max_rounds",
                "max_rounds": max_rounds,
                "remaining": remaining,
                "response": final_review,
            },
        )
    return graph


def apply_interrogation_update(
    graph: Graph,
    update: dict[str, object],
    *,
    question: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    answer_node_ids: set[str] | None = None,
) -> InterrogationUpdateResult:
    candidate = deepcopy(graph)
    before_nodes = {node.id for node in graph.nodes}
    before_edges = {edge.id for edge in graph.edges}
    before_target = target_signature(graph, target_type, target_id)
    try:
        if target_type and target_id and target_id in debt_ids(update):
            mark_interrogation_debt(candidate, target_type, target_id, "interrogation could not ground")
            commit_graph(graph, candidate)
            return InterrogationUpdateResult(True, "")
        apply_candidate_update(candidate, update)
    except (KeyError, TypeError, ValueError) as exc:
        return InterrogationUpdateResult(False, f"interrogation returned an invalid update: {exc}")

    provenance_error = interrogation_provenance_error(question, graph, candidate)
    if provenance_error:
        return InterrogationUpdateResult(False, provenance_error)
    tag_interrogation_sources(graph, candidate)
    changed = candidate.to_dict() != graph.to_dict()
    if target_type and target_id:
        added_nodes = {node.id for node in candidate.nodes} - before_nodes
        added_edges = {edge.id for edge in candidate.edges} - before_edges
        resolved = target_signature(candidate, target_type, target_id) != before_target
        bounded = len(added_nodes) <= 1 and len(added_edges) <= 1
        answer_ids = answer_node_ids or {node.id for node in graph.nodes if node.kind == "answer"}
        connected = additions_support_target(
            candidate,
            target_type,
            target_id,
            added_nodes,
            added_edges,
            answer_ids,
        )
        scoped = mutations_support_target(graph, candidate, target_type, target_id, answer_ids)
        if not changed or not resolved or not bounded or not connected or not scoped:
            return InterrogationUpdateResult(False, "interrogation did not resolve target")
    if not changed:
        return InterrogationUpdateResult(False, "interrogation did not change graph")
    commit_graph(graph, candidate)
    return InterrogationUpdateResult(True, "")


def interrogation_provenance_error(question: str | None, before: Graph, after: Graph) -> str | None:
    if question is None:
        return None
    old = {node.id: node_key(node) for node in before.nodes}
    supported = {edge.target_node_id for edge in after.edges}
    for node in after.nodes:
        if node.id in old and node_key(node) == old[node.id]:
            continue
        if not node_has_provenance(question, node, supported):
            return f"interrogation node has no provenance: {node.id}"
    return None


def node_has_provenance(question: str, node: Node, supported: set[str]) -> bool:
    return node_is_independently_supported(question, node) or node.id in supported


def node_is_independently_supported(question: str, node: Node) -> bool:
    grounded = check_grounding(question, node.claim, node.sources).status == "valid" and (
        not node.claim.rstrip().endswith("?")
        or node.kind in {"question", "query_constraint", QUERY_TARGET}
    )
    return grounded or check_closed_calculation(node.claim).status == "valid"


def canonicalize_answer_node(case: Case, graph: Graph) -> None:
    answer_nodes = [node for node in graph.nodes if node.kind == "answer"]
    if len(answer_nodes) == 1 and case.agent_answer.strip():
        answer_nodes[0].claim = f"answer {case.agent_answer.strip()}"


def tag_interrogation_sources(before: Graph, after: Graph) -> None:
    old = {node.id: node_key(node) for node in before.nodes}
    for node in after.nodes:
        if node.id not in old or node_key(node) != old[node.id]:
            node.sources = list(dict.fromkeys([*node.sources, "interrogation"]))


def apply_candidate_update(graph: Graph, update: dict[str, object]) -> None:
    nodes_by_id = {node.id: node for node in graph.nodes}
    edges_by_id = {edge.id: edge for edge in graph.edges}
    node_keys = {node_key(node): node for node in graph.nodes}
    edge_keys = {edge_key(edge): edge for edge in graph.edges}
    node_aliases: dict[str, str] = {}
    edge_aliases: dict[str, str] = {}

    if "coverage_claim" in update:
        graph.coverage_claim = str(update["coverage_claim"])

    for node_data in object_list(update, "new_nodes"):
        node = Node.from_dict(node_data)
        if node.id in edges_by_id:
            raise ValueError(f"node id collides with edge: {node.id}")
        existing = nodes_by_id.get(node.id)
        if existing:
            if node_key(existing) != node_key(node):
                raise ValueError(f"conflicting node id: {node.id}")
            node_aliases[node.id] = existing.id
            continue
        equivalent = node_keys.get(node_key(node))
        if equivalent:
            node_aliases[node.id] = equivalent.id
            continue
        node.sources = list(dict.fromkeys([*node.sources, "interrogation"]))
        graph.nodes.append(node)
        nodes_by_id[node.id] = node
        node_keys[node_key(node)] = node

    def node_id(value: object) -> str:
        item_id = str(value)
        return node_aliases.get(item_id, item_id)

    for edge_data in object_list(update, "new_edges"):
        edge = Edge.from_dict(edge_data)
        if edge.id in nodes_by_id:
            raise ValueError(f"edge id collides with node: {edge.id}")
        edge.premise_node_ids = sorted(dict.fromkeys(node_id(item) for item in edge.premise_node_ids))
        edge.target_node_id = node_id(edge.target_node_id)
        validate_edge_references(edge, nodes_by_id)
        existing = edges_by_id.get(edge.id)
        if existing:
            if edge_key(existing) != edge_key(edge):
                raise ValueError(f"conflicting edge id: {edge.id}")
            edge_aliases[edge.id] = existing.id
            continue
        equivalent = edge_keys.get(edge_key(edge))
        if equivalent:
            edge_aliases[edge.id] = equivalent.id
            continue
        graph.edges.append(edge)
        edges_by_id[edge.id] = edge
        edge_keys[edge_key(edge)] = edge

    for item_data in object_list(update, "updates"):
        raw_id = str(item_data.get("id", ""))
        item_id = node_aliases.get(raw_id, edge_aliases.get(raw_id, raw_id))
        if item_id in nodes_by_id:
            node = nodes_by_id[item_id]
            if "claim" in item_data:
                node.claim = str(item_data["claim"])
            if "kind" in item_data:
                node.kind = str(item_data["kind"])
        elif item_id in edges_by_id:
            edge = edges_by_id[item_id]
            if "premise_node_ids" in item_data:
                premise_ids = item_data["premise_node_ids"]
                if not isinstance(premise_ids, list):
                    raise TypeError("premise_node_ids must be a list")
                edge.premise_node_ids = sorted(dict.fromkeys(node_id(item) for item in premise_ids))
            if "target_node_id" in item_data:
                edge.target_node_id = node_id(item_data["target_node_id"])
            if "claim" in item_data:
                edge.claim = str(item_data["claim"])
            validate_edge_references(edge, nodes_by_id)
        elif item_id == "coverage":
            coverage_claim = item_data.get("coverage_claim", item_data.get("claim"))
            if coverage_claim is not None:
                graph.coverage_claim = str(coverage_claim)
        else:
            raise ValueError(f"unknown update id: {raw_id}")

    for raw_id in debt_ids(update):
        item_id = node_aliases.get(raw_id, edge_aliases.get(raw_id, raw_id))
        if item_id == "coverage":
            graph.coverage_verification = Verification("debt", "interrogation could not ground")
        elif item_id in nodes_by_id:
            nodes_by_id[item_id].verification = Verification("debt", "interrogation could not ground")
        elif item_id in edges_by_id:
            edges_by_id[item_id].verification = Verification("debt", "interrogation could not ground")
        else:
            raise ValueError(f"unknown debt id: {raw_id}")


def object_list(data: dict[str, object], key: str) -> list[dict[str, object]]:
    value = data.get(key, []) or []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TypeError(f"{key} must be a list of objects")
    return value


def debt_ids(update: dict[str, object]) -> set[str]:
    value = update.get("debt", []) or []
    if not isinstance(value, list):
        raise TypeError("debt must be a list")
    return {str(item) for item in value}


def canonical(text: str) -> str:
    return " ".join(text.split())


def node_key(node: Node) -> tuple[str, str]:
    return canonical(node.claim), canonical(node.kind)


def edge_key(edge: Edge) -> tuple[tuple[str, ...], str, str]:
    return tuple(sorted(set(edge.premise_node_ids))), edge.target_node_id, canonical(edge.claim)


def validate_edge_references(edge: Edge, nodes_by_id: dict[str, Node]) -> None:
    if not edge.premise_node_ids:
        raise ValueError(f"edge has no premises: {edge.id}")
    if any(node_id not in nodes_by_id for node_id in edge.premise_node_ids):
        raise ValueError(f"edge has missing premise: {edge.id}")
    if edge.target_node_id not in nodes_by_id:
        raise ValueError(f"edge has missing target: {edge.id}")


def target_signature(graph: Graph, target_type: str | None, target_id: str | None) -> object:
    if target_type == "node":
        node = next((node for node in graph.nodes if node.id == target_id), None)
        incoming = tuple(sorted(edge_key(edge) for edge in graph.edges if edge.target_node_id == target_id))
        return None if node is None else (node_key(node), incoming)
    if target_type == "edge":
        edge = next((edge for edge in graph.edges if edge.id == target_id), None)
        return None if edge is None else edge_key(edge)
    if target_type == "coverage":
        return frozenset(edge_key(edge) for edge in graph.edges)
    return None


def additions_support_target(
    graph: Graph,
    target_type: str,
    target_id: str,
    added_nodes: set[str],
    added_edges: set[str],
    answer_node_ids: set[str],
) -> bool:
    if not added_nodes and not added_edges:
        return True
    if target_type == "node":
        starts = {target_id}
    elif target_type == "edge":
        edge = next((edge for edge in graph.edges if edge.id == target_id), None)
        starts = {edge.target_node_id} if edge else set()
    else:
        starts = answer_node_ids
    nodes, edges = answer_ancestry(graph, starts)
    return added_nodes <= nodes and added_edges <= edges


def mutations_support_target(
    before: Graph,
    after: Graph,
    target_type: str,
    target_id: str,
    answer_node_ids: set[str],
) -> bool:
    before_nodes = {node.id: node_key(node) for node in before.nodes}
    before_edges = {edge.id: edge_key(edge) for edge in before.edges}
    changed_nodes = {
        node.id for node in after.nodes if node.id in before_nodes and node_key(node) != before_nodes[node.id]
    }
    changed_edges = {
        edge.id for edge in after.edges if edge.id in before_edges and edge_key(edge) != before_edges[edge.id]
    }
    if target_type == "node":
        incoming = {
            edge.id
            for edge in [*before.edges, *after.edges]
            if edge.target_node_id == target_id
        }
        return (
            len(changed_nodes) + len(changed_edges) <= 1
            and changed_nodes <= {target_id}
            and changed_edges <= incoming
        )
    if target_type == "edge":
        return not changed_nodes and changed_edges <= {target_id}
    _, allowed_edges = answer_ancestry(before, answer_node_ids)
    return not changed_nodes and len(changed_edges) <= 1 and changed_edges <= allowed_edges


def answer_ancestry(graph: Graph, starts: set[str]) -> tuple[set[str], set[str]]:
    incoming: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.target_node_id, []).append(edge)
    nodes = set(starts)
    edges: set[str] = set()
    stack = list(starts)
    while stack:
        node_id = stack.pop()
        for edge in incoming.get(node_id, []):
            if edge.id in edges:
                continue
            edges.add(edge.id)
            for premise_id in edge.premise_node_ids:
                if premise_id not in nodes:
                    nodes.add(premise_id)
                    stack.append(premise_id)
    return nodes, edges


def mark_interrogation_debt(graph: Graph, target_type: str, target_id: str, reason: str) -> None:
    if target_type == "coverage":
        graph.coverage_verification = Verification("debt", reason)
        return
    items = graph.nodes if target_type == "node" else graph.edges
    for item in items:
        if item.id == target_id:
            item.verification = Verification("debt", reason)
            return


def commit_graph(graph: Graph, candidate: Graph) -> None:
    graph.nodes = candidate.nodes
    graph.edges = candidate.edges
    graph.coverage_claim = candidate.coverage_claim
    graph.coverage_decisive = candidate.coverage_decisive
    graph.coverage_verification = candidate.coverage_verification
    graph.tool_debt = candidate.tool_debt


def select_interrogation_target(
    graph: Graph,
    nodes: list[Node],
    edges: list[Edge],
    coverage: bool,
    handled: set[tuple[str, str, object]],
) -> dict[str, object] | None:
    candidates: list[dict[str, object]] = []
    for node in nodes:
        candidates.append(
            {
                "target_type": "node",
                "target_id": node.id,
                "target_node": node_prompt_data(node),
                "target_edge": None,
                "target_coverage": False,
            }
        )
    for edge in edges:
        candidates.append(
            {
                "target_type": "edge",
                "target_id": edge.id,
                "target_node": None,
                "target_edge": edge_prompt_data(edge),
                "target_coverage": False,
            }
        )
    if coverage:
        candidates.append(
            {
                "target_type": "coverage",
                "target_id": "coverage",
                "target_node": None,
                "target_edge": None,
                "target_coverage": True,
            }
        )
    handled_ids = {(target_type, target_id) for target_type, target_id, _ in handled}
    for candidate in candidates:
        identity = str(candidate["target_type"]), str(candidate["target_id"])
        if identity not in handled_ids:
            return candidate
    for candidate in candidates:
        target_type = str(candidate["target_type"])
        target_id = str(candidate["target_id"])
        state = target_signature(graph, target_type, target_id)
        if (target_type, target_id, state) not in handled:
            return candidate
    return None


def node_prompt_data(node: Node) -> dict[str, object]:
    return {"id": node.id, "claim": node.claim, "kind": node.kind, "sources": node.sources}


def edge_prompt_data(edge: Edge) -> dict[str, object]:
    return {
        "id": edge.id,
        "premise_node_ids": edge.premise_node_ids,
        "target_node_id": edge.target_node_id,
        "claim": edge.claim,
    }


def graph_prompt_data(graph: Graph) -> dict[str, object]:
    return {
        "nodes": [node_prompt_data(node) for node in graph.nodes],
        "edges": [edge_prompt_data(edge) for edge in graph.edges],
        "coverage_claim": graph.coverage_claim,
    }


def find_interrogation_targets(case: Case, graph: Graph) -> tuple[list[Node], list[Edge], bool, dict[str, object]]:
    answer_ids = matching_answer_ids(case, graph)
    answer_path_nodes, _ = answer_ancestry(graph, answer_ids)
    nodes_by_id = {node.id: node for node in graph.nodes}
    answer_edges = [edge for edge in graph.edges if edge.target_node_id in answer_ids]
    query_targets = [node for node in graph.nodes if node.kind == QUERY_TARGET]
    ungrounded_query_targets = [
        node
        for node in query_targets
        if check_grounding(case.question, node.claim, node.sources).status != "valid"
    ]

    def target_query_gap(reason: str) -> tuple[list[Node], list[Edge], bool, dict[str, object]]:
        if len(answer_edges) == 1:
            edge = answer_edges[0]
            return (
                [],
                [edge],
                False,
                {
                    "nodes": [],
                    "edges": [edge.id],
                    "coverage": False,
                    "reasons": {edge.id: reason},
                },
            )
        return (
            [],
            [],
            True,
            {
                "nodes": [],
                "edges": [],
                "coverage": True,
                "reasons": {"coverage": reason},
            },
        )

    if answer_ids and not query_targets:
        return target_query_gap("missing dedicated query target")
    if ungrounded_query_targets:
        node = ungrounded_query_targets[0]
        return (
            [node],
            [],
            False,
            {
                "nodes": [node.id],
                "edges": [],
                "coverage": False,
                "reasons": {node.id: "query target is not an exact quote from the question"},
            },
        )
    if answer_ids and not {node.id for node in query_targets}.intersection(answer_path_nodes):
        return target_query_gap("query target is not connected to the answer")

    supported_node_ids = {edge.target_node_id for edge in graph.edges}
    ungrounded_query_ids = [
        node.id
        for node in graph.nodes
        if node.kind in {"query_constraint", QUERY_TARGET}
        and check_grounding(case.question, node.claim, node.sources).status != "valid"
    ]
    unsupported_root_ids = [
        node.id
        for node in graph.nodes
        if node.id in answer_path_nodes
        and node.kind != "answer"
        and not node_has_provenance(case.question, node, supported_node_ids)
    ]
    heuristic_node_ids = merge_ids(
        [node.id for node in graph.nodes if is_vague(node.claim)],
        ungrounded_query_ids,
    )
    heuristic_node_ids = merge_ids(heuristic_node_ids, unsupported_root_ids)
    node_ids = {node.id for node in graph.nodes}
    heuristic_edge_ids = [
        edge.id
        for edge in graph.edges
        if not edge.premise_node_ids or not edge.target_node_id or edge.target_node_id not in node_ids
    ]
    review = complete_json(
        "interrogation_targets.md",
        {
            "question": case.question,
            "agent_answer": case.agent_answer,
            "agent_reasoning": case.agent_reasoning,
            "graph": graph_prompt_data(graph),
            "heuristic_targets": {
                "nodes": heuristic_node_ids,
                "edges": heuristic_edge_ids,
            },
        },
    )
    reasons = review.get("reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}
    for node_id in ungrounded_query_ids:
        reasons.setdefault(node_id, "query constraint is not an exact quote from the question")
    for node_id in unsupported_root_ids:
        reasons.setdefault(node_id, "node has no grounded, computed, or derived provenance")
    review = {**review, "reasons": reasons}
    node_ids = merge_ids(id_list(review, "nodes"), heuristic_node_ids)
    edge_ids = merge_ids(id_list(review, "edges"), heuristic_edge_ids)
    coverage = review.get("coverage", False)
    if not isinstance(coverage, bool):
        raise TypeError("coverage must be boolean")
    edges_by_id = {edge.id: edge for edge in graph.edges}
    if coverage and not node_ids and not edge_ids:
        answer_edges = [
            edge for edge in graph.edges if edge.target_node_id in matching_answer_ids(case, graph)
        ]
        if len(answer_edges) == 1:
            edge_ids = [answer_edges[0].id]
            coverage = False
            review = {**review, "coverage_anchor_edge": answer_edges[0].id}
    return (
        [nodes_by_id[node_id] for node_id in node_ids if node_id in nodes_by_id],
        [edges_by_id[edge_id] for edge_id in edge_ids if edge_id in edges_by_id],
        coverage,
        review,
    )


def save_interrogation_event(artifact_dir: Path | None, case_id: str, data: dict[str, object]) -> None:
    if artifact_dir is None:
        return
    append_jsonl(artifact_dir / f"{case_name(case_id)}.interrogation.jsonl", data)


def save_interrogation_graph(artifact_dir: Path | None, case_id: str, graph: Graph) -> None:
    if artifact_dir is None:
        return
    write_json(artifact_dir / f"{case_name(case_id)}.graph.interrogate_latest.json", graph.to_dict())


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
                    "nodes": [
                        {"id": node.id, "claim": node.claim, "kind": node.kind}
                        for node in graph.nodes
                    ],
                    "edges": [
                        {
                            "id": edge.id,
                            "premise_node_ids": edge.premise_node_ids,
                            "target_node_id": edge.target_node_id,
                            "claim": edge.claim,
                        }
                        for edge in graph.edges
                    ],
                    "coverage_claim": graph.coverage_claim,
                },
            },
        )
        node_flags = bool_map(review, "nodes")
        edge_flags = bool_map(review, "edges")
        if not isinstance(review.get("coverage", True), bool):
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
    prune_decisive_to_answer(case, graph)
    complete_decisive_support(case, graph)
    graph.coverage_decisive = True
    return graph


def prune_decisive_to_answer(case: Case, graph: Graph) -> None:
    answer_ids = matching_answer_ids(case, graph)
    incoming: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        if edge.decisive:
            incoming.setdefault(edge.target_node_id, []).append(edge)
    decisive_nodes = set(answer_ids)
    decisive_edges: set[str] = set()
    stack = list(answer_ids)
    while stack:
        node_id = stack.pop()
        for edge in incoming.get(node_id, []):
            if edge.id in decisive_edges:
                continue
            decisive_edges.add(edge.id)
            decisive_nodes.add(edge.target_node_id)
            for premise_id in edge.premise_node_ids:
                if premise_id not in decisive_nodes:
                    decisive_nodes.add(premise_id)
                    stack.append(premise_id)
    for node in graph.nodes:
        node.decisive = node.id in decisive_nodes
    for edge in graph.edges:
        edge.decisive = edge.id in decisive_edges


def complete_decisive_support(case: Case, graph: Graph) -> None:
    nodes_by_id = {node.id: node for node in graph.nodes}
    incoming: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.target_node_id, []).append(edge)
    stack = [node.id for node in graph.nodes if node.decisive]
    seen: set[str] = set()
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = nodes_by_id[node_id]
        if node_is_independently_supported(case.question, node):
            continue
        support = incoming.get(node_id, [])
        if any(edge.decisive for edge in support) or len(support) != 1:
            continue
        edge = support[0]
        edge.decisive = True
        for premise_id in edge.premise_node_ids:
            premise = nodes_by_id.get(premise_id)
            if premise is not None and not premise.decisive:
                premise.decisive = True
                stack.append(premise_id)


def matching_answer_ids(case: Case, graph: Graph) -> set[str]:
    return {
        node.id
        for node in graph.nodes
        if node.kind == "answer"
        and answer_claim_matches(node.claim, case.agent_answer)
    }


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


def id_list(data: dict[str, object], key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return [str(item) for item in value]


def merge_ids(primary: list[str], fallback: list[str]) -> list[str]:
    out = list(primary)
    for item_id in fallback:
        if item_id not in out:
            out.append(item_id)
    return out


def is_vague(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("clearly", "obvious", "some", "about", "roughly"))
