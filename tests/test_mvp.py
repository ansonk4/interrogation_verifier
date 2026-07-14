import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Lock
from unittest.mock import patch

from graph_verifier.core.aggregate import final_status
from graph_verifier.core.graph import (
    InterrogationState,
    apply_interrogation_update,
    canonicalize_answer_node,
    interrogate,
    mark_decisive,
    select_verification_target,
    target_signature,
)
from graph_verifier.core.models import (
    QUERY_TARGET,
    Case,
    Edge,
    Graph,
    Node,
    Verification,
    answer_claim_matches,
)
from graph_verifier.core.verify import (
    ClaimCheck,
    check_answer_edge,
    check_closed_calculation,
    check_grounding,
    safe_eval,
    verify_coverage,
    verify_edge_with_llm,
    verify_graph,
)
from graph_verifier.main import (
    compact_output,
    process_cases,
    run_interrogation_verification,
    validate_case_names,
)
from graph_verifier.utils.llm import LLMError, complete_json


QUESTION = (
    "Provider A costs 100 dollars for 40 units. "
    "Provider B costs 90 dollars for 30 units. Which provider is cheaper?"
)
AGENT_MODEL_CONFIG = "model/openrouter/hy3.json"


def complete_graph(comparison: str = "2.5 < 3", answer: str = "A") -> Graph:
    return Graph(
        nodes=[
            Node("query", "provider is cheaper", QUERY_TARGET, ["question"]),
            Node("n1", "100 / 40 = 2.5", "calculation"),
            Node("n2", "90 / 30 = 3", "calculation"),
            Node("n3", comparison, "comparison"),
            Node("n4", f"answer {answer}", "answer"),
        ],
        edges=[
            Edge("e1", ["n1", "n2"], "n3", comparison),
            Edge(
                "e2",
                ["query", "n1", "n2", "n3"],
                "n4",
                f"Provider A costs 2.5 per unit, Provider B costs 3, and {comparison}, so answer {answer}",
            ),
        ],
        coverage_claim="n1,n2 -> e1 -> n3 -> e2 -> n4",
    )


def status_for(
    graph: Graph,
    answer: str = "A",
    question: str = QUESTION,
    review: dict | None = None,
    reviewer_error: Exception | None = None,
    edge_statuses: dict[str, str] | None = None,
) -> str:
    case = Case("case", question, answer, "")
    if review is None:
        review = {
            "nodes": {node.id: True for node in graph.nodes},
            "edges": {edge.id: True for edge in graph.edges},
            "coverage": True,
            "reasons": {},
        }

    def fake_complete_json(prompt_name, data):
        assert prompt_name == "decisiveness.md"
        if reviewer_error:
            raise reviewer_error
        return review

    with patch("graph_verifier.core.graph.complete_json", side_effect=fake_complete_json):
        mark_decisive(case, graph)

    edge_statuses = {"e2": "valid"} if edge_statuses is None else edge_statuses

    def check_edge(premises, edge, target):
        status = edge_statuses.get(edge.id, "debt")
        return ClaimCheck(status, "test edge verification")

    verify_graph(case, graph, check_edge)
    return final_status(graph).status


def test_complete_correct_graph():
    assert status_for(complete_graph()) == "verified_reliable"


def test_correct_answer_incomplete_graph_gets_coverage_debt():
    graph = Graph(
        nodes=[
            Node("n1", "100 / 40 = 2.5", "calculation"),
            Node("n2", "answer A", "answer"),
        ],
        edges=[Edge("e1", ["n1"], "n2", "A is the answer")],
        coverage_claim="n1 -> e1 -> n2",
    )
    assert status_for(graph) == "coverage_debt"


def test_correct_answer_wrong_premise_is_not_reliable():
    graph = complete_graph()
    graph.nodes[0].claim = "100 / 50 = 2.5"
    assert status_for(graph) != "verified_reliable"


def test_incorrect_arithmetic_decisive_node():
    graph = Graph(
        nodes=[
            Node("n1", "100 / 40 = 3", "calculation"),
            Node("n2", "answer A", "answer"),
        ],
        edges=[Edge("e1", ["n1"], "n2", "3 < 4")],
        coverage_claim="n1 -> e1 -> n2",
    )
    assert status_for(graph) == "answer_refuted"


def test_refuted_comparison_supports_final_answer():
    assert status_for(complete_graph("3 < 2.5", "B"), answer="B") == "answer_refuted"


def test_irrelevant_true_edge_does_not_support_answer():
    graph = Graph(
        nodes=[
            Node("n1", "100 / 40 = 2.5", "calculation"),
            Node("n2", "answer A", "answer"),
        ],
        edges=[Edge("e1", ["n1"], "n2", "40 < 100")],
        coverage_claim="n1 -> e1 -> n2",
    )
    assert status_for(graph) == "coverage_debt"


def test_edge_cannot_use_numbers_missing_from_premises():
    graph = Graph(
        nodes=[
            Node("n1", "100 / 40 = 2.5", "calculation"),
            Node("n2", "answer A", "answer"),
        ],
        edges=[Edge("e1", ["n1"], "n2", "2.5 < 3")],
        coverage_claim="n1 -> e1 -> n2",
    )
    assert status_for(graph) == "coverage_debt"


def test_decisive_edge_forces_its_premises_decisive():
    graph = complete_graph()
    review = {
        "nodes": {"n1": True, "n2": False, "n3": True, "n4": True},
        "edges": {"e1": True, "e2": True},
        "coverage": True,
        "reasons": {},
    }
    assert status_for(graph, review=review) == "verified_reliable"
    assert all(node.decisive for node in graph.nodes)


def test_decisive_node_pulls_in_sole_support_edge():
    case = Case("case", "Given fraction equality, find x.", "1", "")
    graph = Graph(
        nodes=[
            Node("query", "x", QUERY_TARGET),
            Node("given", "Given fraction equality", "query_constraint"),
            Node("derived", "denominators are nonzero", "constraint"),
            Node("answer", "answer 1", "answer"),
        ],
        edges=[
            Edge("support", ["given"], "derived", "fractions require nonzero denominators"),
            Edge("finish", ["query", "derived"], "answer", "therefore answer 1"),
        ],
    )
    review = {
        "nodes": {"query": True, "given": True, "derived": True, "answer": True},
        "edges": {"support": False, "finish": True},
        "coverage": True,
        "reasons": {},
    }
    with patch("graph_verifier.core.graph.complete_json", return_value=review):
        mark_decisive(case, graph)

    assert all(node.decisive for node in graph.nodes)
    assert all(edge.decisive for edge in graph.edges)

    verify_graph(case, graph, lambda *args: ClaimCheck("valid", "support"))
    assert final_status(graph).status == "verified_reliable"


def test_interrogation_unsupported_premise_is_debt_not_refutation():
    graph = Graph(
        nodes=[
            Node("n1", "A has a hidden discount of 10", "premise", ["interrogation"]),
            Node("n2", "answer A", "answer"),
        ],
        edges=[Edge("e1", ["n1"], "n2", "A is the answer")],
        coverage_claim="n1 -> e1 -> n2",
    )
    assert status_for(graph) != "answer_refuted"
    assert status_for(graph) != "verified_reliable"


def test_interrogation_without_original_agent_marks_debt():
    case = Case("case", QUESTION, "A", "A is cheaper.")
    graph = Graph(nodes=[Node("n1", "unsupported claim", "claim")])
    target_calls = 0

    def fake_reviewer(prompt_name, data):
        nonlocal target_calls
        assert prompt_name == "interrogation_targets.md"
        target_calls += 1
        if target_calls == 1:
            return {"nodes": ["n1"], "edges": [], "coverage": False, "reasons": {}}
        return {"nodes": [], "edges": [], "coverage": False, "reasons": {}}

    with (
        patch("graph_verifier.core.graph.complete_json", side_effect=fake_reviewer),
        patch("graph_verifier.core.graph.complete_agent_json") as agent_call,
    ):
        interrogate(case, graph)

    agent_call.assert_not_called()
    assert graph.nodes[0].claim == "unsupported claim"
    assert graph.nodes[0].verification.reason == "original agent unavailable"


def test_coverage_gap_anchors_to_only_answer_edge():
    case = Case("case", "Find the least positive integer x with x > 1", "2", "x > 1, so 2")
    graph = Graph(
        nodes=[
            Node("query", "x", QUERY_TARGET),
            Node("n1", "x > 1", "calculation"),
            Node("answer", "answer 2", "answer"),
        ],
        edges=[Edge("answer_edge", ["query", "n1"], "answer", "therefore answer 2")],
        coverage_claim="n1 -> answer_edge -> answer",
    )
    target_calls = 0

    def fake_reviewer(prompt_name, data):
        nonlocal target_calls
        assert prompt_name == "interrogation_targets.md"
        target_calls += 1
        if target_calls == 1:
            return {
                "nodes": [],
                "edges": [],
                "coverage": True,
                "reasons": {"coverage": "missing positive-integer constraint"},
            }
        return {"nodes": [], "edges": [], "coverage": False, "reasons": {}}

    with (
        patch("graph_verifier.core.graph.complete_json", side_effect=fake_reviewer),
        patch("graph_verifier.core.graph.complete_agent_json") as agent_call,
    ):
        interrogate(case, graph)

    agent_call.assert_not_called()
    assert graph.edges[0].verification.reason == "original agent unavailable"
    assert graph.coverage_verification.reason == ""


def test_interrogation_uses_llm_targets():
    case = Case(
        "case",
        QUESTION,
        "A",
        "A is cheaper.",
        agent_model_config=AGENT_MODEL_CONFIG,
    )
    graph = Graph(
        nodes=[
            Node("query", "provider is cheaper", QUERY_TARGET),
            Node("n1", "Provider A is cheaper", "claim"),
            Node("n2", "answer A", "answer"),
        ],
        edges=[Edge("e1", ["query", "n1"], "n2", "A is the answer")],
        coverage_claim="n1 -> e1 -> n2",
    )
    calls = []
    target_calls = 0

    def fake_reviewer(prompt_name, data):
        nonlocal target_calls
        calls.append((prompt_name, data))
        assert prompt_name == "interrogation_targets.md"
        target_calls += 1
        if target_calls == 1:
            return {"nodes": ["n1"], "edges": ["e1"], "coverage": True, "reasons": {}}
        return {"nodes": [], "edges": [], "coverage": False, "reasons": {}}

    def fake_agent(prompt_name, data, model_config):
        calls.append((prompt_name, data))
        assert prompt_name == "interrogate.md"
        assert model_config == AGENT_MODEL_CONFIG
        assert data["target_type"] in {"node", "edge", "coverage"}
        assert data["target_id"] in {"n1", "e1", "coverage"}
        if len([call for call, _ in calls if call == "interrogate.md"]) == 1:
            assert data["rejection_reason"] == ""
        else:
            assert data["rejection_reason"] == "interrogation did not resolve target"
        return {
            "new_nodes": (
                [{"id": "n3", "claim": "100 / 40 = 2.5", "kind": "calculation"}]
                if data["target_type"] == "node"
                else []
            ),
            "new_edges": [],
            "debt": [],
        }

    with (
        patch("graph_verifier.core.graph.complete_json", side_effect=fake_reviewer),
        patch("graph_verifier.core.graph.complete_agent_json", side_effect=fake_agent),
    ):
        interrogate(case, graph)

    assert [prompt_name for prompt_name, _ in calls] == [
        "interrogation_targets.md",
        "interrogate.md",
        "interrogate.md",
        "interrogation_targets.md",
    ]
    assert [data["target_type"] for prompt_name, data in calls if prompt_name == "interrogate.md"] == [
        "node",
        "node",
    ]
    assert [node.id for node in graph.nodes] == ["query", "n1", "n2"]
    assert graph.nodes[1].verification.reason == (
        "interrogation rejected twice: interrogation did not resolve target"
    )


def test_interrogation_update_mutates_existing_targets():
    graph = Graph(
        nodes=[
            Node("n1", "some vague support", "claim"),
            Node("n_answer", "answer A", "answer"),
        ],
        edges=[Edge("e1", ["n1"], "n_answer", "A is the answer")],
        coverage_claim="n1 -> n_answer",
    )

    changed = apply_interrogation_update(
        graph,
        {
            "new_nodes": [
                {"id": "n2", "claim": "100 / 40 = 2.5", "kind": "calculation"},
                {"id": "n3", "claim": "answer A", "kind": "answer"},
            ],
            "new_edges": [],
            "updates": [
                {"id": "n1", "claim": "Provider A unit price is 2.5", "kind": "calculation"},
                {
                    "id": "e1",
                    "premise_node_ids": ["n1", "n2"],
                    "target_node_id": "n3",
                    "claim": "2.5 supports choosing A",
                },
                {"id": "coverage", "coverage_claim": "n1,n2 -> e1 -> n3"},
            ],
            "debt": [],
        },
    )

    assert changed
    assert graph.nodes[0].claim == "Provider A unit price is 2.5"
    assert graph.nodes[0].kind == "calculation"
    assert graph.nodes[2].id == "n2"
    assert graph.edges[0].premise_node_ids == ["n1", "n2"]
    assert graph.edges[0].target_node_id == "n_answer"
    assert graph.edges[0].claim == "2.5 supports choosing A"
    assert graph.coverage_claim == "n1,n2 -> e1 -> n3"


def test_interrogation_target_selection_sees_updates():
    case = Case(
        "case",
        QUESTION,
        "A",
        "A is cheaper.",
        agent_model_config=AGENT_MODEL_CONFIG,
    )
    graph = Graph(
        nodes=[
            Node("query", "provider is cheaper", QUERY_TARGET),
            Node("n1", "Provider A costs 100 dollars for 40 units", "premise"),
            Node("n2", "answer A", "answer"),
        ],
        edges=[Edge("e1", ["query", "n1"], "n2", "A is cheaper")],
        coverage_claim="n1 -> e1 -> n2",
    )
    target_calls = 0

    def fake_reviewer(prompt_name, data):
        nonlocal target_calls
        assert prompt_name == "interrogation_targets.md"
        target_calls += 1
        if target_calls == 1:
            return {"nodes": [], "edges": ["e1"], "coverage": False, "reasons": {}}
        assert data["graph"]["edges"][0]["premise_node_ids"] == ["n1", "n3", "query"]
        assert data["graph"]["edges"][0]["target_node_id"] == "n2"
        assert data["graph"]["edges"][0]["claim"] == "Provider A is cheaper, so answer A"
        return {"nodes": [], "edges": [], "coverage": False, "reasons": {}}

    def fake_agent(prompt_name, data, model_config):
        assert prompt_name == "interrogate.md"
        assert model_config == AGENT_MODEL_CONFIG
        return {
            "new_nodes": [
                {"id": "n3", "claim": "90 / 30 = 3", "kind": "calculation"},
            ],
            "new_edges": [],
            "updates": [
                {
                    "id": "e1",
                    "premise_node_ids": ["query", "n1", "n3"],
                    "target_node_id": "n2",
                    "claim": "Provider A is cheaper, so answer A",
                }
            ],
            "debt": [],
        }

    with (
        patch("graph_verifier.core.graph.complete_json", side_effect=fake_reviewer),
        patch("graph_verifier.core.graph.complete_agent_json", side_effect=fake_agent),
    ):
        interrogate(case, graph)

    assert target_calls == 2
    assert graph.edges[0].premise_node_ids == ["n1", "n3", "query"]
    assert graph.edges[0].target_node_id == "n2"
    assert graph.nodes[-1].sources == ["interrogation"]


def test_interrogation_repairs_ungrounded_query_target():
    question = (
        r"For what value of $x$ will $\frac{3+x}{5+x}$ and "
        r"$\frac{1+x}{2+x}$ be equal?"
    )
    exact_claim = r"$\frac{3+x}{5+x}$ and $\frac{1+x}{2+x}$ be equal"
    case = Case(
        "case",
        question,
        "1",
        "Set the fractions equal and solve x = 1.",
        agent_model_config=AGENT_MODEL_CONFIG,
    )
    graph = Graph(
        nodes=[
            Node("n1", "frac{3+x}{5+x} = frac{1+x}{2+x}", QUERY_TARGET),
            Node("answer", "answer 1", "answer"),
        ],
        edges=[Edge("e1", ["n1"], "answer", "solve for x")],
    )
    reviewer_calls = 0
    agent_calls = 0

    def fake_reviewer(prompt_name, data):
        nonlocal reviewer_calls
        assert prompt_name == "interrogation_targets.md"
        reviewer_calls += 1
        assert data["heuristic_targets"]["nodes"] == []
        return {"nodes": [], "edges": [], "coverage": False, "reasons": {}}

    def fake_agent(prompt_name, data, model_config):
        nonlocal agent_calls
        assert prompt_name == "interrogate.md"
        assert model_config == AGENT_MODEL_CONFIG
        assert data["target_type"] == "node"
        assert data["target_id"] == "n1"
        assert data["target_reason"] == "query target is not an exact quote from the question"
        agent_calls += 1
        return {
            "new_nodes": [],
            "new_edges": [],
            "updates": [{"id": "n1", "claim": exact_claim, "kind": QUERY_TARGET}],
            "debt": [],
        }

    with (
        patch("graph_verifier.core.graph.complete_json", side_effect=fake_reviewer),
        patch("graph_verifier.core.graph.complete_agent_json", side_effect=fake_agent),
    ):
        interrogate(case, graph)

    assert reviewer_calls == 1
    assert agent_calls == 1
    assert graph.nodes[0].claim == exact_claim
    assert graph.nodes[0].sources == ["interrogation"]


def test_interrogation_repairs_unsupported_root_on_answer_path():
    question = (
        r"For what value of $x$ will $\frac{3+x}{5+x}$ and "
        r"$\frac{1+x}{2+x}$ be equal?"
    )
    case = Case(
        "case",
        question,
        "1",
        "Set the fractions equal and solve x = 1.",
        agent_model_config=AGENT_MODEL_CONFIG,
    )
    graph = Graph(
        nodes=[
            Node(
                "n1",
                r"$\frac{3+x}{5+x}$ and $\frac{1+x}{2+x}$ be equal",
                QUERY_TARGET,
            ),
            Node("n2", "Denominators 5+x and 2+x are nonzero", "constraint"),
            Node("n3", "(3+x)(2+x) = (1+x)(5+x)", "calculation"),
            Node("answer", "answer 1", "answer"),
            Node("unused", "z is positive", "constraint"),
        ],
        edges=[
            Edge("e1", ["n1", "n2"], "n3", "cross multiply"),
            Edge("e2", ["n3"], "answer", "solve x=1"),
        ],
    )
    reviewer_calls = 0

    def fake_reviewer(prompt_name, data):
        nonlocal reviewer_calls
        assert prompt_name == "interrogation_targets.md"
        reviewer_calls += 1
        assert data["heuristic_targets"]["nodes"] == (["n2"] if reviewer_calls == 1 else [])
        return {"nodes": [], "edges": [], "coverage": False, "reasons": {}}

    def fake_agent(prompt_name, data, model_config):
        assert prompt_name == "interrogate.md"
        assert model_config == AGENT_MODEL_CONFIG
        assert data["target_type"] == "node"
        assert data["target_id"] == "n2"
        assert data["target_reason"] == "node has no grounded, computed, or derived provenance"
        return {
            "new_nodes": [],
            "new_edges": [
                {
                    "id": "e0",
                    "premise_node_ids": ["n1"],
                    "target_node_id": "n2",
                    "claim": "fractions are defined only when their denominators are nonzero",
                }
            ],
            "updates": [],
            "debt": [],
        }

    with (
        patch("graph_verifier.core.graph.complete_json", side_effect=fake_reviewer),
        patch("graph_verifier.core.graph.complete_agent_json", side_effect=fake_agent),
    ):
        interrogate(case, graph)

    assert reviewer_calls == 2
    assert graph.edges[-1].id == "e0"
    assert graph.edges[-1].target_node_id == "n2"


def test_changed_edge_can_be_refined():
    case = Case(
        "case",
        QUESTION,
        "A",
        "A is cheaper.",
        agent_model_config=AGENT_MODEL_CONFIG,
    )
    graph = Graph(
        nodes=[
            Node("query", "provider is cheaper", QUERY_TARGET),
            Node("given", "100 / 40 = 2.5"),
            Node("answer", "answer A", "answer"),
        ],
        edges=[Edge("edge", ["query", "given"], "answer", "therefore A")],
    )
    target_calls = 0
    agent_calls = 0

    def fake_reviewer(prompt_name, data):
        nonlocal target_calls
        target_calls += 1
        if target_calls <= 2:
            return {"nodes": [], "edges": ["edge"], "coverage": False, "reasons": {}}
        return {"nodes": [], "edges": [], "coverage": False, "reasons": {}}

    def fake_agent(prompt_name, data, model_config):
        nonlocal agent_calls
        agent_calls += 1
        if agent_calls == 1:
            return {
                "new_nodes": [
                    {
                        "id": "constraint",
                        "claim": "Provider B costs 90 dollars for 30 units",
                        "kind": "premise",
                    }
                ],
                "new_edges": [],
                "updates": [
                    {
                        "id": "edge",
                        "premise_node_ids": ["query", "given", "constraint"],
                        "target_node_id": "answer",
                        "claim": "therefore A",
                    }
                ],
                "debt": [],
            }
        return {
            "new_nodes": [],
            "new_edges": [],
            "updates": [
                {
                    "id": "edge",
                    "premise_node_ids": ["query", "given", "constraint"],
                    "target_node_id": "answer",
                    "claim": "2.5 is the lower unit price, so answer A",
                }
            ],
            "debt": [],
        }

    with (
        patch("graph_verifier.core.graph.complete_json", side_effect=fake_reviewer),
        patch("graph_verifier.core.graph.complete_agent_json", side_effect=fake_agent),
    ):
        interrogate(case, graph)

    assert agent_calls == 2
    assert graph.edges[0].claim == "2.5 is the lower unit price, so answer A"


def test_interrogation_rejects_node_without_provenance():
    question = "What is the least positive integer value of x for which x > 1?"
    graph = Graph(
        nodes=[
            Node("n2", "x > 1", "calculation"),
            Node("n4", "least positive integer value of x", "query_constraint"),
            Node("answer", "answer 2", "answer"),
        ],
        edges=[Edge("e2", ["n2", "n4"], "answer", "therefore answer 2")],
    )
    update = {
        "new_nodes": [
            {
                "id": "unsupported",
                "claim": "the least positive integer greater than 1 is 2",
                "kind": "premise",
            }
        ],
        "new_edges": [],
        "updates": [
            {
                "id": "e2",
                "premise_node_ids": ["n2", "n4", "unsupported"],
                "target_node_id": "answer",
                "claim": "therefore answer 2",
            }
        ],
        "debt": [],
    }

    result = apply_interrogation_update(
        graph,
        update,
        question=question,
        target_type="edge",
        target_id="e2",
    )
    assert not result
    assert result.rejection_reason == "interrogation node has no provenance: unsupported"
    assert [node.id for node in graph.nodes] == ["n2", "n4", "answer"]
    assert graph.edges[0].premise_node_ids == ["n2", "n4"]
    assert graph.edges[0].verification.reason == ""


def test_interrogation_retries_orphan_node_with_rejection_reason():
    case = Case(
        "case",
        "Given x = 1 and the denominators are 5+x and 2+x, determine whether they are nonzero.",
        "1",
        "x = 1; the denominators are nonzero.",
        agent_model_config=AGENT_MODEL_CONFIG,
    )
    graph = Graph(
        nodes=[
            Node("query", "whether they are nonzero", QUERY_TARGET),
            Node("n1", "the denominators are 5+x and 2+x", "premise"),
            Node("n4", "x = 1", "calculation"),
            Node("answer", "answer 1", "answer"),
        ],
        edges=[Edge("e4", ["query", "n4"], "answer", "therefore answer 1")],
    )
    reviewer_calls = 0
    agent_calls = 0

    def fake_reviewer(prompt_name, data):
        nonlocal reviewer_calls
        assert prompt_name == "interrogation_targets.md"
        reviewer_calls += 1
        if reviewer_calls == 1:
            return {"nodes": [], "edges": ["e4"], "coverage": False, "reasons": {}}
        return {"nodes": [], "edges": [], "coverage": False, "reasons": {}}

    def fake_agent(prompt_name, data, model_config):
        nonlocal agent_calls
        assert prompt_name == "interrogate.md"
        assert model_config == AGENT_MODEL_CONFIG
        agent_calls += 1
        update = {
            "new_nodes": [
                {
                    "id": "n7",
                    "claim": "Denominators (5+1 and 2+1) are nonzero",
                    "kind": "calculation",
                }
            ],
            "new_edges": [],
            "updates": [
                {
                    "id": "e4",
                    "premise_node_ids": ["query", "n4", "n7"],
                    "target_node_id": "answer",
                    "claim": "x=1 has nonzero denominators, so answer 1",
                }
            ],
            "debt": [],
        }
        if agent_calls == 1:
            assert data["rejection_reason"] == ""
            return update
        assert data["rejection_reason"] == "interrogation node has no provenance: n7"
        return {
            **update,
            "new_edges": [
                {
                    "id": "e5",
                    "premise_node_ids": ["n1", "n4"],
                    "target_node_id": "n7",
                    "claim": "substitute x=1 into 5+x and 2+x",
                }
            ],
        }

    with (
        patch("graph_verifier.core.graph.complete_json", side_effect=fake_reviewer),
        patch("graph_verifier.core.graph.complete_agent_json", side_effect=fake_agent),
    ):
        interrogate(case, graph)

    assert agent_calls == 2
    assert [node.id for node in graph.nodes] == ["query", "n1", "n4", "answer", "n7"]
    assert [edge.id for edge in graph.edges] == ["e4", "e5"]
    assert graph.edges[0].premise_node_ids == ["n4", "n7", "query"]
    assert graph.edges[1].premise_node_ids == ["n1", "n4"]
    assert graph.edges[1].target_node_id == "n7"
    assert graph.edges[0].verification.reason == ""


def test_interrogation_accepts_all_provenance_receipts():
    grounded = Graph()
    assert apply_interrogation_update(
        grounded,
        {
            "new_nodes": [
                {
                    "id": "constraint",
                    "claim": "least positive integer value of x",
                    "kind": "query_constraint",
                }
            ]
        },
        question="Find the least positive integer value of x",
    )

    computed = Graph()
    assert apply_interrogation_update(
        computed,
        {"new_nodes": [{"id": "calculation", "claim": "90 / 30 = 3", "kind": "calculation"}]},
        question="Use the values 90 and 30",
    )

    derived = Graph(nodes=[Node("given", "given")])
    assert apply_interrogation_update(
        derived,
        {
            "new_nodes": [{"id": "derived", "claim": "derived result", "kind": "calculation"}],
            "new_edges": [
                {
                    "id": "support",
                    "premise_node_ids": ["given"],
                    "target_node_id": "derived",
                    "claim": "derive the result",
                }
            ],
        },
        question="Unrelated question",
    )


def test_interrogation_stops_at_max_rounds():
    case = Case(
        "case",
        QUESTION,
        "A",
        "A is cheaper.",
        agent_model_config=AGENT_MODEL_CONFIG,
    )
    graph = Graph(
        nodes=[Node("n1", "some vague support", "claim"), Node("n2", "another vague claim", "claim")],
        coverage_claim="n1 -> answer A",
    )
    target_calls = 0
    interrogation_calls = 0

    def fake_reviewer(prompt_name, data):
        nonlocal target_calls, interrogation_calls
        assert prompt_name == "interrogation_targets.md"
        target_calls += 1
        if target_calls > 2:
            return {"nodes": [], "edges": [], "coverage": False, "reasons": {}}
        return {"nodes": ["n1", "n2"], "edges": [], "coverage": False, "reasons": {}}

    def fake_agent(prompt_name, data, model_config):
        nonlocal interrogation_calls
        assert prompt_name == "interrogate.md"
        assert model_config == AGENT_MODEL_CONFIG
        interrogation_calls += 1
        claim = {
            "n1": "Provider A costs 100 dollars for 40 units",
            "n2": "Provider B costs 90 dollars for 30 units",
        }[data["target_id"]]
        return {
            "new_nodes": [],
            "new_edges": [],
            "updates": [
                    {
                        "id": data["target_id"],
                        "claim": claim,
                    "kind": "claim",
                }
            ],
            "debt": [],
        }

    with (
        patch("graph_verifier.core.graph.complete_json", side_effect=fake_reviewer),
        patch("graph_verifier.core.graph.complete_agent_json", side_effect=fake_agent),
    ):
        interrogate(case, graph, max_rounds=2)

    assert target_calls == 3
    assert interrogation_calls == 2
    assert graph.tool_debt == []
    assert [node.claim for node in graph.nodes] == [
        "Provider A costs 100 dollars for 40 units",
        "Provider B costs 90 dollars for 30 units",
    ]


def test_interrogation_reports_debt_when_targets_remain_after_max_rounds():
    case = Case(
        "case",
        QUESTION,
        "A",
        "",
        agent_model_config=AGENT_MODEL_CONFIG,
    )
    graph = Graph(nodes=[Node(f"n{i}", f"claim {i}", "claim") for i in range(3)])

    def fake_reviewer(prompt_name, data):
        assert prompt_name == "interrogation_targets.md"
        return {"nodes": ["n0", "n1", "n2"], "edges": [], "coverage": False, "reasons": {}}

    def fake_agent(prompt_name, data, model_config):
        assert prompt_name == "interrogate.md"
        assert model_config == AGENT_MODEL_CONFIG
        return {
            "new_nodes": [],
            "new_edges": [],
            "updates": [{"id": data["target_id"], "claim": f"resolved {data['target_id']}"}],
            "debt": [],
        }

    with (
        patch("graph_verifier.core.graph.complete_json", side_effect=fake_reviewer),
        patch("graph_verifier.core.graph.complete_agent_json", side_effect=fake_agent),
    ):
        interrogate(case, graph, max_rounds=2)
    assert graph.tool_debt == ["interrogation reached max rounds: 2"]


def test_reviewer_selected_incomplete_graph_gets_coverage_debt():
    graph = Graph(
        nodes=[
            Node("n1", "100 / 40 = 2.5", "calculation"),
            Node("n2", "answer A", "answer"),
        ],
        edges=[Edge("e1", ["n1"], "n2", "2.5 < 3")],
        coverage_claim="n1 -> e1 -> n2",
    )
    review = {
        "nodes": {"n1": True, "n2": True},
        "edges": {"e1": True},
        "coverage": True,
        "reasons": {"coverage": "missing Provider B calculation"},
    }
    assert status_for(graph, review=review) == "coverage_debt"


def test_supplied_decisive_labels_are_ignored():
    graph = Graph.from_dict(
        {
            "nodes": [
                {"id": "n1", "claim": "100 / 40 = 2.5", "decisive": True},
                {"id": "n2", "claim": "answer A", "kind": "answer", "decisive": True},
            ],
            "edges": [
                {
                    "id": "e1",
                    "premise_node_ids": ["n1"],
                    "target_node_id": "n2",
                    "claim": "2.5 < 3",
                    "decisive": True,
                }
            ],
            "coverage_claim": "n1 -> e1 -> n2",
            "coverage_decisive": False,
        }
    )
    assert not any(node.decisive for node in graph.nodes)
    assert not any(edge.decisive for edge in graph.edges)
    assert graph.coverage_decisive

    review = {"nodes": {"n1": False, "n2": False}, "edges": {"e1": False}, "coverage": True, "reasons": {}}
    assert status_for(graph, review=review) == "coverage_debt"
    assert all(node.decisive for node in graph.nodes)
    assert graph.edges[0].decisive
    assert graph.coverage_decisive


def test_reviewer_failure_is_tool_error():
    graph = complete_graph()
    assert status_for(graph, reviewer_error=LLMError("down")) == "tool_error"
    assert graph.tool_debt == ["decisiveness failed: down"]
    assert not any(node.decisive for node in graph.nodes)
    assert not any(edge.decisive for edge in graph.edges)
    assert graph.coverage_decisive


def test_late_tool_error_does_not_override_valid_proof():
    graph = complete_graph()
    graph.tool_debt.append("late endpoint failure")
    assert status_for(graph) == "verified_reliable"
    assert "non-fatal tool error" in final_status(graph).reason


def test_edge_with_missing_premise_node_is_debt():
    graph = Graph(
        nodes=[Node("n1", "answer A", "answer")],
        edges=[Edge("e1", ["missing"], "n1", "1 = 1")],
    )
    verify_graph(Case("case", QUESTION, "A", ""), graph)
    assert graph.edges[0].verification.status == "debt"
    assert graph.edges[0].verification.reason == "missing premise: missing"


def test_edge_with_missing_target_node_is_debt():
    graph = Graph(
        nodes=[Node("n1", "100 / 40 = 2.5", "calculation")],
        edges=[Edge("e1", ["n1"], "missing", "2.5 = 2.5")],
    )
    verify_graph(Case("case", QUESTION, "A", ""), graph)
    assert graph.edges[0].verification.status == "debt"
    assert graph.edges[0].verification.reason == "missing target: missing"


def test_edge_with_empty_premises_is_debt():
    graph = Graph(
        nodes=[Node("n1", "answer A", "answer")],
        edges=[Edge("e1", [], "n1", "1 = 1")],
    )
    verify_graph(Case("case", QUESTION, "A", ""), graph)
    assert graph.edges[0].verification.status == "debt"
    assert graph.edges[0].verification.reason == "empty premise_node_ids"


def test_multi_premise_comparison_target_verifies():
    graph = Graph(
        nodes=[
            Node("n1", "100 / 40 = 2.5", "calculation", decisive=True),
            Node("n2", "90 / 30 = 3", "calculation", decisive=True),
            Node("n3", "2.5 < 3", "comparison", decisive=True),
        ],
        edges=[Edge("e1", ["n1", "n2"], "n3", "2.5 < 3", decisive=True)],
    )
    verify_graph(Case("case", QUESTION, "A", ""), graph)
    assert graph.edges[0].verification.status == "valid"
    assert graph.nodes[2].verification.status == "valid"


def test_old_conclusion_field_is_rejected_as_malformed():
    try:
        Graph.from_dict(
            {
                "nodes": [{"id": "n1", "claim": "100 / 40 = 2.5"}],
                "edges": [
                    {
                        "id": "e1",
                        "premise_node_ids": ["n1"],
                        "claim": "2.5 < 3",
                        "conclusion": "answer A",
                    }
                ],
            }
        )
    except ValueError as exc:
        assert "conclusion" in str(exc)
    else:
        raise AssertionError("old conclusion field was accepted")


def test_number_occurrence_does_not_validate_a_claim_or_answer():
    question = "What is the least positive integer x for which 3x > 2x+1?"
    graph = Graph(
        nodes=[
            Node("given", "3x > 2x+1", "premise"),
            Node("claim", "The least positive integer greater than 1 is 2", "calculation"),
            Node("answer", "answer 2", "answer"),
        ]
    )
    verify_graph(Case("case", question, "2", ""), graph)
    assert graph.nodes[0].verification.status == "valid"
    assert graph.nodes[1].verification.status == "debt"
    assert graph.nodes[2].verification.reason == "answer requires verified support"


def test_symbolic_edge_verifier_gets_only_local_claims():
    question = "What is the least positive integer x for which 3x > 2x+1?"
    graph = Graph(
        nodes=[
            Node("given", "3x > 2x+1", "premise", decisive=True),
            Node("derived", "x > 1", "calculation", ["interrogation"], decisive=True),
        ],
        edges=[
            Edge(
                "edge",
                ["given"],
                "derived",
                "subtracting 2x from both sides preserves the inequality",
                decisive=True,
            )
        ],
    )
    calls = []

    def fake_complete_json(prompt_name, data):
        calls.append((prompt_name, data))
        return {
            "status": "valid",
            "reason": "subtracting the same expression preserves order",
            "used_premise_node_ids": ["given"],
        }

    with patch("graph_verifier.core.verify.complete_json", side_effect=fake_complete_json):
        verify_graph(Case("case", question, "2", ""), graph, verify_edge_with_llm)

    assert calls == [
        (
            "verify_edge.md",
            {
                "premises": [{"id": "given", "claim": "3x > 2x+1", "kind": "premise"}],
                "edge_claim": "subtracting 2x from both sides preserves the inequality",
                "target_claim": "x > 1",
            },
        )
    ]
    assert graph.edges[0].verification.status == "valid"
    assert graph.nodes[1].verification.status == "valid"


def test_symbolic_edge_verifier_waits_for_valid_premises():
    graph = Graph(
        nodes=[Node("unsupported", "hidden premise", decisive=True), Node("target", "x > 1", decisive=True)],
        edges=[Edge("edge", ["unsupported"], "target", "rule", decisive=True)],
    )
    checker_calls = 0

    def check_edge(premises, edge, target):
        nonlocal checker_calls
        checker_calls += 1
        return ClaimCheck("valid", "unexpected")

    verify_graph(Case("case", "3x > 2x+1", "2", ""), graph, check_edge)
    assert checker_calls == 0
    assert graph.edges[0].verification.reason == "unverified premise: unsupported"


def test_valid_edge_cannot_overwrite_a_refuted_target():
    graph = Graph(
        nodes=[
            Node("premise", "100 / 40 = 2.5", "calculation", decisive=True),
            Node("target", "100 / 40 = 3", "calculation", decisive=True),
        ],
        edges=[Edge("edge", ["premise"], "target", "unsupported rule", decisive=True)],
    )

    def check_edge(premises, edge, target):
        return ClaimCheck("valid", "claimed support")

    verify_graph(Case("case", "100 dollars for 40 units", "3", ""), graph, check_edge)
    assert graph.edges[0].verification.status == "refuted"
    assert graph.nodes[1].verification.status == "refuted"


def test_false_edge_claim_cannot_ride_on_a_true_target():
    graph = Graph(
        nodes=[
            Node("n1", "100 / 40 = 2.5", "calculation", decisive=True),
            Node("n2", "90 / 30 = 3", "calculation", decisive=True),
            Node("target", "2.5 < 3", "comparison", decisive=True),
        ],
        edges=[Edge("edge", ["n1", "n2"], "target", "2.5 > 3", decisive=True)],
    )
    verify_graph(Case("case", QUESTION, "A", ""), graph)
    assert graph.nodes[2].verification.status == "valid"
    assert graph.edges[0].verification.reason == "edge claim does not establish target"


def test_duplicate_ids_fail_closed():
    graph = Graph(nodes=[Node("same", "first"), Node("same", "second")])
    verify_graph(Case("case", "question", "answer", ""), graph)
    assert graph.tool_debt == ["duplicate node id"]
    assert graph.coverage_verification.status == "debt"


def test_malformed_edge_verifier_response_becomes_debt():
    with patch("graph_verifier.core.verify.complete_json", return_value=[]):
        check = verify_edge_with_llm(
            [Node("n1", "premise")],
            Edge("edge", ["n1"], "target", "rule"),
            Node("target", "conclusion"),
        )
    assert check.status == "debt"
    assert check.reason == "edge verifier failed: response is not an object"


def test_decisive_edge_verifier_failure_becomes_tool_error():
    graph = complete_graph()
    for node in graph.nodes:
        node.decisive = True
    for edge in graph.edges:
        edge.decisive = True
    with patch("graph_verifier.core.verify.complete_json", return_value=[]):
        verify_graph(Case("case", QUESTION, "A", ""), graph, verify_edge_with_llm)
    assert graph.tool_debt == [
        "edge verification failed for e2: edge verifier failed: response is not an object"
    ]
    assert final_status(graph).status == "tool_error"


def test_edge_verifier_exception_is_tool_error():
    case = Case("case", QUESTION, "A", "")
    graph = complete_graph()
    for item in [*graph.nodes, *graph.edges]:
        item.decisive = True

    def failed_checker(*args):
        raise LLMError("endpoint down")

    verify_graph(case, graph, failed_checker)
    assert graph.tool_debt == [
        "edge verification failed for e2: edge verifier failed: endpoint down"
    ]
    assert final_status(graph).status == "tool_error"


def test_non_decisive_edge_cannot_promote_a_decisive_node():
    graph = Graph(
        nodes=[
            Node("given", "3x > 2x+1", "premise"),
            Node("derived", "x > 1", "calculation", decisive=True),
            Node("answer", "answer 2", "answer", decisive=True),
        ],
        edges=[
            Edge("hidden", ["given"], "derived", "subtract 2x from both sides"),
            Edge("answer_edge", ["derived"], "answer", "answer 2", decisive=True),
        ],
    )
    verify_graph(
        Case("case", "Find x where 3x > 2x+1", "2", ""),
        graph,
        lambda *args: ClaimCheck("valid", "support"),
    )
    assert graph.edges[0].verification.status == "debt"
    assert graph.nodes[1].verification.status == "debt"
    assert graph.coverage_verification.status == "debt"


def test_answer_endpoint_is_canonicalized_from_agent_answer():
    case = Case("case", "question", "3/5", "")
    graph = Graph(nodes=[Node("answer", "The remaining fractional part is 3/5", "answer")])
    canonicalize_answer_node(case, graph)
    assert graph.nodes[0].claim == "answer 3/5"


def test_answer_matching_preserves_math_punctuation_and_structure():
    assert answer_claim_matches("answer 2", "answer 2")
    assert answer_claim_matches("answer 187.5", "187.5")
    assert not answer_claim_matches("answer 187.5", "1875")
    assert answer_claim_matches("answer (4,4)", "(4, 4)")
    assert not answer_claim_matches("answer (4,4)", "(44)")
    assert answer_claim_matches(r"answer \frac{1}{3}", "1/3")


def test_coverage_matches_decimal_coordinate_and_fraction_answers():
    for answer, claim in [
        ("187.5", "answer 187.5"),
        ("(4,4)", "answer (4, 4)"),
        ("1/3", r"answer \frac{1}{3}"),
    ]:
        graph = Graph(
            nodes=[
                Node(
                    "query",
                    "requested value",
                    QUERY_TARGET,
                    decisive=True,
                    verification=Verification("valid", "grounded"),
                ),
                Node(
                    "result",
                    "verified result",
                    decisive=True,
                    verification=Verification("valid", "supported"),
                ),
                Node(
                    "answer",
                    claim,
                    "answer",
                    decisive=True,
                    verification=Verification("valid", "supported"),
                ),
            ],
            edges=[
                Edge(
                    "finish",
                    ["query", "result"],
                    "answer",
                    "the verified result answers the query",
                    decisive=True,
                    verification=Verification("valid", "supported"),
                )
            ],
        )
        assert verify_coverage(Case("case", "question", answer, ""), graph).status == "valid"


def test_verified_terminal_value_maps_to_structured_answer():
    decimal = check_answer_edge(
        [Node("result", "value = 187.5")],
        Node("answer", "answer 187.5", "answer"),
        True,
    )
    coordinate = check_answer_edge(
        [Node("result", "midpoint = (4,4)")],
        Node("answer", "answer (4, 4)", "answer"),
        True,
    )
    expression = check_answer_edge(
        [Node("result", "value = 2 + 2")],
        Node("answer", "answer 4", "answer"),
        True,
    )
    assert decimal is not None and decimal.status == "valid"
    assert coordinate is not None and coordinate.status == "valid"
    assert expression is not None and expression.status == "valid"


def test_latex_wrappers_do_not_break_exact_grounding():
    check = check_grounding(
        "Find the least positive integer value of $x$ for which $x>1$.",
        "least positive integer value of x",
        ["interrogation"],
    )
    assert check.status == "valid"
    punctuation = check_grounding(
        "Harry, Ron and Neville race. If there are no ties, how many orders?",
        "Harry, Ron and Neville race. If there are no ties.",
        [],
    )
    assert punctuation.status == "valid"


def test_observed_mvp_math_syntax_is_locally_executable():
    assert safe_eval("(235-221)(235+221)").value == 6384
    assert safe_eval("3!").value == 6
    assert safe_eval("sqrt(144)").value == 12
    assert safe_eval("11²").value == 121
    assert safe_eval(r"\frac{3}{5}").value == Fraction(3, 5)
    assert check_closed_calculation("121 < 140 < 144").status == "valid"
    assert check_closed_calculation("11 < sqrt(140) < 12").status == "valid"
    assert check_closed_calculation("11 < √140 < 12").status == "valid"
    assert check_closed_calculation("235 - 221 = 14 and 235 + 221 = 456").status == "valid"
    assert (
        check_closed_calculation("average of 80 and 90 = (80+90)/2 = 85").status
        == "valid"
    )


def test_closed_math_does_not_depend_on_extractor_kind():
    graph = Graph(nodes=[Node("bound", "121 < 140 < 144", "inequality", decisive=True)])
    verify_graph(Case("case", "Evaluate sqrt(140)", "12", ""), graph)
    assert graph.nodes[0].verification.status == "valid"


def test_verified_result_can_map_to_canonical_answer():
    graph = Graph(
        nodes=[
            Node("given", "a", QUERY_TARGET, decisive=True),
            Node("result", "a = 4", "calculation", decisive=True),
            Node("answer", "answer 4", "answer", decisive=True),
        ],
        edges=[
            Edge("derive", ["given"], "result", "solve for a", decisive=True),
            Edge("finish", ["result"], "answer", "a is the requested answer", decisive=True),
        ],
    )
    verify_graph(Case("case", "Find a", "4", ""), graph, lambda *args: ClaimCheck("valid", "solved"))
    assert graph.edges[1].verification.status == "valid"
    assert graph.coverage_verification.status == "valid"


def test_unconnected_true_number_cannot_become_answer():
    graph = Graph(
        nodes=[
            Node("unrelated", "2 + 2 = 4", "calculation", decisive=True),
            Node("answer", "answer 4", "answer", decisive=True),
        ],
        edges=[Edge("finish", ["unrelated"], "answer", "therefore answer 4", decisive=True)],
    )
    verify_graph(Case("case", "What is 9 - 5?", "4", ""), graph)
    assert graph.edges[0].verification.status == "debt"
    assert graph.coverage_verification.status == "debt"


def test_exact_interrogative_query_constraint_has_provenance():
    graph = Graph()
    assert apply_interrogation_update(
        graph,
        {
            "new_nodes": [
                {
                    "id": "query",
                    "claim": "What is the value of $x$?",
                    "kind": "query_constraint",
                }
            ]
        },
        question="What is the value of $x$?",
    )


def test_edge_can_use_verified_summary_in_labeled_calculation():
    graph = Graph(
        nodes=[
            Node("sum", "70 + 95 = 165", "calculation", decisive=True),
            Node("union", "150 students in the union", "given", decisive=True),
            Node("both", "both = 15", "calculation", decisive=True),
        ],
        edges=[
            Edge(
                "inclusion",
                ["sum", "union"],
                "both",
                "by inclusion-exclusion, both = (70+95) - 150 = 165 - 150 = 15",
                decisive=True,
            )
        ],
    )
    verify_graph(Case("case", "150 students in the union", "15", ""), graph)
    assert graph.edges[0].verification.status == "valid"
    assert graph.nodes[2].verification.status == "valid"


def test_square_root_bound_uses_numbers_exposed_by_verified_identities():
    graph = Graph(
        nodes=[
            Node("query", r"\sqrt{140}", "query_constraint", decisive=True),
            Node("lower", "11² = 121", "calculation", decisive=True),
            Node("upper", "12² = 144", "calculation", decisive=True),
            Node("bound", r"11 < \sqrt{140} < 12", "calculation", decisive=True),
        ],
        edges=[
            Edge(
                "bound_sqrt",
                ["query", "lower", "upper"],
                "bound",
                "The verified squares bound 140, so 11 < √140 < 12",
                decisive=True,
            )
        ],
    )
    verify_graph(Case("case", "Evaluate sqrt(140)", "12", ""), graph)
    assert graph.edges[0].verification.status == "valid"


def test_interrogation_update_is_idempotent_and_deduplicates_sources():
    graph = Graph(nodes=[Node("n1", "given", "premise"), Node("answer", "answer A", "answer")])
    update = {
        "new_nodes": [
            {
                "id": "n2",
                "claim": "1 + 1 = 2",
                "kind": "calculation",
                "sources": ["interrogation"],
            }
        ],
        "new_edges": [
            {"id": "e1", "premise_node_ids": ["n1"], "target_node_id": "n2", "claim": "1 + 1 = 2"}
        ],
        "updates": [],
        "debt": [],
    }
    assert apply_interrogation_update(graph, update)
    assert not apply_interrogation_update(graph, update)
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 1
    assert graph.nodes[-1].sources == ["interrogation"]


def test_duplicate_node_is_reused_and_edge_reference_is_rewritten():
    graph = Graph(nodes=[Node("n1", "quoted premise", "premise"), Node("answer", "answer A", "answer")])
    update = {
        "new_nodes": [{"id": "duplicate", "claim": "quoted premise", "kind": "premise"}],
        "new_edges": [
            {
                "id": "e1",
                "premise_node_ids": ["duplicate"],
                "target_node_id": "answer",
                "claim": "quoted premise supports answer A",
            }
        ],
        "updates": [],
        "debt": [],
    }
    assert apply_interrogation_update(graph, update)
    assert [node.id for node in graph.nodes] == ["n1", "answer"]
    assert graph.edges[0].premise_node_ids == ["n1"]


def test_conflicting_update_is_atomic():
    graph = Graph(nodes=[Node("n1", "original", "premise")])
    update = {
        "new_nodes": [{"id": "n1", "claim": "conflict", "kind": "premise"}],
        "new_edges": [],
        "updates": [],
        "debt": [],
    }
    assert not apply_interrogation_update(graph, update)
    assert graph.nodes[0].claim == "original"


def test_unconnected_target_repair_is_rolled_back():
    graph = Graph(nodes=[Node("target", "vague claim", "claim"), Node("answer", "answer A", "answer")])
    update = {
        "new_nodes": [{"id": "replacement", "claim": "concrete claim", "kind": "calculation"}],
        "new_edges": [],
        "updates": [],
        "debt": [],
    }
    result = apply_interrogation_update(graph, update, target_type="node", target_id="target")
    assert not result
    assert result.rejection_reason == "interrogation did not resolve target"
    assert [node.id for node in graph.nodes] == ["target", "answer"]
    assert graph.nodes[0].verification.reason == ""


def test_explicit_target_debt_discards_extra_edits():
    graph = Graph(nodes=[Node("target", "unsupported", "claim")])
    update = {
        "new_nodes": [{"id": "extra", "claim": "unrelated", "kind": "claim"}],
        "new_edges": [],
        "updates": [],
        "debt": ["target"],
    }
    assert apply_interrogation_update(graph, update, target_type="node", target_id="target")
    assert [node.id for node in graph.nodes] == ["target"]
    assert graph.nodes[0].verification.reason == "interrogation could not ground"


def test_coverage_claim_only_is_not_a_repair():
    graph = Graph(nodes=[Node("answer", "answer A", "answer")], coverage_claim="missing")
    update = {
        "new_nodes": [],
        "new_edges": [],
        "updates": [{"id": "coverage", "coverage_claim": "looks complete"}],
        "debt": [],
    }
    result = apply_interrogation_update(graph, update, target_type="coverage", target_id="coverage")
    assert not result
    assert result.rejection_reason == "interrogation did not resolve target"
    assert graph.coverage_claim == "missing"
    assert graph.coverage_verification.reason == ""


def test_coverage_repair_cannot_add_a_different_answer_branch():
    graph = Graph(nodes=[Node("answer_a", "answer A", "answer")])
    update = {
        "new_nodes": [{"id": "answer_b", "claim": "answer B", "kind": "answer"}],
        "new_edges": [
            {
                "id": "edge_b",
                "premise_node_ids": ["answer_a"],
                "target_node_id": "answer_b",
                "claim": "choose B",
            }
        ],
        "updates": [],
        "debt": [],
    }
    assert not apply_interrogation_update(
        graph,
        update,
        target_type="coverage",
        target_id="coverage",
        answer_node_ids={"answer_a"},
    )
    assert [node.id for node in graph.nodes] == ["answer_a"]


def test_target_repair_cannot_mutate_an_unrelated_item():
    graph = Graph(nodes=[Node("target", "vague", "claim"), Node("other", "keep", "claim")])
    update = {
        "new_nodes": [],
        "new_edges": [],
        "updates": [
            {"id": "target", "claim": "fixed"},
            {"id": "other", "claim": "corrupted"},
        ],
        "debt": [],
    }
    assert not apply_interrogation_update(graph, update, target_type="node", target_id="target")
    assert [node.claim for node in graph.nodes] == ["vague", "keep"]


def test_one_malformed_edge_does_not_block_repairing_another():
    graph = Graph(
        nodes=[Node("premise", "given", "premise"), Node("answer", "answer A", "answer")],
        edges=[
            Edge("target", ["premise"], "missing", "rule"),
            Edge("other", ["missing"], "answer", "other rule"),
        ],
    )
    update = {
        "new_nodes": [],
        "new_edges": [],
        "updates": [{"id": "target", "target_node_id": "answer"}],
        "debt": [],
    }
    assert apply_interrogation_update(graph, update, target_type="edge", target_id="target")
    assert graph.edges[0].target_node_id == "answer"
    assert graph.edges[1].premise_node_ids == ["missing"]


def test_deduplication_preserves_case_sensitive_symbols():
    graph = Graph(nodes=[Node("lower", "x = 1", "calculation"), Node("answer", "answer A", "answer")])
    update = {
        "new_nodes": [{"id": "upper", "claim": "X = 1", "kind": "calculation"}],
        "new_edges": [
            {
                "id": "edge",
                "premise_node_ids": ["upper"],
                "target_node_id": "answer",
                "claim": "X supports A",
            }
        ],
        "updates": [],
        "debt": [],
    }
    assert apply_interrogation_update(graph, update)
    assert [node.id for node in graph.nodes] == ["lower", "answer", "upper"]
    assert graph.edges[0].premise_node_ids == ["upper"]


def test_decisiveness_drops_disconnected_branch():
    case = Case("case", QUESTION, "A", "")
    graph = Graph(
        nodes=[
            Node("main", "100 / 40 = 2.5", "calculation"),
            Node("answer", "answer A", "answer"),
            Node("other1", "90 / 30 = 3", "calculation"),
            Node("other2", "3 > 2.5", "comparison"),
        ],
        edges=[
            Edge("main_edge", ["main"], "answer", "choose A"),
            Edge("other_edge", ["other1"], "other2", "3 > 2.5"),
        ],
    )
    review = {
        "nodes": {node.id: True for node in graph.nodes},
        "edges": {edge.id: True for edge in graph.edges},
        "coverage": False,
        "reasons": {},
    }
    with patch("graph_verifier.core.graph.complete_json", return_value=review):
        mark_decisive(case, graph)
    assert [node.id for node in graph.nodes if node.decisive] == ["main", "answer"]
    assert [edge.id for edge in graph.edges if edge.decisive] == ["main_edge"]
    assert graph.coverage_decisive


def test_coverage_requires_an_explicit_answer_node():
    case = Case("case", "1 + 1", "2", "")
    graph = Graph(
        nodes=[Node("n1", "1 + 1 = 2", "calculation", decisive=True)],
        coverage_verification=Verification(),
    )
    graph.nodes[0].verification = Verification("valid", "computed")
    assert verify_coverage(case, graph).status == "debt"


def test_coverage_rejects_numeric_answer_prefixes_and_fractions():
    case = Case("case", "q", "2", "")
    for claim in ("answer -2", "answer 2/3", "answer 2.5"):
        graph = Graph(
            nodes=[
                Node("premise", "premise", decisive=True, verification=Verification("valid", "ok")),
                Node("answer", claim, "answer", decisive=True, verification=Verification("valid", "ok")),
            ],
            edges=[
                Edge(
                    "edge",
                    ["premise"],
                    "answer",
                    "rule",
                    decisive=True,
                    verification=Verification("valid", "ok"),
                )
            ],
        )
        assert verify_coverage(case, graph).status == "debt"


def test_coverage_requires_a_query_target():
    case = Case("case", "What is 1 + 1?", "2", "")
    graph = Graph(
        nodes=[
            Node(
                "premise",
                "1 + 1 = 2",
                "calculation",
                decisive=True,
                verification=Verification("valid", "computed"),
            ),
            Node(
                "answer",
                "answer 2",
                "answer",
                decisive=True,
                verification=Verification("valid", "supported"),
            ),
        ],
        edges=[
            Edge(
                "finish",
                ["premise"],
                "answer",
                "the computed value is the answer",
                decisive=True,
                verification=Verification("valid", "supported"),
            )
        ],
    )

    coverage = verify_coverage(case, graph)
    assert coverage.status == "debt"
    assert coverage.reason == "missing query target"


def test_coverage_requires_query_target_on_answer_path():
    case = Case("case", "What is 1 + 1?", "2", "")
    graph = Graph(
        nodes=[
            Node(
                "query",
                "1 + 1",
                QUERY_TARGET,
                verification=Verification("valid", "grounded"),
            ),
            Node(
                "premise",
                "1 + 1 = 2",
                "calculation",
                decisive=True,
                verification=Verification("valid", "computed"),
            ),
            Node(
                "answer",
                "answer 2",
                "answer",
                decisive=True,
                verification=Verification("valid", "supported"),
            ),
        ],
        edges=[
            Edge(
                "finish",
                ["premise"],
                "answer",
                "the computed value is the answer",
                decisive=True,
                verification=Verification("valid", "supported"),
            )
        ],
    )

    coverage = verify_coverage(case, graph)
    assert coverage.status == "debt"
    assert coverage.reason == "query target not on valid answer path"


def test_missing_query_target_is_deterministically_repaired():
    case = Case(
        "case",
        "Find the value of the first term in the geometric sequence a,b,c,32,64.",
        "4",
        "The ratio is 2, so a = 4.",
        agent_model_config=AGENT_MODEL_CONFIG,
    )
    graph = Graph(
        nodes=[
            Node("q1", "the geometric sequence a,b,c,32,64", "query_constraint"),
            Node("n5", "32 / 8 = 4", "calculation"),
            Node("answer", "answer 4", "answer"),
        ],
        edges=[Edge("e6", ["q1", "n5"], "answer", "a is 4, so answer 4")],
    )
    reviewer_calls = 0
    agent_calls = 0

    def fake_reviewer(prompt_name, data):
        nonlocal reviewer_calls
        reviewer_calls += 1
        assert prompt_name == "interrogation_targets.md"
        assert any(node["kind"] == QUERY_TARGET for node in data["graph"]["nodes"])
        return {"nodes": [], "edges": [], "coverage": False, "reasons": {}}

    def fake_agent(prompt_name, data, model_config):
        nonlocal agent_calls
        agent_calls += 1
        assert prompt_name == "interrogate.md"
        assert model_config == AGENT_MODEL_CONFIG
        assert data["target_type"] == "edge"
        assert data["target_id"] == "e6"
        assert data["target_reason"] == "missing dedicated query target"
        return {
            "new_nodes": [
                {
                    "id": "q_target",
                    "claim": "the value of the first term",
                    "kind": QUERY_TARGET,
                    "sources": ["question"],
                }
            ],
            "new_edges": [],
            "updates": [
                {
                    "id": "e6",
                    "premise_node_ids": ["q1", "n5", "q_target"],
                    "target_node_id": "answer",
                    "claim": "a is the first term and equals 4, which is the requested value",
                }
            ],
            "debt": [],
        }

    with (
        patch("graph_verifier.core.graph.complete_json", side_effect=fake_reviewer),
        patch("graph_verifier.core.graph.complete_agent_json", side_effect=fake_agent),
    ):
        interrogate(case, graph)

    assert reviewer_calls == 1
    assert agent_calls == 1
    assert graph.tool_debt == []
    assert {node.id for node in graph.nodes if node.kind == QUERY_TARGET} == {"q_target"}
    assert set(graph.edges[0].premise_node_ids) == {"q1", "n5", "q_target"}

    for node in graph.nodes:
        node.decisive = True
    for edge in graph.edges:
        edge.decisive = True
    verify_graph(case, graph, lambda *args: ClaimCheck("valid", "supported"))
    assert graph.coverage_verification.status == "valid"


def verification_feedback_graph() -> Graph:
    return Graph(
        nodes=[
            Node(
                "given",
                "given fact",
                decisive=True,
                verification=Verification("valid", "grounded"),
            ),
            Node(
                "derived",
                "derived claim",
                decisive=True,
                verification=Verification("debt", "not grounded"),
            ),
            Node(
                "answer",
                "answer A",
                "answer",
                decisive=True,
                verification=Verification("debt", "answer requires verified support"),
            ),
        ],
        edges=[
            Edge(
                "cause",
                ["given"],
                "derived",
                "unsupported rule",
                decisive=True,
                verification=Verification("debt", "edge claim does not establish target"),
            ),
            Edge(
                "finish",
                ["derived"],
                "answer",
                "derived claim gives answer A",
                decisive=True,
                verification=Verification("debt", "unverified premise: derived"),
            ),
        ],
        coverage_verification=Verification("debt", "no valid decisive path to answer"),
    )


def test_verification_feedback_selects_causal_edge():
    target = select_verification_target(verification_feedback_graph())
    assert target is not None
    assert target["target_type"] == "edge"
    assert target["target_id"] == "cause"
    assert target["target_reason"] == "edge claim does not establish target"


def test_verification_feedback_follows_debt_to_root_node():
    graph = Graph(
        nodes=[
            Node(
                "root",
                "unsupported root",
                decisive=True,
                verification=Verification("debt", "not grounded"),
            ),
            Node(
                "derived",
                "derived claim",
                decisive=True,
                verification=Verification("debt", "not grounded"),
            ),
            Node(
                "answer",
                "answer A",
                "answer",
                decisive=True,
                verification=Verification("debt", "answer requires verified support"),
            ),
        ],
        edges=[
            Edge(
                "first",
                ["root"],
                "derived",
                "derive",
                decisive=True,
                verification=Verification("debt", "unverified premise: root"),
            ),
            Edge(
                "finish",
                ["derived"],
                "answer",
                "finish",
                decisive=True,
                verification=Verification("debt", "unverified premise: derived"),
            ),
        ],
        coverage_verification=Verification("debt", "no valid decisive path to answer"),
    )

    target = select_verification_target(graph)
    assert target is not None
    assert target["target_type"] == "node"
    assert target["target_id"] == "root"
    assert target["target_reason"] == "not grounded"


def test_verification_feedback_uses_coverage_only_as_fallback():
    graph = Graph(
        nodes=[
            Node(
                "query",
                "requested value",
                QUERY_TARGET,
                decisive=True,
                verification=Verification("valid", "grounded"),
            ),
            Node(
                "answer",
                "answer A",
                "answer",
                decisive=True,
                verification=Verification("debt", "answer requires verified support"),
            ),
        ],
        coverage_verification=Verification("debt", "no valid decisive path to answer"),
    )

    target = select_verification_target(graph)
    assert target is not None
    assert target["target_type"] == "coverage"
    assert target["target_reason"] == "no valid decisive path to answer"


def test_verification_feedback_respects_handled_signatures_and_terminal_failures():
    graph = verification_feedback_graph()
    handled = {("edge", "cause", target_signature(graph, "edge", "cause"))}
    assert select_verification_target(graph, handled) is None

    graph.edges[0].claim = "changed unsupported rule"
    assert select_verification_target(graph, handled)["target_id"] == "cause"

    graph.edges[0].verification = Verification("refuted", "rule is false")
    assert select_verification_target(graph, handled) is None
    graph.edges[0].verification = Verification("debt", "edge verifier failed: endpoint down")
    assert select_verification_target(graph, handled) is None


def test_forced_feedback_target_passes_exact_verifier_reason():
    case = Case("case", QUESTION, "A", "", agent_model_config=AGENT_MODEL_CONFIG)
    graph = verification_feedback_graph()
    target = select_verification_target(graph)
    assert target is not None
    before = target_signature(graph, "edge", "cause")
    state = InterrogationState()
    payloads = []

    def fake_agent(prompt_name, data, model_config):
        assert prompt_name == "interrogate.md"
        assert model_config == AGENT_MODEL_CONFIG
        payloads.append(data)
        return {
            "new_nodes": [],
            "new_edges": [],
            "updates": [
                {
                    "id": "cause",
                    "premise_node_ids": ["given"],
                    "target_node_id": "derived",
                    "claim": "given fact entails the derived claim",
                }
            ],
            "debt": [],
        }

    with (
        patch(
            "graph_verifier.core.graph.complete_json",
            side_effect=AssertionError("forced target called the ordinary selector"),
        ),
        patch("graph_verifier.core.graph.complete_agent_json", side_effect=fake_agent),
    ):
        interrogate(case, graph, max_rounds=1, state=state, forced_target=target)

    assert len(payloads) == 1
    assert payloads[0]["target_id"] == "cause"
    assert payloads[0]["target_reason"] == "edge claim does not establish target"
    assert payloads[0]["target_edge"]["claim"] == "unsupported rule"
    assert state.rounds_used == 1
    assert ("edge", "cause", before) in state.handled
    assert graph.edges[0].claim == "given fact entails the derived claim"


def test_feedback_pipeline_reuses_graph_and_reverifies_after_repair():
    case = Case("case", QUESTION, "A", "", agent_model_config=AGENT_MODEL_CONFIG)
    graph = verification_feedback_graph()
    calls = {"interrogate": 0, "decisive": 0, "verify": 0}

    def fake_interrogate(case_arg, graph_arg, artifact_dir, max_rounds, state, forced_target):
        assert case_arg is case
        assert graph_arg is graph
        calls["interrogate"] += 1
        if forced_target is not None:
            assert forced_target["target_id"] == "cause"
            graph_arg.edges[0].claim = "repaired rule"
            state.rounds_used += 1
        return graph_arg

    def fake_decisive(case_arg, graph_arg):
        assert case_arg is case
        assert graph_arg is graph
        calls["decisive"] += 1
        for item in [*graph_arg.nodes, *graph_arg.edges]:
            item.decisive = True
        return graph_arg

    def fake_verify(case_arg, graph_arg, edge_checker):
        assert case_arg is case
        assert graph_arg is graph
        calls["verify"] += 1
        if graph_arg.edges[0].claim == "repaired rule":
            for item in [*graph_arg.nodes, *graph_arg.edges]:
                item.verification = Verification("valid", "verified")
            graph_arg.coverage_verification = Verification("valid", "complete")
        return graph_arg

    with (
        patch("graph_verifier.main.interrogate", side_effect=fake_interrogate),
        patch("graph_verifier.main.mark_decisive", side_effect=fake_decisive),
        patch("graph_verifier.main.verify_graph", side_effect=fake_verify),
        patch("graph_verifier.main.save_graph"),
    ):
        result = run_interrogation_verification(
            case,
            graph,
            Path("artifacts"),
            3,
            lambda *args: ClaimCheck("valid", "verified"),
        )

    assert result is graph
    assert calls == {"interrogate": 2, "decisive": 2, "verify": 2}
    assert final_status(graph).status == "verified_reliable"


def test_feedback_pipeline_enforces_one_global_repair_budget():
    case = Case("case", QUESTION, "A", "", agent_model_config=AGENT_MODEL_CONFIG)
    graph = verification_feedback_graph()
    calls = {"interrogate": 0, "verify": 0}

    def fake_interrogate(case_arg, graph_arg, artifact_dir, max_rounds, state, forced_target):
        calls["interrogate"] += 1
        if forced_target is not None:
            graph_arg.edges[0].claim += " changed"
            state.rounds_used += 1
        return graph_arg

    def fake_verify(case_arg, graph_arg, edge_checker):
        calls["verify"] += 1
        return graph_arg

    with (
        patch("graph_verifier.main.interrogate", side_effect=fake_interrogate),
        patch("graph_verifier.main.mark_decisive", side_effect=lambda case_arg, graph_arg: graph_arg),
        patch("graph_verifier.main.verify_graph", side_effect=fake_verify),
        patch("graph_verifier.main.save_graph"),
        patch("graph_verifier.main.save_interrogation_event"),
    ):
        run_interrogation_verification(case, graph, Path("artifacts"), 1)

    assert calls == {"interrogate": 2, "verify": 2}
    assert graph.tool_debt == ["verification feedback reached max rounds: 1"]


def test_feedback_pipeline_stops_without_reverification_on_no_progress():
    case = Case("case", QUESTION, "A", "", agent_model_config=AGENT_MODEL_CONFIG)
    graph = verification_feedback_graph()
    calls = {"interrogate": 0, "decisive": 0, "verify": 0}

    def fake_interrogate(case_arg, graph_arg, artifact_dir, max_rounds, state, forced_target):
        calls["interrogate"] += 1
        if forced_target is not None:
            state.rounds_used += 1
            state.handled.add(
                ("edge", "cause", target_signature(graph_arg, "edge", "cause"))
            )
            graph_arg.edges[0].verification = Verification(
                "debt", "interrogation could not ground"
            )
        return graph_arg

    def fake_decisive(case_arg, graph_arg):
        calls["decisive"] += 1
        return graph_arg

    def fake_verify(case_arg, graph_arg, edge_checker):
        calls["verify"] += 1
        return graph_arg

    with (
        patch("graph_verifier.main.interrogate", side_effect=fake_interrogate),
        patch("graph_verifier.main.mark_decisive", side_effect=fake_decisive),
        patch("graph_verifier.main.verify_graph", side_effect=fake_verify),
        patch("graph_verifier.main.save_graph"),
    ):
        run_interrogation_verification(case, graph, Path("artifacts"), 3)

    assert calls == {"interrogate": 2, "decisive": 1, "verify": 1}
    assert graph.edges[0].verification.reason == "interrogation could not ground"


def test_complete_json_retries_a_malformed_response():
    calls = 0

    def fake_complete(content, model_config):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise LLMError("malformed endpoint JSON")
        return '{"ok": true}'

    with (
        patch("graph_verifier.utils.llm.complete", side_effect=fake_complete),
        patch("graph_verifier.utils.llm.time.sleep") as sleep,
    ):
        assert complete_json("graph_extract.md", {}) == {"ok": True}
    assert calls == 3
    assert [item.args for item in sleep.call_args_list] == [(1.0,), (2.0,)]


def test_complete_json_retries_null_content_with_backoff():
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    responses = [
        FakeResponse({"choices": [{"message": {"content": None}}]}),
        FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]}),
    ]
    with TemporaryDirectory() as directory:
        config = Path(directory) / "model.json"
        config.write_text(
            json.dumps({"api_key": "test", "llm_url": "https://example.test", "model": "test"})
        )
        with (
            patch("graph_verifier.utils.llm.urllib.request.urlopen", side_effect=responses),
            patch("graph_verifier.utils.llm.time.sleep") as sleep,
        ):
            assert complete_json("graph_extract.md", {}, config, attempts=2) == {"ok": True}
    sleep.assert_called_once_with(1.0)


def test_compact_output_counts_only_decisive_items():
    graph = Graph(
        nodes=[
            Node("kept", "kept", decisive=True, verification=Verification("valid", "ok")),
            Node("ignored", "ignored", verification=Verification("debt", "irrelevant")),
        ],
        coverage_verification=Verification("valid", "path"),
    )
    output = compact_output(Case("case", "q", "a", ""), "interrogation", graph, "verified_reliable")
    assert output["valid"] == 2
    assert output["debt"] == 0


def test_compact_output_separates_tool_errors_from_debt():
    graph = Graph(tool_debt=["endpoint failed"], coverage_decisive=False)
    output = compact_output(Case("case", "q", "a", ""), "interrogation", graph, "tool_error")
    assert output["debt"] == 0
    assert output["tool_errors"] == 1


def test_case_processing_is_concurrent_bounded_and_ordered():
    cases = [
        Case("first", "q", "a", ""),
        Case("second", "q", "a", ""),
        Case("third", "q", "a", ""),
    ]
    barrier = Barrier(2)
    lock = Lock()
    active = 0
    peak = 0

    def fake_process(case, *, mode, artifact_dir, max_interrogation_rounds):
        nonlocal active, peak
        assert mode == "interrogation"
        assert artifact_dir == Path("artifacts")
        assert max_interrogation_rounds == 3
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if case.id != "third":
                barrier.wait(timeout=2)
            return {"id": case.id, "status": "verification_debt"}
        finally:
            with lock:
                active -= 1

    with patch("graph_verifier.main.process_case", side_effect=fake_process):
        outputs = list(process_cases(cases, "interrogation", Path("artifacts"), 3, 2))

    assert peak == 2
    assert [output["id"] for output in outputs] == ["first", "second", "third"]


def test_case_processing_rejects_invalid_concurrency():
    try:
        list(process_cases([], "interrogation", Path("artifacts"), 3, 0))
    except ValueError as exc:
        assert "concurrency" in str(exc)
    else:
        raise AssertionError("invalid concurrency was accepted")


def test_case_artifact_name_collisions_are_rejected():
    try:
        validate_case_names([Case("same/id", "q", "a", ""), Case("same_id", "q", "a", "")])
    except ValueError as exc:
        assert "artifact name collision" in str(exc)
    else:
        raise AssertionError("artifact name collision was accepted")


if __name__ == "__main__":
    for test in [
        test_complete_correct_graph,
        test_correct_answer_incomplete_graph_gets_coverage_debt,
        test_correct_answer_wrong_premise_is_not_reliable,
        test_incorrect_arithmetic_decisive_node,
        test_refuted_comparison_supports_final_answer,
        test_irrelevant_true_edge_does_not_support_answer,
        test_edge_cannot_use_numbers_missing_from_premises,
        test_decisive_edge_forces_its_premises_decisive,
        test_decisive_node_pulls_in_sole_support_edge,
        test_interrogation_unsupported_premise_is_debt_not_refutation,
        test_interrogation_without_original_agent_marks_debt,
        test_coverage_gap_anchors_to_only_answer_edge,
        test_interrogation_uses_llm_targets,
        test_interrogation_update_mutates_existing_targets,
        test_interrogation_target_selection_sees_updates,
        test_interrogation_repairs_ungrounded_query_target,
        test_interrogation_repairs_unsupported_root_on_answer_path,
        test_changed_edge_can_be_refined,
        test_interrogation_rejects_node_without_provenance,
        test_interrogation_retries_orphan_node_with_rejection_reason,
        test_interrogation_accepts_all_provenance_receipts,
        test_interrogation_stops_at_max_rounds,
        test_interrogation_reports_debt_when_targets_remain_after_max_rounds,
        test_reviewer_selected_incomplete_graph_gets_coverage_debt,
        test_supplied_decisive_labels_are_ignored,
        test_reviewer_failure_is_tool_error,
        test_late_tool_error_does_not_override_valid_proof,
        test_edge_with_missing_premise_node_is_debt,
        test_edge_with_missing_target_node_is_debt,
        test_edge_with_empty_premises_is_debt,
        test_multi_premise_comparison_target_verifies,
        test_old_conclusion_field_is_rejected_as_malformed,
        test_number_occurrence_does_not_validate_a_claim_or_answer,
        test_symbolic_edge_verifier_gets_only_local_claims,
        test_symbolic_edge_verifier_waits_for_valid_premises,
        test_valid_edge_cannot_overwrite_a_refuted_target,
        test_false_edge_claim_cannot_ride_on_a_true_target,
        test_duplicate_ids_fail_closed,
        test_malformed_edge_verifier_response_becomes_debt,
        test_decisive_edge_verifier_failure_becomes_tool_error,
        test_edge_verifier_exception_is_tool_error,
        test_non_decisive_edge_cannot_promote_a_decisive_node,
        test_answer_endpoint_is_canonicalized_from_agent_answer,
        test_answer_matching_preserves_math_punctuation_and_structure,
        test_coverage_matches_decimal_coordinate_and_fraction_answers,
        test_verified_terminal_value_maps_to_structured_answer,
        test_latex_wrappers_do_not_break_exact_grounding,
        test_observed_mvp_math_syntax_is_locally_executable,
        test_closed_math_does_not_depend_on_extractor_kind,
        test_verified_result_can_map_to_canonical_answer,
        test_unconnected_true_number_cannot_become_answer,
        test_exact_interrogative_query_constraint_has_provenance,
        test_edge_can_use_verified_summary_in_labeled_calculation,
        test_square_root_bound_uses_numbers_exposed_by_verified_identities,
        test_interrogation_update_is_idempotent_and_deduplicates_sources,
        test_duplicate_node_is_reused_and_edge_reference_is_rewritten,
        test_conflicting_update_is_atomic,
        test_unconnected_target_repair_is_rolled_back,
        test_explicit_target_debt_discards_extra_edits,
        test_coverage_claim_only_is_not_a_repair,
        test_coverage_repair_cannot_add_a_different_answer_branch,
        test_target_repair_cannot_mutate_an_unrelated_item,
        test_one_malformed_edge_does_not_block_repairing_another,
        test_deduplication_preserves_case_sensitive_symbols,
        test_decisiveness_drops_disconnected_branch,
        test_coverage_requires_an_explicit_answer_node,
        test_coverage_rejects_numeric_answer_prefixes_and_fractions,
        test_coverage_requires_a_query_target,
        test_coverage_requires_query_target_on_answer_path,
        test_missing_query_target_is_deterministically_repaired,
        test_verification_feedback_selects_causal_edge,
        test_verification_feedback_follows_debt_to_root_node,
        test_verification_feedback_uses_coverage_only_as_fallback,
        test_verification_feedback_respects_handled_signatures_and_terminal_failures,
        test_forced_feedback_target_passes_exact_verifier_reason,
        test_feedback_pipeline_reuses_graph_and_reverifies_after_repair,
        test_feedback_pipeline_enforces_one_global_repair_budget,
        test_feedback_pipeline_stops_without_reverification_on_no_progress,
        test_complete_json_retries_a_malformed_response,
        test_complete_json_retries_null_content_with_backoff,
        test_compact_output_counts_only_decisive_items,
        test_compact_output_separates_tool_errors_from_debt,
        test_case_processing_is_concurrent_bounded_and_ordered,
        test_case_processing_rejects_invalid_concurrency,
        test_case_artifact_name_collisions_are_rejected,
    ]:
        test()
    print("ok")
