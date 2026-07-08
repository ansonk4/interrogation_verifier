from unittest.mock import patch

from graph_verifier.core.aggregate import final_status
from graph_verifier.core.graph import mark_decisive
from graph_verifier.core.models import Case, Edge, Graph, Node
from graph_verifier.core.verify import verify_graph
from graph_verifier.utils.llm import LLMError


QUESTION = "Provider A costs 100 dollars for 40 units. Provider B costs 90 dollars for 30 units."


def status_for(
    graph: Graph,
    answer: str = "A",
    question: str = QUESTION,
    review: dict | None = None,
    reviewer_error: Exception | None = None,
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
    verify_graph(case, graph)
    return final_status(graph).status


def test_complete_correct_graph():
    graph = Graph(
        nodes=[
            Node("n1", "100 / 40 = 2.5", "calculation"),
            Node("n2", "90 / 30 = 3", "calculation"),
        ],
        edges=[Edge("e1", ["n1", "n2"], "answer A", "2.5 < 3")],
        coverage_claim="n1,n2 -> e1 -> answer A",
    )
    assert status_for(graph) == "verified_reliable"


def test_correct_answer_incomplete_graph_gets_coverage_debt():
    graph = Graph(
        nodes=[Node("n1", "100 / 40 = 2.5", "calculation")],
        edges=[Edge("e1", ["n1"], "answer A", "A is the answer")],
        coverage_claim="n1 -> answer A",
    )
    assert status_for(graph) == "coverage_debt"


def test_correct_answer_wrong_premise_is_not_reliable():
    graph = Graph(
        nodes=[Node("n1", "100 / 50 = 2.5", "calculation")],
        edges=[Edge("e1", ["n1"], "answer A", "2.5 < 3")],
        coverage_claim="n1 -> e1 -> answer A",
    )
    assert status_for(graph) != "verified_reliable"


def test_incorrect_arithmetic_decisive_node():
    graph = Graph(
        nodes=[Node("n1", "100 / 40 = 3", "calculation")],
        edges=[Edge("e1", ["n1"], "answer A", "3 < 4")],
        coverage_claim="n1 -> e1 -> answer A",
    )
    assert status_for(graph) == "answer_refuted"


def test_refuted_comparison_supports_final_answer():
    graph = Graph(
        nodes=[
            Node("n1", "100 / 40 = 2.5", "calculation"),
            Node("n2", "90 / 30 = 3", "calculation"),
        ],
        edges=[Edge("e1", ["n1", "n2"], "answer B", "3 < 2.5")],
        coverage_claim="n1,n2 -> e1 -> answer B",
    )
    assert status_for(graph, answer="B") == "answer_refuted"


def test_interrogation_unsupported_premise_is_debt_not_refutation():
    graph = Graph(
        nodes=[Node("n1", "A has a hidden discount of 10", "premise", ["interrogation"])],
        edges=[Edge("e1", ["n1"], "answer A", "A is the answer")],
        coverage_claim="n1 -> e1 -> answer A",
    )
    assert status_for(graph) != "answer_refuted"
    assert status_for(graph) != "verified_reliable"


def test_reviewer_selected_incomplete_graph_gets_coverage_debt():
    graph = Graph(
        nodes=[Node("n1", "100 / 40 = 2.5", "calculation")],
        edges=[Edge("e1", ["n1"], "answer A", "2.5 < 3")],
        coverage_claim="n1 -> e1 -> answer A",
    )
    review = {
        "nodes": {"n1": True},
        "edges": {"e1": True},
        "coverage": True,
        "reasons": {"coverage": "missing Provider B calculation"},
    }
    assert status_for(graph, review=review) == "coverage_debt"


def test_supplied_decisive_labels_are_ignored():
    graph = Graph.from_dict(
        {
            "nodes": [{"id": "n1", "claim": "100 / 40 = 2.5", "decisive": True}],
            "edges": [
                {
                    "id": "e1",
                    "premise_node_ids": ["n1"],
                    "claim": "2.5 < 3",
                    "conclusion": "answer A",
                    "decisive": True,
                }
            ],
            "coverage_claim": "n1 -> e1 -> answer A",
            "coverage_decisive": False,
        }
    )
    review = {"nodes": {"n1": False}, "edges": {"e1": False}, "coverage": True, "reasons": {}}
    assert status_for(graph, review=review) == "coverage_debt"
    assert not graph.nodes[0].decisive
    assert not graph.edges[0].decisive
    assert graph.coverage_decisive


def test_reviewer_failure_is_verification_debt():
    graph = Graph(
        nodes=[
            Node("n1", "100 / 40 = 2.5", "calculation"),
            Node("n2", "90 / 30 = 3", "calculation"),
        ],
        edges=[Edge("e1", ["n1", "n2"], "answer A", "2.5 < 3")],
        coverage_claim="n1,n2 -> e1 -> answer A",
    )
    assert status_for(graph, reviewer_error=LLMError("down")) == "verification_debt"
    assert graph.tool_debt == ["decisiveness failed: down"]
    assert not any(node.decisive for node in graph.nodes)
    assert not any(edge.decisive for edge in graph.edges)
    assert graph.coverage_decisive


if __name__ == "__main__":
    for test in [
        test_complete_correct_graph,
        test_correct_answer_incomplete_graph_gets_coverage_debt,
        test_correct_answer_wrong_premise_is_not_reliable,
        test_incorrect_arithmetic_decisive_node,
        test_refuted_comparison_supports_final_answer,
        test_interrogation_unsupported_premise_is_debt_not_refutation,
        test_reviewer_selected_incomplete_graph_gets_coverage_debt,
        test_supplied_decisive_labels_are_ignored,
        test_reviewer_failure_is_verification_debt,
    ]:
        test()
    print("ok")
