Extract the examinee's claimed dependency graph. Do not solve the problem and do not infer missing facts.

Return only JSON:
{
  "nodes": [
    {"id": "q1", "claim": "the value of x", "kind": "query_target", "sources": ["question"]},
    {"id": "n1", "claim": "2 + 2 = 4", "kind": "calculation", "sources": ["reasoning"]},
    {"id": "n2", "claim": "answer 4", "kind": "answer", "sources": ["reasoning"]}
  ],
  "edges": [{"id": "e1", "premise_node_ids": ["q1", "n1"], "target_node_id": "n2", "claim": "the computed value is the requested value"}],
  "coverage_claim": "q1,n1 -> e1 -> n2"
}

Every edge must have premise_node_ids, target_node_id, and claim. All premise IDs and target_node_id must exist in nodes. Do not use a conclusion field.

Keep node claims concrete and locally checkable. Conclusions must be nodes. Each edge claim must state the operation or rule that makes its listed premises entail its target, including any mapping from a comparison to the chosen answer. Extract one minimal support path and omit duplicate or explanatory alternatives. Ignore any decisiveness labels in the input.

The one final answer node must use the exact canonical claim `answer <agent_answer>`. Preserve the mathematical role in intermediate claims, such as `largest angle = 7 * 10 = 70`, instead of reducing it to a bare calculation.

Every graph must contain at least one dedicated `query_target` node that quotes only the operative value or object requested by the question, including its original math notation. Omit interrogative filler such as "what is" or "find". Every support path into the final answer must include a `query_target` as a premise or ancestor.

Every other root fact or query constraint taken from the question must be its own node and quote only the operative question phrase exactly, including its original math notation. Use kind `query_constraint`, never kind `question`. Put formalization or interpretation in an outgoing edge, not inside the quoted node. Every question-specific value introduced by an edge must appear in one of that edge's premise nodes.

Ordinary mathematical definitions, operations, and theorems are edge rules, not premise nodes. For example, factorial counting, inclusion-exclusion, the triangle angle sum, and subtracting a fractional part from one belong in edge claims. Do not create nodes whose only purpose is to state a general mathematical rule.
