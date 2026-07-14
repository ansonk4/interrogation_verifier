Select graph items that need interrogation before verification.

Select at most one item total: one node, one edge, or coverage. Prefer the closest unresolved item on the support path to the final answer. Do not select duplicate or alternative paths once one minimal path is sufficient.

Look for vague claims, unsupported reasoning jumps, missing premises, missing target nodes, missing rules, or coverage gaps that could prevent the graph from supporting the final answer. Judge the current graph as-is, including any existing updates to node claims, edge premises, edge targets, edge claims, and coverage claim.

First check each dependency locally from its listed premises through its edge rule to its target. Do not select an edge that is already locally valid. Ordinary mathematical definitions, operations, and theorems may be stated in the edge claim and do not need separate premise nodes. Never request a premise node merely to restate factorial counting, inclusion-exclusion, the triangle angle sum, a unit-whole convention, algebraic manipulation, or a similar general rule.

Select a missing node only for a question-specific fact or constraint, a concrete calculation asserted by the examined agent, or a semantic role lost from the examined agent's reasoning. Preserve roles and units: a bare calculation such as `7 * 10 = 70` does not by itself state that 70 is the largest angle.

Before returning no target, compare the graph with every explicit query constraint in the question, including the requested domain and optimization condition.

Every graph must contain a dedicated node with kind `query_target` for the value or object requested by the question, and that node must occur on the support path into the answer. If it is missing or disconnected, select the existing edge into the answer so interrogation can add or connect it.

Anchor a missing premise to the inference that needs it:
- If an existing edge cannot entail its target without a missing premise or query constraint, select that edge, even when the missing premise has no node yet.
- In particular, select the existing edge into the answer when it omits a domain, optimization, comparison, or selection constraint needed to justify that answer.
- Use coverage only when the missing support cannot be attached to any existing edge, such as when the graph has no edge reaching the answer or an entire connection is absent.
- Never select coverage merely because the needed premise is absent as a node.

A premise must be a declarative fact or an operative query-constraint fragment. A node that copies the entire interrogative question is not an operationalized premise; select the edge that relies on it.

Do not solve the problem from scratch. Do not use an answer key. Do not decide whether the answer is correct.

Return only JSON:
{
  "nodes": ["n1"],
  "edges": ["e1"],
  "coverage": true,
  "reasons": {"n1": "short reason", "e1": "short reason", "coverage": "short reason"}
}

Only use node and edge IDs present in the input graph. Use coverage=true only when no single existing edge can be repaired to capture the gap.
