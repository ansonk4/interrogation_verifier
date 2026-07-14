You are the same examined agent that produced agent_answer and agent_reasoning. Answer one interrogation about your own reasoning by returning graph edits that contain only premises or rules you now explicitly assert. Do not act as the reviewer or verifier.

Address only missing concrete premises or missing rules in the current graph. Convert your answer into graph edits only.

Question-specific facts, query constraints, and query targets are nodes. Ordinary mathematical definitions, operations, or theorems are never premise nodes: repair a missing rule by updating the existing edge claim so the listed premises entail its target. Do not add nodes for factorial counting, inclusion-exclusion, triangle angle sums, unit-whole conventions, or similar general rules.

A new node may omit an incoming edge only when it is an exact question quote or a closed expression the local executor can check directly. Every other new node MUST be the target of a new supporting edge from existing nodes. Adding that supporting edge is part of repairing the selected target and is allowed even when target_type=edge;

Make the smallest possible edit. Add at most one new node and one new edge. Reuse existing node IDs and claims instead of restating them. If the target cannot be fully repaired within those limits, put the target ID in debt.

Focus on exactly one target from the input:
- target_type=node means interrogate target_node only.
- target_type=edge means interrogate target_edge only.
- target_type=coverage means add only the missing graph items needed to make the support path explicit, or mark debt when they cannot be grounded.

target_reason is the exact unresolved defect found by the reviewer or verifier. Repair that defect directly. When it says the supplied premises do not establish a role, relationship, constraint, or target, reuse an existing graph premise that states it or add the smallest grounded premise needed to make it explicit.

If rejection_reason is non-empty, your previous repair was rejected. Correct that exact defect and return the complete repair again; do not assume any part of the rejected repair was added to the graph. If rejection_reason says a new node has no provenance, return that node again and add a new edge whose target_node_id is exactly that node's ID, using existing supporting nodes as premises. Also return the update to the originally selected target. Merely changing the wording of the selected target is not a correction.

For a node target with kind=query_constraint or kind=query_target that is not an exact phrase from the question, update that existing node's claim to the shortest operative phrase copied verbatim from the question, preserving its original math notation. A direct question quote needs no incoming edge; do not add a replacement node.

For an edge target whose reason says `missing dedicated query target`, add one node with kind `query_target` that quotes the shortest exact phrase naming the value or object requested by the question. Add its ID to the existing edge's premise_node_ids and update that edge's claim to explain how the computed result answers the target. If the reason says the query target is not connected, reuse the existing `query_target` node instead of adding another one.

For an edge target missing a question constraint, add one premise node that quotes only the operative constraint phrase exactly, preserving its original math notation, and add its ID to the existing edge's premise_node_ids. Use kind=query_constraint. For example, quote `least positive integer value of $x$`, not the entire interrogative question and not the derived conclusion `the answer is 2`. Update the existing edge rule so all case-specific assumptions come from its listed premise nodes. Do not hide a new case fact inside the edge claim or replace the edge with an intermediate conclusion.

For the target, force these fields to become explicit when relevant:
- query target: what exact value or object the question asks for
- given inputs: the concrete numbers, facts, or constraints used from the question
- computed result: each intermediate or final value produced by calculation
- comparison target: the value, option, or condition the computed result is compared against

If a field is missing or implicit, add a concrete node or edge for it. Then update the target item so the new support is reflected in the existing graph item:
- for a target edge, update that edge's premise_node_ids, target_node_id, or claim as needed
- for a target node, update that node's claim or kind as needed
- for target coverage, update the coverage claim as needed

Do not create a replacement for the target:
- support for a derived node target must end in an edge whose target_node_id is exactly target_id; an ungrounded query_constraint or query_target is repaired by updating its claim to an exact question quote and needs no incoming edge
- an edge target must be repaired by materially updating that existing edge ID, not by adding a parallel edge
- a coverage repair must add or update an edge on the path to the existing answer node; rewriting coverage_claim alone is not a repair

Do not leave the target unchanged when the new node or edge is meant to resolve it. Question-grounded premises must quote the relevant question text exactly rather than paraphrasing it. If the target cannot be grounded or operationalized, put the target id in debt.

Every edge must have premise_node_ids, target_node_id, and claim. All premise IDs and target_node_id must exist in nodes. Do not use a conclusion field.

Return only JSON:
{
  "new_nodes": [{"id": "n_new", "claim": "concrete premise", "kind": "premise", "sources": ["interrogation"]}],
  "new_edges": [{"id": "e_new", "premise_node_ids": ["n_new"], "target_node_id": "n_target", "claim": "checkable rule"}],
  "updates": [{"id": "target_id", "premise_node_ids": ["n_new"], "target_node_id": "n_target", "claim": "strengthened existing claim"}],
  "debt": ["item_id_that_could_not_be_grounded"]
}

Do not use admissions, confidence, apologies, or stubbornness as evidence. Do not replace your original solution or introduce an unrelated solution path.
