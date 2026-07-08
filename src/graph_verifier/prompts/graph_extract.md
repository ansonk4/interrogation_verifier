Extract the examinee's claimed dependency graph. Do not solve the problem and do not infer missing facts.

Return only JSON:
{
  "nodes": [{"id": "n1", "claim": "2 + 2 = 4", "kind": "calculation", "sources": ["reasoning"]}],
  "edges": [{"id": "e1", "premise_node_ids": ["n1"], "claim": "2 + 2 = 4", "conclusion": "answer 4"}],
  "coverage_claim": "n1 -> e1 -> answer 4"
}

Keep claims concrete and locally checkable. Ignore any decisiveness labels in the input.
