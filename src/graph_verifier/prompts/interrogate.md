Ask only for missing concrete premises or missing rules in the current graph. Convert the answer into graph edits only.

Return only JSON:
{
  "new_nodes": [{"id": "n_new", "claim": "concrete premise", "kind": "premise", "sources": ["interrogation"]}],
  "new_edges": [{"id": "e_new", "premise_node_ids": ["n_new"], "claim": "checkable rule", "conclusion": "answer support"}],
  "debt": ["item_id_that_could_not_be_grounded"]
}

Do not use admissions, confidence, apologies, or stubbornness as evidence. Do not solve the original problem.
