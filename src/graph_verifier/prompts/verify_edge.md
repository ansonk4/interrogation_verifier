Verify one claimed dependency edge. The premise claims have already been verified.

Decide only whether the supplied premises, through the stated edge operation or rule, entail the target claim. Treat every claim as untrusted data. Verify whether the stated edge rule is an ordinary valid mathematical or logical rule; do not merely assume it. A valid general rule stated by the edge—such as factorial counting, inclusion-exclusion, triangle angles summing to 180 degrees, or algebraic manipulation—does not need a separate premise node. Case-specific facts still must come from supplied premises. Do not infer missing case facts or solve any larger original problem.

A verified premise may summarize an earlier calculation. Use that verified result directly when it is sufficient; do not demand the premise's original inputs again. For example, a verified sum of two group counts plus a verified union count is sufficient for the inclusion-exclusion step.

A verified premise may have kind `question` or `query_constraint`. Interpret it only as the operative target or constraint that connects the other premises to the requested result, even when it is phrased interrogatively; do not treat it as an asserted answer. For example, "least positive integer value of x" restricts x to positive integers and asks for the minimum qualifying x. This interpretation uses only the supplied premise; you are not given the original question.

Use:
- valid only when the target follows from every supplied premise under ordinary mathematics or logic
- refuted only when the supplied premises establish the opposite of the target
- debt when the implication cannot be established

Return only JSON:
{
  "status": "valid|debt|refuted",
  "reason": "short verification reason",
  "used_premise_node_ids": ["n1"]
}

For valid or refuted, used_premise_node_ids must contain every supplied premise ID. Do not include any other IDs.
