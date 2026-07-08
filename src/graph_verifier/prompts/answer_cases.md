You are the examinee model. Solve each problem independently from the question only.

Return only JSON in this shape:
{
  "cases": [
    {
      "source_row": 123,
      "agent_answer": "final answer only",
      "agent_reasoning": "concise step-by-step reasoning"
    }
  ]
}

Rules:
- Include every input case exactly once.
- Do not include a dependency graph.
- Do not mention answer keys, expected answers, datasets, or offline evaluation.
- Keep reasoning concise but complete enough for another system to verify.
