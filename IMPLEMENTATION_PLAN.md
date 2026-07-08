# Implementation Plan: Interrogation-Driven Dependency Graph Verifier

## Goal

Implement the MVP described in `proposal.md`: given a question, an examinee answer, and its reasoning, build and verify a dependency graph, then grant `verified_reliable` only when every decisive node, decisive edge, and coverage item is positively verified.

The MVP should optimize for inspectability, not framework completeness.

## Non-Negotiable Rules

1. The verifier must not use the answer key or solve the original question from scratch.
2. The question text may only be used to ground claimed facts from the graph.
3. Interrogation responses may update the graph, but must never directly determine the final status.
4. Decisiveness must be assigned by the reviewer, not accepted from the examinee.
5. Debt is not refutation. Only explicit contradiction or a verified false decisive item that invalidates the answer becomes `answer_refuted`.
6. No web access in the verification workflow.

## MVP Dataset Choice

Use a deterministic subset of the Hugging Face dataset `qwedsacf/competition_math`, stored locally as JSONL after preparation.

Reason: this is the MATH competition dataset, with each item containing a problem, a step-by-step solution, difficulty level, and subject type. It is suitable for graph verification because the prep script can derive benchmark cases from the solution while runtime verification is still forbidden from reading the answer key or re-solving the problem.

Source fields:

- `problem`: original question
- `solution`: step-by-step solution with final answer in `\boxed{...}`
- `level`: difficulty from `Level 1` to `Level 5`
- `type`: subject area

Prepare `data/mvp_cases.jsonl` from `qwedsacf/competition_math` with 8-12 cases.

For the MVP, prefer `Level 1` and `Level 2` examples from `Prealgebra`, `Algebra`, and `Counting & Probability`, because they are more likely to be checkable with local arithmetic/comparison verification.

Prepared records should include:

- correct answer with complete reasoning
- correct answer with incomplete reasoning
- correct answer with a wrong premise that happens to land on the right answer
- incorrect answer from a bad calculation
- incorrect answer from a bad comparison
- incomplete graph that needs interrogation to expose a missing premise

Each record:

```json
{
  "id": "mvp_001",
  "question": "...",
  "agent_answer": "...",
  "agent_reasoning": "...",
  "expected_answer": "...",
  "source": {
    "dataset": "qwedsacf/competition_math",
    "split": "train",
    "level": "Level 1",
    "type": "Prealgebra"
  },
  "notes": "Only used for offline evaluation, never runtime verification."
}
```

`expected_answer` is extracted from the source `solution` and is allowed only for benchmark metrics after the verifier has produced its status.

Do not let runtime verification read `solution` or `expected_answer`.

## Target CLI

Keep one command:

```bash
uv run graph-verifier data/mvp_cases.jsonl --mode interrogation
```

Supported modes:

- `direct`
- `one-shot-graph`
- `interrogation`

Output one compact JSON object per case:

```json
{
  "id": "mvp_001",
  "mode": "interrogation",
  "status": "coverage_debt",
  "decisive": {"nodes": 3, "edges": 2, "coverage": true},
  "valid": 4,
  "debt": 1,
  "refuted": 0
}
```

## Source Layout

Use the existing `src/graph_verifier/` structure for package code. Keep dataset files in top-level `data/`.

```text
src/graph_verifier/
  main.py
  core/
    models.py
    graph.py
    verify.py
    aggregate.py
  prompts/
    graph_extract.md
    interrogate.md
    decisiveness.md
    direct.md
  utils/
    jsonl.py
    llm.py
data/
  prepare_math_subset.py
  mvp_cases.jsonl
tests/
  test_mvp.py
```

Do not add a database, service layer, plugin system, config framework, or async job runner.

## Dataset Preparation

Implement `data/prepare_math_subset.py`.

Command:

```bash
uv run python data/prepare_math_subset.py
```

Behavior:

1. Load `qwedsacf/competition_math` from Hugging Face.
2. Use the `train` split for MVP case construction.
3. Filter first to `Level 1` and `Level 2` rows in `Prealgebra`, `Algebra`, and `Counting & Probability`.
4. Deterministically select 8-12 rows with a fixed seed.
5. Extract `expected_answer` from the source `solution` only for offline evaluation.
6. Write `data/mvp_cases.jsonl`.

If Hugging Face loading requires a dependency, add it with:

```bash
uv add datasets
```

The generated JSONL is the runtime input. Runtime verifier code must not call Hugging Face or read source `solution`.

## Data Models

Define small dataclasses or Pydantic models only if already useful. Prefer stdlib dataclasses first.

Core objects:

- `Case`: `id`, `question`, `agent_answer`, `agent_reasoning`, optional `expected_answer`
- `Graph`: `nodes`, `edges`, `coverage_claim`
- `Node`: `id`, `claim`, `kind`, `sources`, `decisive`, `verification`
- `Edge`: `id`, `premise_node_ids`, `conclusion`, `claim`, `decisive`, `verification`
- `Verification`: `status`, `reason`, optional `evidence`

Verification statuses:

- `valid`
- `debt`
- `refuted`

Final statuses:

- `verified_reliable`
- `answer_refuted`
- `node_debt`
- `coverage_debt`
- `verification_debt`

## LLM Usage

Use the default endpoint from repo guidance:

```text
model/openrouter/hy3.json
```

Use the LLM only for:

1. graph extraction from examinee reasoning
2. interrogation questions and graph updates
3. independent decisiveness labeling
4. edge and coverage formalization when local code cannot do it directly

Never ask the LLM:

- "What is the correct answer?"
- "Solve this problem."
- "Does the agent answer match the expected answer?"

Every LLM prompt should require structured JSON output.

## Stage 1: One-Shot Graph Builder

Implement `core.graph.build_graph(case)`.

Prompt the examinee model to extract:

- factual nodes
- calculation nodes
- comparison nodes
- reasoning edges
- claimed coverage path from premises to answer

The graph builder may read the question, answer, and reasoning, but its output is only a claim. Nothing from this stage is trusted until verification.

Acceptance check:

- every graph has at least one node and one coverage claim
- malformed LLM JSON becomes `verification_debt`, not a crash

## Stage 2: Interrogation Loop

Implement `core.graph.interrogate(case, graph, max_rounds=3)`.

In each round:

1. Find vague decisive-looking nodes.
2. Find edges with unclear premises or missing rules.
3. Ask targeted follow-up questions.
4. Convert the response into graph edits only:
   - new premise -> new unverified node
   - new rule -> new unverified edge support
   - cannot provide basis -> debt marker

Stop when:

- all decisive-looking items are verifiable or debt, or
- `max_rounds` is reached

Do not use admissions, confidence, stubbornness, or apology language as evidence.

## Stage 3: Decisiveness

Implement `core.graph.mark_decisive(case, graph)`.

Reviewer prompt:

```text
For each node and edge, would removing or reversing it change or significantly weaken the final answer?
Return decisive=true only under that counterfactual test.
Do not reuse any decisiveness labels supplied by the examinee.
```

Local fallback:

- if a node is directly referenced by a decisive edge, mark it decisive
- if an edge directly concludes the answer or a comparison used by the answer, mark it decisive
- mark coverage decisive for every non-direct mode

## Stage 4: Verification

Implement `core.verify.verify_graph(case, graph)`.

Verification order:

1. computable nodes
2. facts grounded in question text
3. premises added during interrogation
4. reasoning edges
5. coverage

### Computable Nodes

Use local Python for arithmetic, counting, equality, ordering, and simple unit comparisons.

Allowed:

- parse arithmetic expressions claimed by the graph
- recompute `100 / 40 = 2.50`
- compare graph-claimed values, such as `2.50 < 3.00`

Not allowed:

- derive missing numbers from the original question unless the graph claims them
- solve the whole problem independently

### Question Grounding

Check whether claimed inputs appear in the question text.

For MVP, use conservative string/number matching:

- exact numbers
- simple normalized decimals/fractions
- exact named entities when present

If grounding is not found, leave debt unless another local computation verifies the claim.

### Added Premises

If an interrogation-added premise is not grounded in the question and not computable from verified graph facts, mark it `debt`.

### Reasoning Edges

For each decisive edge, verify:

- premises exist
- premise nodes are valid
- conclusion follows from the claimed operation, comparison, or rule

If the premises derive the opposite conclusion, mark the edge `refuted`.
If the conclusion cannot be derived, mark it `debt`.

### Coverage

Verify that the decisive valid nodes and edges entail the agent's answer.

For MVP, coverage is valid only when there is an explicit path:

```text
grounded/computed premises -> valid decisive edges -> agent_answer
```

If the path is missing, mark `coverage_debt`.
If the path derives a conclusion contradicting the answer, mark `answer_refuted`.

## Stage 5: Aggregation

Implement `core.aggregate.final_status(graph)`.

Rules:

1. Any decisive refuted node, edge, or coverage item -> `answer_refuted`.
2. Any decisive coverage debt -> `coverage_debt`.
3. Any decisive node debt -> `node_debt`.
4. Any decisive edge or tool/LLM failure debt -> `verification_debt`.
5. All decisive nodes, edges, and coverage valid -> `verified_reliable`.

Keep the reason string short and point to item IDs.

## Mode Behavior

### `direct`

Ask an LLM to judge trustworthiness directly from question, answer, and reasoning.

This mode is a baseline only. Its output must be labeled separately and must not share aggregation code with graph modes.

### `one-shot-graph`

Run:

1. graph extraction
2. decisiveness
3. verification
4. aggregation

Skip interrogation.

### `interrogation`

Run:

1. graph extraction
2. interrogation
3. decisiveness
4. verification
5. aggregation

## Required Tests

Create one small pytest file or stdlib assert-based test. Use `uv run pytest` only if pytest is added; otherwise use `uv run python tests/test_mvp.py`.

Minimum cases:

1. Complete correct graph -> `verified_reliable`
2. Correct answer with incomplete graph -> not `verified_reliable`, expected `coverage_debt`
3. Correct answer with wrong premise -> not `verified_reliable`
4. Incorrect arithmetic decisive node -> not `verified_reliable`
5. Refuted comparison that supports final answer -> `answer_refuted`
6. Interrogation adds unsupported premise -> debt, not refutation

The required diagnostic is case 2:

```text
correct answer + incomplete graph -> refuse reliable status
```

If this fails, the verifier is probably re-solving the problem and must be fixed before adding features.

## Implementation Order

1. Add data model objects in `core/models.py`.
2. Add JSONL loader in `utils/jsonl.py` and CLI mode parsing in `main.py`.
3. Add local verification helpers for arithmetic, comparison, and grounding in `core/verify.py`.
4. Add aggregation logic in `core/aggregate.py`.
5. Add deterministic tests using hand-written graph objects.
6. Add `data/prepare_math_subset.py` for `qwedsacf/competition_math`.
7. Generate `data/mvp_cases.jsonl`.
8. Add graph extraction LLM wrapper in `utils/llm.py` using `model/openrouter/hy3.json`.
9. Add one-shot graph mode.
10. Add interrogation loop.
11. Add direct baseline mode.
12. Run all modes on the MVP dataset and save no extra artifacts unless explicitly requested.

This order keeps the safety-critical verifier testable before LLM graph extraction is introduced.

## Done Criteria

The MVP is done when:

- `uv run python data/prepare_math_subset.py` creates `data/mvp_cases.jsonl`
- `uv run graph-verifier data/mvp_cases.jsonl --mode one-shot-graph` runs
- `uv run graph-verifier data/mvp_cases.jsonl --mode interrogation` runs
- no incorrect case receives `verified_reliable`
- at least one complete correct case receives `verified_reliable`
- the diagnostic incomplete-correct case receives debt, not `verified_reliable`
- runtime verification never reads `expected_answer`
- output is compact JSONL and easy to inspect

## Explicitly Out of Scope

- web retrieval
- full ProcessBench ingestion
- full MATH dataset ingestion
- UI
- database persistence
- multi-agent orchestration
- custom symbolic math engine
- broad natural-language theorem proving

Add these only after the MVP proves the asymmetric granting rule on the local subset.
