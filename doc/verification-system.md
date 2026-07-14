# Verification System

The verifier decides whether an agent's answer is supported by a complete, checkable reasoning
path. It never compares against the expected answer.

## Flow

```text
question + agent answer + reasoning
  -> dependency graph
  -> complete answer-related candidate cone
  -> node and edge verification
  -> smallest verified proof selection
  -> verifier-targeted interrogation when repairable debt remains
  -> repeat verification or produce final status
```

### 1. Build the graph

The model extracts nodes, dependency edges, an explicit answer node, and a coverage claim.
Everything extracted is initially untrusted. Supplied decisiveness labels are ignored.

### 2. Prepare answer candidates

Candidate selection is deterministic. Starting from every explicit answer node that matches the
agent answer, the system walks backward through every incoming edge and premise. This produces the
complete answer-related support cone while excluding disconnected branches.

Candidate flags are temporary. They permit all answer-related alternatives to be checked; they do
not yet assert that an item is part of the final proof. Candidate selection does not call an LLM.

### 3. Verify candidates

Nodes are verified by exact question grounding or local arithmetic and comparison checks. An answer
node cannot validate itself; it requires a verified incoming edge.

An edge is an AND dependency: every listed premise must be valid and the edge rule must entail its
target. Multiple incoming edges are OR alternatives: any one valid edge can support the target.

Locally checkable edges use the built-in evaluator. Other candidate edges use a restricted LLM
check containing only verified premise claims, the edge claim, and the target claim. The edge
verifier does not receive the original question, reasoning, expected answer, or answer key.

### 4. Select the decisive proof

Only after verification does the system assign final decisive flags. It searches the candidate cone
backward from the answer and retains the best proof, ordered by:

1. fewer refuted items;
2. fewer debt items;
3. inclusion of a query target;
4. fewer total nodes and edges;
5. stable ID ordering for deterministic ties.

Consequently, a complete valid proof wins over refuted or unresolved alternatives. If no valid proof
exists, the least-defective answer path remains decisive so repair can target its nearest causal
failure rather than a downstream symptom.

### 5. Repair causal verification debt

The system stops immediately for a reliable proof, a decisive refutation, or a required tool
failure. Otherwise it walks backward along the selected path and chooses one causal debt item:

- a debt edge whose premises are valid;
- an unsupported root node when downstream edges are blocked by it;
- the unique answer edge for a missing or disconnected query target;
- coverage only when no more specific item exists.

The exact verifier reason and graph item are sent to the configured original-agent model. A repair
may not disconnect the canonical answer from its existing support. Accepted edits are followed by a
fresh candidate pass and complete verification on the same graph; extraction is never repeated.

The initial graph is verified before any repair is requested. The configured round limit is shared
across all verifier-targeted repairs.

### 6. Check coverage and report status

Coverage is valid only when the selected proof consists of valid nodes and edges, ends at an explicit
answer node matching the agent answer, and contains a query target.

Answer matching preserves mathematical punctuation and compares exact numeric structures, including
decimals, rational or LaTeX fractions, and coordinate tuples. Prose normalization is not used to
rewrite answer values.

| Status | Meaning |
|---|---|
| `verified_reliable` | The selected complete answer proof is valid. |
| `answer_refuted` | The selected answer path contains a proven false item. |
| `coverage_debt` | The selected graph does not completely connect the query to the answer. |
| `node_debt` | A decisive node could not be verified. |
| `verification_debt` | A decisive reasoning edge could not be verified. |
| `tool_error` | A required model or verification tool failed before a valid proof was established. |

Debt means insufficient verified support, not that the answer is necessarily wrong.

LLM responses must contain string content holding a JSON object. Null, malformed, and failed
responses are retried up to three times with exponential backoff. A tool error after a complete valid
proof is retained in the artifact but does not override `verified_reliable`.

## Safety properties

- The expected answer is never used during verification.
- Matching numbers alone never validates a claim.
- Duplicate graph IDs and malformed LLM responses fail closed.
- Disconnected branches cannot become decisive.
- A refuted alternative cannot poison a separate valid answer proof.
- Interrogation cannot retarget the only answer-support edge away from the canonical answer.
- Reported counts include only the selected proof, coverage, and tool errors.
