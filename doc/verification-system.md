# Verification System

The verifier decides whether an agent's answer is supported by a complete, checkable reasoning path. It does not compare against the expected answer.

## Flow

```text
question + agent answer + reasoning
  -> dependency graph
  -> targeted interrogation
  -> decisive subgraph
  -> node and edge verification
  -> coverage check
  -> verifier-targeted interrogation when repairable debt remains
  -> final status
```

### 1. Build the graph

The model extracts:

- **Nodes:** premises, calculations, comparisons, and the explicit answer.
- **Edges:** reasoning steps connecting premises to conclusions.
- **Coverage:** the claimed path from the question to the answer.

Everything extracted is initially untrusted.

### 2. Interrogate missing support

The system handles one unresolved target per round. The default limit is 20 rounds.

#### How a target is selected

1. Local checks propose obvious candidates:
   - nodes containing vague wording such as `clearly`, `obvious`, `some`, `about`, or `roughly`;
   - edges with no premises, no target, or a target ID missing from the graph.
2. A target-selection LLM reviews the question, answer, reasoning, current graph, and local candidates. It looks for vague claims, unsupported jumps, missing rules or premises, and missing query constraints.
3. LLM and local candidates are merged. The first target not already handled is selected in this order:
   1. node;
   2. edge;
   3. coverage.

A missing premise is anchored to the existing edge that needs it. If the reviewer returns only a coverage gap but exactly one edge enters the explicit answer node, that edge is selected instead. Coverage remains the target when there is no unique existing answer edge to repair.

The prompt requests at most one candidate, but the code safely handles multiple candidates by taking one and selecting again from the updated graph next round. New target IDs are handled before retries. A target can be revisited only when its claim, premises, target, or connected support changed; an unchanged target is not retried. The round limit bounds refinement.

#### How a repair is applied

The configured original-agent model proposes graph edits for the selected target. Its model configuration must be recorded as `agent_model_config` on the case; without it, the target becomes debt rather than being repaired by the reviewer. The agent's JSON response and model configuration are saved in the interrogation artifact.

The edit is first applied to a copy of the graph and is committed only when all checks pass:

- every new or changed node has provenance: an exact non-interrogative question quote, a locally verified calculation, or an incoming support edge;
- at most one new node and one new edge;
- the target materially changes or gains direct support;
- added items are connected to the target or existing answer path;
- changes to existing items stay within the selected target's support path;
- duplicate items are reused; conflicting IDs, broken references, unrelated edits, and alternate answer branches are rejected.

Question requirements are represented as exact operative fragments, such as `least positive integer value of x`, with `kind=query_constraint`. Entire interrogative questions and already-derived answers are not premise nodes. The edge verifier receives these grounded fragments as explicit requirements without receiving the original question.

If the LLM explicitly returns the target as debt, extra edits are discarded and only that target is marked. Invalid or unresolved repairs are also reduced to localized target debt instead of expanding the graph.

#### When interrogation ends

The proactive interrogation pass stops when:

- the input graph already has tool debt;
- target selection finds no unhandled target;
- target selection or agent repair fails;
- the round limit is reached.

After the final allowed round, one extra target-selection call checks whether work remains. Remaining work or a failed final check becomes tool debt. A rejected or unchanged target is not retried; a materially changed target may receive another refinement round.

#### LLM calls

For a case with `agent_model_config`, target selection is a reviewer call and repair is an original-agent call:

- a round with a target uses **2 calls**: target selection + repair;
- a terminating round with no target uses **1 call**;
- exhausting the round limit adds **1 final target-selection call**.

Therefore:

- no target on the first check: `1` call;
- `k` handled targets followed by no target: `2k + 1` calls;
- all `N` rounds used: `2N + 1` calls, so the default `N = 20` allows at most `41` interrogation calls.

Without `agent_model_config`, no repair call is made. Each selected target is marked as debt and the reviewer continues looking for another target.

Graph extraction, decisiveness review, and edge verification are separate stages and are not included in these counts.

The same `max-interrogation-rounds` budget is shared with verifier-targeted feedback. An empty target-selection check does not consume a repair round; each selected proactive or verifier target does.

### 3. Mark decisive items

An independent review selects the minimal reasoning path needed for the final answer. Disconnected, duplicate, alternative, and merely explanatory items are removed from the decisive subgraph.

### 4. Verify nodes and edges

Nodes are verified by exact question grounding or local arithmetic/comparison checks. An answer node cannot validate itself; it requires a verified incoming edge.

Edges are checked only after all their premises are valid. Locally checkable edges use the built-in evaluator. Symbolic or natural-language edges use a restricted LLM check containing only:

- verified premise claims;
- the edge claim;
- the target claim.

The edge verifier does not receive the original question, agent reasoning, expected answer, or answer key.

### 5. Check coverage

Coverage is valid only when there is a complete path of decisive, valid nodes and edges ending at an explicit answer node that exactly matches the agent's answer.

### 6. Feed repairable verification debt back

After verification, the system stops immediately for a reliable graph, a decisive refutation, or tool failure. Otherwise it walks backward from the answer and selects one causal debt item:

- a debt edge whose premises are valid;
- an unsupported root node when downstream edges fail only because of that node;
- coverage only when no more specific node or edge failure exists.

Answer-node debt, downstream `unverified premise` edges, and generic coverage debt are treated as symptoms while an upstream cause exists. The exact verifier reason and graph item are forced into the original-agent interrogation call, bypassing ordinary target selection for that repair. A structural repair is followed by a fresh decisiveness review and complete verification on the same graph. Graph extraction is never repeated.

Feedback stops on reliability, refutation, tool failure, explicit agent debt, an unchanged target signature, a previously handled signature, or the shared round limit.

### 7. Produce the final status

| Status | Meaning |
|---|---|
| `verified_reliable` | Every decisive item and the complete answer path are verified. |
| `answer_refuted` | A decisive claim or reasoning step is proven false. |
| `coverage_debt` | The graph does not fully reach the answer. |
| `node_debt` | A decisive node could not be verified. |
| `verification_debt` | A decisive edge or verification tool could not be verified. |

Debt means **insufficient verified support**, not that the answer is necessarily wrong.

## Safety properties

- The expected answer is never used during verification.
- Matching numbers alone never validates a claim.
- Duplicate graph IDs and malformed LLM responses fail closed.
- Only decisive edges can validate derived nodes.
- Reported counts include decisive items, coverage, and tool debt—not irrelevant graph noise.
