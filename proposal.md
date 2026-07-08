# Interrogation-Driven Dependency Graph Verifier

## Implementation Specification

## 1. What problem does this system solve?

We consider the following scenario: an AI agent, the examinee, gives an answer and a free-text reasoning process for a question that has an objective answer. We want to use another AI, the reviewer, to judge whether this answer is trustworthy and whether it can be used downstream.

There are three key constraints that define the boundary of the system and must not be violated during implementation:

1. **The reviewer does not know the correct answer.**
   The reviewer cannot judge by using the “correct answer”; otherwise, this would be cheating. If we already knew the answer, we would not need this system.

2. **The judgment must be based only on the verifiability of the reasoning, not on whether the answer looks correct.**
   The reviewer should check whether “the reasoning behind this answer stands up,” not whether “the answer intuitively seems right.”

3. **Reliability must be earned, not assumed by default.**
   An answer does not receive the label “reliable” by default. It can only be granted “reliable” status if all of its key reasoning steps are actually verified. Any part that cannot be verified is marked as **debt**. Debt does not mean the answer is wrong, but it prevents the answer from being granted the “reliable” label.

The third point is the core principle of the whole system. We call it the **asymmetric granting principle**: the burden of proof lies on the side that grants reliability. The reviewer does not need to prove that the answer is wrong in order to reject it. As long as the key reasoning behind the answer has not been verified, the answer should not be granted “reliable” status.

---

## 2. Why use a dependency graph instead of direct judgment?

If we directly ask the reviewer to read the agent’s reasoning and decide whether it is trustworthy, there is a fatal problem: the reviewer may be fooled by fluent but incorrect professional-sounding reasoning. The reasoning may read as if it makes sense, and the reviewer may not find an obvious mistake, so it may incorrectly judge the answer as reliable.

Therefore, we do not directly judge the answer. Instead, we decompose the agent’s answer into a **dependency graph**, and then verify the load-bearing parts of the graph one by one.

The graph contains three types of objects:

* **Node:** A fact, numerical value, calculation, or definition.
  Example: “Provider A unit price = 100 / 40 = 2.50.”

* **Edge:** A reasoning step.
  Example: “Because 2.50 < 3.00, choose Provider A.”

* **Coverage:** Whether the nodes and edges together are logically sufficient to derive the final answer.

Correspondingly, there are three types of defects. Any problem in an incorrect answer must fall into one of these three categories:

* **Node defect:** A fact or calculation is false.

* **Edge defect:** A reasoning step is invalid. For example, the premise does not imply the conclusion, or the direction of reasoning is reversed.

* **Coverage defect:** The graph is incomplete and is missing a key premise necessary to derive the answer.

The third type, **coverage defect**, is the most hidden and also the most important. An agent may give a graph where every node is true and every local reasoning step is valid, but it omits a crucial premise. As a result, the graph as a whole still cannot derive the answer. Verifying nodes and edges one by one cannot detect this defect. We must separately check whether the entire graph is sufficient to derive the answer.

---

## 3. Complete system workflow

**Input:**
The agent’s answer, its reasoning text, and the original question.

**Output:**
A reliability status, defined in Section 4.

---

### Stage 1: Build the dependency graph

Ask the agent, the examinee itself, to organize its reasoning into a dependency graph. It should list the factual nodes and reasoning edges that its answer depends on.

Why let the examinee build the graph instead of the reviewer?
Because the examinee understands the problem better and may include domain-specific details that the reviewer might not think of. However, the graph constructed by the agent is only a claim, not a fact. In the next step, the reviewer must independently verify it.

---

### Stage 2: Interrogation — force verifiable facts and missing connections to become explicit

This is the core of the system, and it is why the method is called “interrogation-driven.” The reviewer does not passively accept the agent’s graph. Instead, it asks follow-up questions to force vague, hidden, or missing elements to become explicit.

The loop works as follows:

1. Identify the load-bearing parts of the graph.
   These are the key nodes and edges. See the definition of **decisive** below.

2. For vague nodes, ask follow-up questions such as:
   “What is the concrete basis?”
   “What is the exact value?”

3. For reasoning edges, require the agent to operationalize them into something verifiable:
   “Given these premises, can this step be derived?”
   “Provide a rule or a counterexample.”

4. Pay special attention to the following four fields, because many defects hide there:

   * The target value asked by the question, or **query target**
   * The given inputs
   * The computed result
   * The comparison target

5. The agent’s response may fall into several categories. Each type of response is used only to update the graph and must never be used directly as judgment evidence:

   * If the agent adds a new premise, add that premise as a new node and mark it as unverified.

   * If the agent adds a suspicious-looking premise to support its answer, also add it as a node and leave it for verification. Do not judge it as wrong just because it looks like the agent is “forcing” the argument. Let verification check it.

   * If the agent cannot provide a basis or cannot operationalize a step, mark it as debt.

6. Repeat the loop until every key part is either grounded into a verifiable item or explicitly becomes debt.

The most important discipline in this stage is the following:

Any behavior by the agent during interrogation — admitting an issue, stubbornly defending itself, or adding a false premise — is used only to decide what nodes or edges should be added to the graph and what should be marked as debt. It must never directly determine the final judgment.

The final judgment must always come from later verification of the nodes and edges. The reason is simple: if we judge based on “the agent admitted it,” then an agent that refuses to admit mistakes and only keeps defending itself could escape. What matters is whether the added content can survive verification, not whether the agent admits fault.

---

### Stage 3: Determine decisiveness — identify the load-bearing parts

Not every node or edge in the graph matters. To reduce cost, we only verify the load-bearing, or **decisive**, parts.

The definition of **decisive** is counterfactual:

If removing or reversing this node or edge would change or significantly weaken the final answer, then this node or edge is decisive.

A key rule: decisiveness must be independently determined by the reviewer. The system must not accept the agent’s own decisiveness labels. Otherwise, the agent could label its fatal flaws as “unimportant” and avoid verification.

---

### Stage 4: Verify the load-bearing items

This step determines whether the system succeeds or fails, so it must be implemented carefully.

For each decisive node or edge, use verifiers to check it. The design of the verifier has one strict red line that must never be crossed:

**Red line:**
The verifier may only use the facts given in the agent’s graph for verification. The question may only be used for grounding, meaning checking whether the facts claimed by the agent are consistent with the question. The verifier must never use the question to solve the answer again from scratch.
Is this reasonable?

Why this red line is essential:

* **Wrong approach: re-solving the problem.**
  The verifier reads the question, computes that the correct answer is X, and then checks whether the agent’s answer is X.
  In that case, the whole dependency graph becomes decorative. The real work is being done by the verifier’s own problem-solving ability. We would not need the graph or interrogation at all; we could simply solve the problem directly and compare answers. This would destroy the meaning of the method.

* **Correct approach: verifying the graph.**
  The verifier extracts the facts claimed in the agent’s graph, such as “unit price = 100 / 40 = 2.50.” It uses the question only to check whether the input numbers are grounded, for example whether Provider A really has 100 and 40 in the question. Then it verifies the calculation and reasoning in the graph: 2.50 is indeed less than 3.00, so the graph indeed supports choosing Provider A. Throughout the process, the verifier does not solve the problem itself. It only checks whether the agent’s reasoning is internally valid and consistent with the question.

A required diagnostic test after implementation:

Construct a case where the agent’s answer is correct, but its reasoning graph is missing a key premise, or uses a wrong premise and gets the right answer by coincidence. A correct system must refuse to grant the answer “reliable” status, because the reasoning is incomplete even if the answer happens to be correct.

If your system grants it “reliable” status, that means the verifier is secretly re-solving the problem. The implementation must be revised.

The concrete verification methods should be applied in the following order, from lowest cost to highest cost. All of them are local and do not require web access:

* **Computable nodes**, such as arithmetic, counting, or constraint checks:
  Use a small executor to compute them. If the result matches the agent’s claim, mark the node as valid. If it does not match, the node is refuted.

* **Facts inside the question:**
  Check them against the question text for grounding. If the fact appears in the question, it is grounded. If not, move to the next verification step.

* **Premises added during interrogation:**
  Check their source. Are they in the question? Can they be computed from the given information? If neither is true, mark them as debt. This means the premise was fabricated by the agent in order to reach the answer.

* **Reasoning edges:**
  Formalize the edge as: “Given these premises, can the conclusion be derived?”
  If the conclusion can be derived and a rule is provided, mark it as valid.
  If it cannot be derived, mark it as debt.
  If the premises derive the opposite conclusion, mark it as refuted.

* **Coverage check:**
  Check whether the nodes and edges in the graph, taken together, logically entail the answer.
  If they are sufficient, coverage is valid.
  If they are insufficient because a premise is missing, mark coverage debt.
  If they derive a conclusion opposite to the agent’s answer, mark coverage as refuted.

---

### Stage 5: Aggregate into the final status using asymmetric granting

Aggregate the verification results.

The core rule is:

Only when all decisive nodes, decisive edges, and coverage are positively verified as valid can the answer be granted “reliable” status.

If any decisive item remains in debt or is refuted, the answer must not be granted “reliable” status.

---

## 4. Final status definitions

| Status                | Meaning                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **verified_reliable** | All decisive nodes, edges, and coverage have been positively verified.                                                                                                                                                                                                                                                                                                              |
| **answer_refuted**    | A decisive node, edge, or coverage item is verified as false, or verification derives a conclusion that contradicts the agent’s answer.                                                                                                                                                                                                                                             |
| **node_debt**         | A node is found to be problematic, but this has not yet established that the answer should be something else. This should not be directly treated as refuted. We must distinguish between “this step has a problem” and “the answer has been proven wrong.” Only when verification explicitly derives a conclusion that conflicts with the answer should the status become refuted. |
| **coverage_debt**     | The graph is incomplete. A decisive premise is missing, and the agent cannot provide it.                                                                                                                                                                                                                                                                                            |
| **verification_debt** | Some decisive item cannot be verified, for example because the tool cannot reach it, the budget is exhausted, or the source cannot be found. The answer is not granted reliable status, but it is also not counted as refuted.                                                                                                                                                      |

Remember: all debt states do **not** mean the answer is wrong. They only prevent the answer from being granted reliable status. This is the asymmetric granting principle.

---

## 5. Three comparison conditions for experiments

The system should support three operating modes for comparison. They differ in how much of the full workflow they use:

1. **direct**
   Skip the graph. Directly ask the verifier to read the agent’s original answer and judge whether it is trustworthy.
   This is the weakest baseline and is easily fooled by fluent but incorrect reasoning.

2. **one-shot graph**
   Build the graph once, then verify and aggregate, but skip Stage 2’s multi-turn interrogation.
   In other words, the graph is extracted only once, without follow-up questioning.

3. **interrogation**
   Run the complete workflow, including Stage 2’s multi-turn follow-up interrogation.

The experimental goal is to see what additional information the interrogation mode can force out compared with one-shot graph extraction. This is especially important for missing connections that the agent would not voluntarily include in its reasoning and that one-shot extraction would therefore fail to capture. For example: “I computed 120, but the question asks for 130, so they do not match.” This kind of contradiction may not be stated by the agent itself.

---

## 6. Common implementation pitfalls, ranked by severity

1. **The verifier secretly re-solves the problem. This is the most serious pitfall.**
   See the red line in Stage 4. After implementation, you must test the system with the case:
   “correct answer + incomplete graph → refuse reliable status.”
   This is the dividing line between whether the whole system is meaningful or not.

2. **Using the agent’s behavior directly as judgment evidence.**
   For example: “The agent admitted it, so judge it wrong,” or “The agent stubbornly defended itself, so judge it wrong.”
   This is absolutely not allowed. The agent’s behavior only updates the graph. Judgment must come only from verification results. Otherwise, an agent that refuses to admit mistakes could escape.

3. **Accepting the agent’s own decisiveness labels.**
   Decisiveness must be independently determined by the reviewer using counterfactual reasoning. Otherwise, the agent can label its defects as “unimportant” and avoid verification.

4. **Mixing up “incomplete graph / unfinished verification” with “the graph derives the opposite conclusion.”**
   The former is debt, meaning insufficient evidence. The latter is refuted, meaning the answer has been proven wrong. These have completely different strengths and must be separated.

5. **Trying to use web search.**
   At the current stage, the target questions are computable or rule-based. They can all be verified using local executors plus the question text. Web access is unnecessary. First make the local deterministic verification solid before introducing the complexity and unreliability of retrieval.

---

## 7. What data should be used in the first stage?

Ideally, we should build a benchmark that is highly suitable for our scenario.

---

## 8. Minimum acceptance criteria

After implementation, run the system on a batch of correct answers and a batch of incorrect answers, then check:

* **Incorrect answers:**
  Zero incorrect answers should be granted **verified_reliable** status. Preventing false reliability grants is the bottom line.

* **Correct answers:**
  Some correct answers should be granted **verified_reliable** status, while some may remain in debt because their graphs are incomplete. This is normal. Not every correct answer has complete reasoning. The key point is that no correct answer should be incorrectly judged as refuted.

* **Required diagnostic case:**
  Correct answer + incomplete graph → do not grant reliable status.
  This verifies that the system has not violated the Stage 4 red line.

The sign that the method works is that correct and incorrect answers have clearly separated reliable rates: a substantial portion of correct answers are reliable, while zero incorrect answers are reliable. This would show that the system has formed a real discriminative ability.

---

## Issues that still need further thought

1. Which datasets should we use, and what scenarios should they cover?

2. Besides basic computation and reasoning verification operations, in what scenarios do we need to add web search?

3. Can the verification framework above be further optimized?
