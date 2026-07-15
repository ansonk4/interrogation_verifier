# GRACE Error Taxonomy

GRACE organizes step-level context faithfulness errors into two tracks, each with four categories. The taxonomy was discovered bottom-up via unsupervised clustering of free-form LLM critiques over 30K+ unfaithful reasoning steps.

---

## GRACE-Inference (Deductive Reasoning Errors)

Covers: ReClor, LogiQA

These errors occur when the model correctly identifies relevant premises from the context but applies invalid logical operations.

### Reversed Reasoning (`reversed_reasoning`)

**Annotation question:** Did the step get the direction of a relationship backwards?

**Definition:** The step reverses or confuses the direction of a logical relationship. This includes swapping cause and effect, treating a sufficient condition as necessary (or vice versa), affirming the consequent, denying the antecedent, or inverting if-then implications.

### Wrong Argument Reading (`wrong_argument_reading`)

**Annotation question:** Did the step misidentify what the argument is saying or doing?

**Definition:** The step misidentifies the logical structure, roles, or components of an argument. This includes confusing premises with conclusions, mislabeling the type of flaw in an argument, misidentifying the point of disagreement between speakers, or treating distinct concepts as semantically equivalent without textual support.

### Rule Violation (`rule_violation`)

**Annotation question:** Did the step ignore or break a constraint stated in the text?

**Definition:** The step violates an explicit rule, constraint, or boundary stated in the context. This includes breaking one-to-one mappings, ignoring mutual exclusivity, incorrect set operations, invalid elimination of candidates, violating disjunction logic, or failing at algebraic/combinatorial constraint satisfaction.

### Overreaching Claim (`overreaching_claim`)

**Annotation question:** Did the step claim more than the evidence actually supports?

**Definition:** The step extends a conclusion beyond what the premises logically support. This includes overgeneralizing from specific cases, misapplying quantifier scope, confusing relative with absolute claims, making unjustified causal attributions, drawing normative conclusions from descriptive facts, or treating absence of evidence as evidence of absence.

---

## GRACE-Grounding (Factual Grounding Errors)

Covers: MuSiQue, 2WikiMHQA

These errors occur when the step's factual claims are not faithful to what the context states.

### Groundedness Violation (`groundedness_violation`)

**Annotation question:** Does the step claim something not supported by the context?

**Definition:** The step makes a claim — whether a fabricated detail, a true-but-unsourced fact from training data, or a plausible-but-ungrounded inference — that is not supported by the provided context.

### Contradiction (`contradiction`)

**Annotation question:** Does the context explicitly say the opposite?

**Definition:** The step directly and unambiguously opposes an explicit statement in the context. The context says X; the step says not-X.

### Confusion (`confusion`)

**Annotation question:** Did it mix up entities, facts, or relationships?

**Definition:** The step uses information that IS in the context, but attaches it to the wrong entity, merges distinct entities, reverses a relationship direction, or confuses properties between entities.

### Evidence Neglect (`evidence_neglect`)

**Annotation question:** Does it ignore or deny available information?

**Definition:** The step claims information is missing or unavailable when it IS present, or fails to track entity state changes across a narrative.
