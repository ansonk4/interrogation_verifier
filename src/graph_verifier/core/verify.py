from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

from graph_verifier.core.models import DEBT, REFUTED, VALID, Case, Graph, Verification


_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


@dataclass
class EvalResult:
    value: Fraction
    inputs: list[Fraction] = field(default_factory=list)


@dataclass
class ClaimCheck:
    status: str
    reason: str
    result: Fraction | None = None
    inputs: list[Fraction] = field(default_factory=list)


def verify_graph(case: Case, graph: Graph) -> Graph:
    known_numbers = set(numbers_in(case.question))
    node_by_id = {node.id: node for node in graph.nodes}

    for node in graph.nodes:
        check = check_claim(node.claim, known_numbers)
        if check.status == DEBT:
            check = check_grounding(case.question, node.claim, node.sources)
        node.verification = Verification(check.status, check.reason)
        if check.status == VALID and check.result is not None:
            known_numbers.add(check.result)

    for edge in graph.edges:
        missing = [node_id for node_id in edge.premise_node_ids if node_id not in node_by_id]
        if missing:
            edge.verification = Verification(DEBT, "missing premise: " + ",".join(missing))
            continue
        invalid = [
            node_id
            for node_id in edge.premise_node_ids
            if node_by_id[node_id].verification.status != VALID
        ]
        if invalid:
            edge.verification = Verification(DEBT, "unverified premise: " + ",".join(invalid))
            continue
        check = check_claim(edge.claim or edge.conclusion, known_numbers)
        if check.status == VALID and check.result is not None:
            known_numbers.add(check.result)
        edge.verification = Verification(check.status, check.reason)

    graph.coverage_verification = verify_coverage(case, graph)
    return graph


def verify_coverage(case: Case, graph: Graph) -> Verification:
    answer = normalize_text(case.agent_answer)
    valid_answer_edges = [
        edge
        for edge in graph.edges
        if edge.decisive
        and edge.verification.status == VALID
        and answer
        and answer in normalize_text(edge.conclusion)
    ]
    if not valid_answer_edges:
        return Verification(DEBT, "no valid decisive path to answer")
    decisive_nodes = [node for node in graph.nodes if node.decisive]
    decisive_edges = [edge for edge in graph.edges if edge.decisive]
    if all(node.verification.status == VALID for node in decisive_nodes) and all(
        edge.verification.status == VALID for edge in decisive_edges
    ):
        return Verification(VALID, "valid decisive path to answer")
    return Verification(DEBT, "decisive path has unverified items")


def check_claim(claim: str, known_numbers: set[Fraction]) -> ClaimCheck:
    equation = check_equation(claim, known_numbers)
    if equation.status != DEBT:
        return equation
    comparison = check_comparison(claim, known_numbers)
    if comparison.status != DEBT:
        return comparison
    return ClaimCheck(DEBT, "not locally computable")


def check_equation(claim: str, known_numbers: set[Fraction]) -> ClaimCheck:
    parts = [part.strip(" .;$") for part in claim.split("=")]
    if len(parts) < 2:
        return ClaimCheck(DEBT, "no equation")
    for left_text, right_text in zip(parts, parts[1:]):
        left = safe_eval(left_text)
        right = safe_eval(right_text)
        if left is None or right is None:
            continue
        unsupported = unsupported_inputs(left.inputs, known_numbers)
        if unsupported:
            return ClaimCheck(DEBT, "ungrounded input: " + format_fraction(unsupported[0]))
        if left.value == right.value:
            return ClaimCheck(VALID, "computed", result=right.value, inputs=left.inputs)
        return ClaimCheck(REFUTED, "computed opposite")
    return ClaimCheck(DEBT, "no computable equation")


def check_comparison(claim: str, known_numbers: set[Fraction]) -> ClaimCheck:
    for op in ("<=", ">=", "<", ">"):
        if op not in claim:
            continue
        left_text, right_text = claim.split(op, 1)
        left = safe_eval(left_text.strip())
        right = safe_eval(right_text.strip(" .;$"))
        if left is None or right is None:
            continue
        unsupported = unsupported_inputs(left.inputs + right.inputs, known_numbers)
        if unsupported:
            return ClaimCheck(DEBT, "ungrounded input: " + format_fraction(unsupported[0]))
        ok = {
            "<": left.value < right.value,
            ">": left.value > right.value,
            "<=": left.value <= right.value,
            ">=": left.value >= right.value,
        }[op]
        return ClaimCheck(VALID if ok else REFUTED, "comparison")
    return ClaimCheck(DEBT, "no comparison")


def check_grounding(question: str, claim: str, sources: list[str]) -> ClaimCheck:
    if "interrogation" in sources and normalize_text(claim) not in normalize_text(question):
        return ClaimCheck(DEBT, "interrogation premise not grounded")
    claim_numbers = set(numbers_in(claim))
    if claim_numbers and claim_numbers <= set(numbers_in(question)):
        return ClaimCheck(VALID, "numbers grounded in question")
    normalized_claim = normalize_text(claim)
    if normalized_claim and normalized_claim in normalize_text(question):
        return ClaimCheck(VALID, "text grounded in question")
    return ClaimCheck(DEBT, "not grounded")


def safe_eval(text: str) -> EvalResult | None:
    expr = clean_expr(text)
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        return eval_node(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None


def eval_node(node: ast.AST) -> EvalResult:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        value = Fraction(str(node.value))
        return EvalResult(value, [value])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = eval_node(node.operand)
        return EvalResult(-inner.value, [-value for value in inner.inputs])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return eval_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = eval_node(node.left)
        right = eval_node(node.right)
        inputs = left.inputs + right.inputs
        if isinstance(node.op, ast.Add):
            return EvalResult(left.value + right.value, inputs)
        if isinstance(node.op, ast.Sub):
            return EvalResult(left.value - right.value, inputs)
        if isinstance(node.op, ast.Mult):
            return EvalResult(left.value * right.value, inputs)
        if isinstance(node.op, ast.Div):
            return EvalResult(left.value / right.value, inputs)
        if isinstance(node.op, ast.Pow):
            if right.value.denominator != 1 or abs(right.value.numerator) > 12:
                raise ValueError("unsupported exponent")
            return EvalResult(left.value**right.value.numerator, inputs)
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def clean_expr(text: str) -> str:
    text = text.strip()
    text = text.replace("^", "**").replace("$", "").replace(",", "")
    text = text.replace("\\cdot", "*").replace("×", "*").replace("−", "-")
    text = text.strip()
    if re.search(r"[A-Za-z_\\]", text):
        return ""
    return text


def numbers_in(text: str) -> list[Fraction]:
    numeric_text = text.replace(",", "").lower()
    word_text = numeric_text.replace("-", " ")
    numbers = [Fraction(match.group(0)) for match in _NUMBER_RE.finditer(numeric_text)]
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", word_text):
            numbers.append(Fraction(value))
    return numbers


def unsupported_inputs(inputs: list[Fraction], known_numbers: set[Fraction]) -> list[Fraction]:
    return [value for value in inputs if value not in known_numbers]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    as_float = value.numerator / value.denominator
    if math.isfinite(as_float):
        return f"{as_float:g}"
    return f"{value.numerator}/{value.denominator}"
