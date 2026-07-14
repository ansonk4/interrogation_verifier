from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

from graph_verifier.core.models import (
    DEBT,
    REFUTED,
    QUERY_TARGET,
    VALID,
    Case,
    Edge,
    Graph,
    Node,
    Verification,
    answer_claim_matches,
    graph_id_error,
)
from graph_verifier.utils.llm import LLMError, complete_json


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


def verify_graph(
    case: Case,
    graph: Graph,
    edge_checker: Callable[[list[Node], Edge, Node], ClaimCheck] | None = None,
) -> Graph:
    id_error = graph_id_error(graph)
    if id_error:
        graph.tool_debt.append(id_error)
        graph.coverage_verification = Verification(DEBT, id_error)
        return graph
    question_numbers = set(numbers_in(case.question))
    node_by_id = {node.id: node for node in graph.nodes}
    incoming_edges: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        incoming_edges.setdefault(edge.target_node_id, []).append(edge)
    node_numbers: dict[str, set[Fraction]] = {}
    edge_checks: dict[str, ClaimCheck] = {}

    for node in graph.nodes:
        if node.kind == "answer":
            check = ClaimCheck(DEBT, "answer requires verified support")
        else:
            check = check_claim(node.claim, question_numbers)
            if check.status == DEBT:
                check = check_closed_calculation(node.claim)
            if check.status == DEBT:
                check = check_grounding(case.question, node.claim, node.sources)
        node.verification = Verification(check.status, check.reason)
        node_numbers[node.id] = exposed_numbers(check)

    for edge in graph.edges:
        edge.verification = Verification(DEBT, "not verified")

    for _ in range(max(1, len(graph.edges))):
        changed = False
        for edge in graph.edges:
            verification, check = verify_edge(
                edge,
                node_by_id,
                node_numbers,
                edge_checker,
                edge_checks,
                incoming_edges,
            )
            changed = set_verification(edge, verification) or changed
            if edge.target_node_id not in node_by_id:
                continue
            target = node_by_id[edge.target_node_id]
            if edge.decisive and verification.status == VALID and check and target.verification.status == DEBT:
                changed = set_verification(target, Verification(VALID, verification.reason)) or changed
                numbers = exposed_numbers(check)
                if node_numbers.get(target.id) != numbers:
                    node_numbers[target.id] = numbers
                    changed = True
        if not changed:
            break

    graph.coverage_verification = verify_coverage(case, graph)
    for edge in graph.edges:
        reason = edge.verification.reason
        if (
            edge.decisive
            and reason.startswith("edge verifier failed:")
            and reason not in graph.tool_debt
        ):
            graph.tool_debt.append(reason)
    return graph


def set_verification(item: Node | Edge, verification: Verification) -> bool:
    if item.verification == verification:
        return False
    item.verification = verification
    return True


def validate_edge_structure(edge: Edge, node_by_id: dict[str, Node]) -> Verification | None:
    if not edge.premise_node_ids:
        return Verification(DEBT, "empty premise_node_ids")
    missing = [node_id for node_id in edge.premise_node_ids if node_id not in node_by_id]
    if missing:
        return Verification(DEBT, "missing premise: " + ",".join(missing))
    if not edge.target_node_id or edge.target_node_id not in node_by_id:
        return Verification(DEBT, "missing target: " + edge.target_node_id)
    return None


def verify_edge(
    edge: Edge,
    node_by_id: dict[str, Node],
    node_numbers: dict[str, set[Fraction]],
    edge_checker: Callable[[list[Node], Edge, Node], ClaimCheck] | None = None,
    edge_checks: dict[str, ClaimCheck] | None = None,
    incoming_edges: dict[str, list[Edge]] | None = None,
) -> tuple[Verification, ClaimCheck | None]:
    structure = validate_edge_structure(edge, node_by_id)
    if structure:
        return structure, None
    invalid = [
        node_id
        for node_id in edge.premise_node_ids
        if node_by_id[node_id].verification.status != VALID
    ]
    if invalid:
        return Verification(DEBT, "unverified premise: " + ",".join(invalid)), None
    non_decisive = [
        node_id
        for node_id in edge.premise_node_ids
        if edge.decisive and not node_by_id[node_id].decisive
    ]
    if non_decisive:
        return Verification(DEBT, "non-decisive premise: " + ",".join(non_decisive)), None
    known_numbers = set().union(*(node_numbers[node_id] for node_id in edge.premise_node_ids))
    target = node_by_id[edge.target_node_id]
    premise_supported = bool(
        len(edge.premise_node_ids) == 1
        and any(
            incoming.verification.status == VALID
            for incoming in (incoming_edges or {}).get(edge.premise_node_ids[0], [])
        )
    )
    answer_check = check_answer_edge(
        [node_by_id[node_id] for node_id in edge.premise_node_ids], target, premise_supported
    )
    if answer_check:
        return Verification(VALID, answer_check.reason), answer_check
    calculation_check = check_edge_calculation(edge, target, known_numbers)
    if calculation_check:
        return edge_verification(edge.premise_node_ids, calculation_check, node_numbers), calculation_check
    target_check = check_claim(target.claim, known_numbers)
    if target_check.status == REFUTED:
        return Verification(REFUTED, target_check.reason), target_check
    if target.verification.status == REFUTED:
        return Verification(DEBT, "target is independently refuted"), None
    normalized_edge = normalize_text(edge.claim)
    normalized_target = normalize_text(target.claim)
    same_claim = normalized_edge == normalized_target or normalized_edge.endswith(normalized_target)
    if (target_check.status == DEBT or not same_claim) and edge_checker and edge.decisive:
        cache = edge_checks if edge_checks is not None else {}
        if edge.id not in cache:
            try:
                result = edge_checker(
                    [node_by_id[node_id] for node_id in edge.premise_node_ids],
                    edge,
                    target,
                )
                if not isinstance(result, ClaimCheck):
                    raise TypeError("edge checker must return ClaimCheck")
                cache[edge.id] = result
            except (LLMError, KeyError, TypeError, ValueError) as exc:
                cache[edge.id] = ClaimCheck(DEBT, f"edge verifier failed: {exc}")
        fallback = cache[edge.id]
        return Verification(fallback.status, fallback.reason), fallback
    if target_check.status == VALID and not same_claim:
        return Verification(DEBT, "edge claim does not establish target"), None
    verification = edge_verification(edge.premise_node_ids, target_check, node_numbers)
    return verification, target_check


def verify_edge_with_llm(premises: list[Node], edge: Edge, target: Node) -> ClaimCheck:
    data = complete_json(
        "verify_edge.md",
        {
            "premises": [
                {"id": node.id, "claim": node.claim, "kind": node.kind} for node in premises
            ],
            "edge_claim": edge.claim,
            "target_claim": target.claim,
        },
    )
    if not isinstance(data, dict):
        return ClaimCheck(DEBT, "edge verifier failed: response is not an object")
    status = str(data.get("status", DEBT)).lower()
    if status not in {VALID, DEBT, REFUTED}:
        return ClaimCheck(DEBT, "edge verifier failed: invalid status")
    used = data.get("used_premise_node_ids", [])
    if not isinstance(used, list):
        return ClaimCheck(DEBT, "edge verifier failed: invalid premise ids")
    used_ids = {str(node_id) for node_id in used}
    premise_ids = {node.id for node in premises}
    if used_ids - premise_ids or (status != DEBT and used_ids != premise_ids):
        return ClaimCheck(DEBT, "edge verifier failed: invalid premise use")
    return ClaimCheck(status, str(data.get("reason", "edge verification"))[:160])


def edge_verification(
    premise_node_ids: list[str], check: ClaimCheck, node_numbers: dict[str, set[Fraction]]
) -> Verification:
    if check.status != VALID:
        return Verification(check.status, check.reason)
    unused = [
        node_id
        for node_id in premise_node_ids
        if node_numbers[node_id] and not node_numbers[node_id].intersection(check.inputs)
    ]
    if unused:
        return Verification(DEBT, "unused premise: " + ",".join(unused))
    return Verification(VALID, check.reason)


def exposed_numbers(check: ClaimCheck) -> set[Fraction]:
    if check.status != VALID:
        return set()
    numbers = set(check.inputs)
    if check.result is not None:
        numbers.add(check.result)
    return numbers


def verify_coverage(case: Case, graph: Graph) -> Verification:
    answer = normalize_text(case.agent_answer)
    node_by_id = {node.id: node for node in graph.nodes}
    query_target_ids = {node.id for node in graph.nodes if node.kind == QUERY_TARGET}
    if not query_target_ids:
        return Verification(DEBT, "missing query target")
    valid_decisive_edges = [
        edge
        for edge in graph.edges
        if edge.decisive and edge.verification.status == VALID and edge.target_node_id in node_by_id
    ]
    answer_node_ids = [
        edge.target_node_id
        for edge in valid_decisive_edges
        if answer
        and node_by_id[edge.target_node_id].kind == "answer"
        and node_by_id[edge.target_node_id].decisive
        and node_by_id[edge.target_node_id].verification.status == VALID
        and answer_claim_matches(node_by_id[edge.target_node_id].claim, answer)
    ]
    if not answer_node_ids:
        return Verification(DEBT, "no valid decisive path to answer")
    decisive_nodes = [node for node in graph.nodes if node.decisive]
    decisive_edges = [edge for edge in graph.edges if edge.decisive]
    if not all(node.verification.status == VALID for node in decisive_nodes) or not all(
        edge.verification.status == VALID for edge in decisive_edges
    ):
        return Verification(DEBT, "decisive path has unverified items")

    incoming: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        if edge.decisive and edge.verification.status == VALID:
            incoming.setdefault(edge.target_node_id, []).append(edge)

    path_nodes: set[str] = set()
    path_edges: set[str] = set()
    stack = list(answer_node_ids)
    while stack:
        node_id = stack.pop()
        if node_id in path_nodes:
            continue
        path_nodes.add(node_id)
        for edge in incoming.get(node_id, []):
            path_edges.add(edge.id)
            stack.extend(edge.premise_node_ids)

    missing_nodes = [node.id for node in decisive_nodes if node.id not in path_nodes]
    if missing_nodes:
        return Verification(DEBT, "decisive node not on answer path: " + ",".join(missing_nodes))
    missing_edges = [edge.id for edge in decisive_edges if edge.id not in path_edges]
    if missing_edges:
        return Verification(DEBT, "decisive edge not on answer path: " + ",".join(missing_edges))
    if not query_target_ids.intersection(path_nodes):
        return Verification(DEBT, "query target not on valid answer path")
    return Verification(VALID, "valid decisive path to answer")


def check_claim(claim: str, known_numbers: set[Fraction]) -> ClaimCheck:
    equation = check_equation(claim, known_numbers)
    if equation.status != DEBT:
        return equation
    comparison = check_comparison(claim, known_numbers)
    if comparison.status != DEBT:
        return comparison
    if equation.reason.startswith("ungrounded input"):
        return equation
    if comparison.reason.startswith("ungrounded input"):
        return comparison
    return ClaimCheck(DEBT, "not locally computable")


def check_closed_calculation(claim: str) -> ClaimCheck:
    parts = re.split(r"\s+and\s+", claim, flags=re.IGNORECASE)
    checks = [check_claim(part, set(numbers_in(part))) for part in parts]
    if len(checks) > 1:
        if all(check.status == VALID for check in checks):
            return ClaimCheck(
                VALID,
                "computed",
                result=checks[-1].result,
                inputs=[value for check in checks for value in check.inputs],
            )
        refuted = next((check for check in checks if check.status == REFUTED), None)
        return refuted or ClaimCheck(DEBT, "not locally computable")
    return checks[0]


def check_answer_edge(
    premises: list[Node], target: Node, premise_supported: bool
) -> ClaimCheck | None:
    if len(premises) != 1 or target.kind != "answer" or not premise_supported:
        return None
    answer_text = normalize_text(target.claim).removeprefix("answer ")
    answer = safe_eval(answer_text)
    if answer is None:
        return None
    value = terminal_value(premises[0].claim)
    if value is not None:
        return (
            ClaimCheck(VALID, "verified result is the stated answer", result=answer.value)
            if value.value == answer.value
            else None
        )
    return None


def check_edge_calculation(
    edge: Edge, target: Node, known_numbers: set[Fraction]
) -> ClaimCheck | None:
    if "=" not in edge.claim or "=" not in target.claim:
        return None
    target_label = normalize_text(target.claim.split("=", 1)[0])
    edge_label = normalize_text(edge.claim.split("=", 1)[0])
    if not re.search(r"[a-z]", target_label) or target_label not in edge_label:
        return None
    target_value = terminal_value(target.claim)
    edge_check = check_claim(edge.claim, known_numbers)
    if target_value is None or edge_check.status != VALID or edge_check.result != target_value.value:
        return None
    return ClaimCheck(
        VALID,
        "edge calculation establishes labeled target",
        result=target_value.value,
        inputs=edge_check.inputs,
    )


def terminal_value(claim: str) -> EvalResult | None:
    for part in reversed(claim.split("=")):
        value = safe_eval(part.strip(" .;"))
        if value is not None:
            return value
    return None


def check_equation(claim: str, known_numbers: set[Fraction]) -> ClaimCheck:
    parts = [part.strip(" .;$") for part in claim.split("=")]
    if len(parts) < 2:
        return ClaimCheck(DEBT, "no equation")
    first_unsupported: Fraction | None = None
    for left_text, right_text in zip(parts, parts[1:]):
        left = safe_eval(left_text)
        right = safe_eval(right_text)
        if left is None or right is None:
            continue
        unsupported = unsupported_inputs(left.inputs, known_numbers)
        if unsupported:
            first_unsupported = first_unsupported or unsupported[0]
            continue
        if left.value == right.value:
            return ClaimCheck(VALID, "computed", result=right.value, inputs=left.inputs)
        return ClaimCheck(REFUTED, "computed opposite")
    if first_unsupported is not None:
        return ClaimCheck(DEBT, "ungrounded input: " + format_fraction(first_unsupported))
    return ClaimCheck(DEBT, "no computable equation")


def check_comparison(claim: str, known_numbers: set[Fraction]) -> ClaimCheck:
    expr = clean_expr(claim.strip(" .;$"))
    if expr:
        sqrt_chain = re.fullmatch(r"(-?\d+)\s*<\s*sqrt\((\d+)\)\s*<\s*(-?\d+)", expr)
        if sqrt_chain:
            lower, value, upper = (Fraction(item) for item in sqrt_chain.groups())
            inputs = [lower, value, upper]
            unsupported = unsupported_inputs(inputs, known_numbers)
            if unsupported:
                return ClaimCheck(DEBT, "ungrounded input: " + format_fraction(unsupported[0]))
            valid = lower >= 0 and lower * lower < value < upper * upper
            return ClaimCheck(VALID if valid else REFUTED, "comparison", inputs=inputs)
        try:
            tree = ast.parse(expr, mode="eval")
            if isinstance(tree.body, ast.Compare):
                operands = [eval_node(tree.body.left), *(eval_node(item) for item in tree.body.comparators)]
                inputs = [value for operand in operands for value in operand.inputs]
                unsupported = unsupported_inputs(inputs, known_numbers)
                if unsupported:
                    return ClaimCheck(DEBT, "ungrounded input: " + format_fraction(unsupported[0]))
                comparisons = {
                    ast.Lt: lambda left, right: left < right,
                    ast.Gt: lambda left, right: left > right,
                    ast.LtE: lambda left, right: left <= right,
                    ast.GtE: lambda left, right: left >= right,
                }
                ok = all(
                    comparisons[type(op)](left.value, right.value)
                    for op, left, right in zip(tree.body.ops, operands, operands[1:])
                )
                return ClaimCheck(VALID if ok else REFUTED, "comparison", inputs=inputs)
        except (KeyError, SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            pass
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
        return ClaimCheck(VALID if ok else REFUTED, "comparison", inputs=left.inputs + right.inputs)
    return ClaimCheck(DEBT, "no comparison")


def check_grounding(question: str, claim: str, sources: list[str]) -> ClaimCheck:
    normalized_claim = normalize_text(claim)
    if normalized_claim and normalized_claim in normalize_text(question):
        return ClaimCheck(VALID, "text grounded in question", inputs=numbers_in(claim))
    if "interrogation" in sources:
        return ClaimCheck(DEBT, "interrogation premise not grounded")
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
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1:
        inner = eval_node(node.args[0])
        if node.func.id == "factorial" and inner.value.denominator == 1:
            value = inner.value.numerator
            if 0 <= value <= 100:
                return EvalResult(Fraction(math.factorial(value)), inner.inputs)
        if node.func.id == "sqrt" and inner.value >= 0:
            numerator = math.isqrt(inner.value.numerator)
            denominator = math.isqrt(inner.value.denominator)
            if numerator * numerator == inner.value.numerator and denominator * denominator == inner.value.denominator:
                return EvalResult(Fraction(numerator, denominator), inner.inputs)
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def clean_expr(text: str) -> str:
    text = text.strip()
    text = text.replace("²", "**2").replace("³", "**3")
    text = re.sub(r"√\s*(\d+)", r"sqrt(\1)", text)
    fraction = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while fraction.search(text):
        text = fraction.sub(r"((\1)/(\2))", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"(\d+|\([^()]+\))!", r"factorial(\1)", text)
    text = text.replace("^", "**").replace("$", "").replace(",", "")
    text = text.replace("\\cdot", "*").replace("×", "*").replace("−", "-")
    text = re.sub(r"\s*(?:degrees?|°)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\)\s*\(", ")*(", text)
    text = text.strip()
    names_removed = re.sub(r"\b(?:factorial|sqrt)\b", "", text)
    if re.search(r"[A-Za-z_\\]", names_removed):
        return ""
    return text


def numbers_in(text: str) -> list[Fraction]:
    numeric_text = text.replace(",", "").replace("²", " 2").replace("³", " 3").lower()
    word_text = numeric_text.replace("-", " ")
    numbers = [Fraction(match.group(0)) for match in _NUMBER_RE.finditer(numeric_text)]
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", word_text):
            numbers.append(Fraction(value))
    return numbers


def unsupported_inputs(inputs: list[Fraction], known_numbers: set[Fraction]) -> list[Fraction]:
    return [value for value in inputs if value not in known_numbers]


def normalize_text(text: str) -> str:
    text = text.lower().replace("$", "").replace("\\(", "").replace("\\)", "")
    text = re.sub(r"\\(?:left|right|displaystyle)\b", "", text)
    text = text.replace("²", "^2").replace("³", "^3")
    text = re.sub(r"√\s*(\d+)", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"[.,!?;:]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    as_float = value.numerator / value.denominator
    if math.isfinite(as_float):
        return f"{as_float:g}"
    return f"{value.numerator}/{value.denominator}"
