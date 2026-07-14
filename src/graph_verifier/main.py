from __future__ import annotations

import argparse
import logging
import json
import os
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from graph_verifier.core.aggregate import final_status
from graph_verifier.core.graph import (
    InterrogationState,
    build_graph,
    interrogate,
    prepare_answer_candidates,
    save_interrogation_event,
    select_verification_target,
    target_signature,
)
from graph_verifier.core.models import Case, Graph
from graph_verifier.core.verify import verify_edge_with_llm, verify_graph
from graph_verifier.utils.artifacts import case_name, write_json
from graph_verifier.utils.jsonl import read_jsonl
from graph_verifier.utils.llm import LLMError, complete_json


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases")
    parser.add_argument("--mode", choices=["direct", "one-shot-graph", "interrogation"], default="interrogation")
    parser.add_argument("--max-interrogation-rounds", type=int, default=20)
    parser.add_argument("--concurrency", type=positive_int, default=10)
    args = parser.parse_args(list(argv) if argv is not None else None)
    cases = [Case.from_dict(row) for row in read_jsonl(args.cases)]
    validate_case_names(cases)
    artifact_dir = setup_run_outputs(args.cases, args.mode, args.concurrency)

    for output in process_cases(
        cases,
        args.mode,
        artifact_dir,
        args.max_interrogation_rounds,
        args.concurrency,
    ):
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    logging.info("run end cases=%s", len(cases))
    return 0


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def validate_case_names(cases: Iterable[Case]) -> None:
    seen: dict[str, str] = {}
    for case in cases:
        name = case_name(case.id)
        if name in seen:
            raise ValueError(f"case artifact name collision: {seen[name]!r} and {case.id!r}")
        seen[name] = case.id


def process_cases(
    cases: Iterable[Case],
    mode: str,
    artifact_dir: Path,
    max_interrogation_rounds: int,
    concurrency: int,
) -> Iterable[dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    action = partial(
        process_case,
        mode=mode,
        artifact_dir=artifact_dir,
        max_interrogation_rounds=max_interrogation_rounds,
    )
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="case") as executor:
        yield from executor.map(action, cases)


def process_case(
    case: Case,
    *,
    mode: str,
    artifact_dir: Path,
    max_interrogation_rounds: int,
) -> dict[str, Any]:
    logging.info("case start id=%s", case.id)
    if mode == "direct":
        output = run_stage("direct", run_direct, case, case_id=case.id)
    else:
        graph = run_stage("build_graph", build_graph, case, case_id=case.id)
        save_graph(artifact_dir, case.id, "build_graph", graph)
        if mode == "interrogation":
            graph = run_interrogation_verification(
                case,
                graph,
                artifact_dir,
                max_interrogation_rounds,
                verify_edge_with_llm,
            )
        else:
            graph = run_stage(
                "prepare_candidates",
                prepare_answer_candidates,
                case,
                graph,
                case_id=case.id,
            )
            save_graph(artifact_dir, case.id, "prepare_candidates", graph)
            graph = run_stage(
                "verify_graph",
                verify_graph,
                case,
                graph,
                verify_edge_with_llm,
                case_id=case.id,
            )
            save_graph(artifact_dir, case.id, "verify_graph", graph)
        result = run_stage("aggregate", final_status, graph, case_id=case.id)
        output = compact_output(case, mode, graph, result.status)
    logging.info("case end id=%s status=%s", case.id, output["status"])
    return output


def run_interrogation_verification(
    case: Case,
    graph: Graph,
    artifact_dir: Path,
    max_interrogation_rounds: int,
    edge_checker=verify_edge_with_llm,
) -> Graph:
    state = InterrogationState()
    feedback_cycle = 0
    while True:
        stage_suffix = "" if feedback_cycle == 0 else f"_feedback_{feedback_cycle}"
        graph = run_stage(
            "prepare_candidates" + stage_suffix,
            prepare_answer_candidates,
            case,
            graph,
            case_id=case.id,
        )
        save_graph(artifact_dir, case.id, "prepare_candidates", graph)
        graph = run_stage(
            "verify_graph" + stage_suffix,
            verify_graph,
            case,
            graph,
            edge_checker,
            case_id=case.id,
        )
        save_graph(artifact_dir, case.id, "verify_graph", graph)

        result = final_status(graph)
        if result.status in {"verified_reliable", "answer_refuted"} or graph.tool_debt:
            break
        target = select_verification_target(graph, state.handled)
        if target is None:
            break
        if state.rounds_used >= max_interrogation_rounds:
            reason = f"verification feedback reached max rounds: {max_interrogation_rounds}"
            graph.tool_debt.append(reason)
            save_interrogation_event(
                artifact_dir,
                case.id,
                {
                    "round": state.selection_checks + 1,
                    "event": "verification_feedback_max_rounds",
                    "max_rounds": max_interrogation_rounds,
                    "selected": target,
                },
            )
            save_graph(artifact_dir, case.id, "verify_graph", graph)
            break

        target_type = str(target["target_type"])
        target_id = str(target["target_id"])
        before = target_signature(graph, target_type, target_id)
        feedback_cycle += 1
        graph = run_stage(
            f"interrogate_feedback_{feedback_cycle}",
            interrogate,
            case,
            graph,
            artifact_dir,
            max_interrogation_rounds,
            state,
            target,
            case_id=case.id,
        )
        save_graph(artifact_dir, case.id, "interrogate", graph)
        after = target_signature(graph, target_type, target_id)
        if graph.tool_debt or after == before:
            break
    return graph


def run_stage(name: str, action, *args, case_id: str):
    start = time.perf_counter()
    logging.info("stage start case=%s name=%s", case_id, name)
    try:
        result = action(*args)
    except Exception:
        logging.exception(
            "stage error case=%s name=%s elapsed=%.2fs",
            case_id,
            name,
            time.perf_counter() - start,
        )
        raise
    logging.info(
        "stage end case=%s name=%s elapsed=%.2fs",
        case_id,
        name,
        time.perf_counter() - start,
    )
    return result


def setup_run_outputs(cases: str, mode: str, concurrency: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_dir = Path("runs") / f"graph-verifier-{stamp}-{os.getpid()}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"graph-verifier-{stamp}-{os.getpid()}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        force=True,
    )
    logging.info(
        "run start cases=%s mode=%s concurrency=%s artifacts=%s",
        cases,
        mode,
        concurrency,
        artifact_dir,
    )
    print(f"log: {log_path}", file=sys.stderr)
    print(f"artifacts: {artifact_dir}", file=sys.stderr)
    return artifact_dir


def save_graph(artifact_dir: Path, case_id: str, stage: str, graph: Graph) -> None:
    path = artifact_dir / f"{case_name(case_id)}.graph.{stage}.json"
    write_json(path, graph.to_dict())


def run_direct(case: Case) -> dict[str, Any]:
    try:
        data = complete_json(
            "direct.md",
            {
                "question": case.question,
                "agent_answer": case.agent_answer,
                "agent_reasoning": case.agent_reasoning,
            },
        )
        status = str(data.get("status", "tool_error"))
    except LLMError as exc:
        status = "tool_error"
        data = {"reason": str(exc)}
    return {
        "id": case.id,
        "mode": "direct",
        "status": status,
        "reason": str(data.get("reason", ""))[:120],
    }


def compact_output(case: Case, mode: str, graph: Graph, status: str) -> dict[str, Any]:
    decisive_nodes = [node for node in graph.nodes if node.decisive]
    decisive_edges = [edge for edge in graph.edges if edge.decisive]
    items = [*decisive_nodes, *decisive_edges]
    valid = sum(1 for item in items if item.verification.status == "valid")
    debt = sum(1 for item in items if item.verification.status == "debt")
    refuted = sum(1 for item in items if item.verification.status == "refuted")
    if graph.coverage_decisive:
        coverage_status = graph.coverage_verification.status
        if coverage_status == "valid":
            valid += 1
        elif coverage_status == "refuted":
            refuted += 1
        else:
            debt += 1
    return {
        "id": case.id,
        "mode": mode,
        "status": status,
        "decisive": {
            "nodes": len(decisive_nodes),
            "edges": len(decisive_edges),
            "coverage": graph.coverage_decisive,
        },
        "valid": valid,
        "debt": debt,
        "refuted": refuted,
        "tool_errors": len(graph.tool_debt),
    }


if __name__ == "__main__":
    raise SystemExit(main())
