from __future__ import annotations

import argparse
import logging
import json
import os
import sys
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from graph_verifier.core.aggregate import final_status
from graph_verifier.core.graph import build_graph, interrogate, mark_decisive
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
    args = parser.parse_args(list(argv) if argv is not None else None)
    artifact_dir = setup_run_outputs(args.cases, args.mode)

    case_count = 0
    for row in read_jsonl(args.cases):
        case = Case.from_dict(row)
        case_count += 1
        logging.info("case start id=%s", case.id)
        if args.mode == "direct":
            output = run_stage("direct", run_direct, case)
        else:
            graph = run_stage("build_graph", build_graph, case)
            save_graph(artifact_dir, case.id, "build_graph", graph)
            if args.mode == "interrogation":
                graph = run_stage(
                    "interrogate",
                    interrogate,
                    case,
                    graph,
                    artifact_dir,
                    args.max_interrogation_rounds,
                )
                save_graph(artifact_dir, case.id, "interrogate", graph)
            graph = run_stage("mark_decisive", mark_decisive, case, graph)
            save_graph(artifact_dir, case.id, "mark_decisive", graph)
            graph = run_stage("verify_graph", verify_graph, case, graph, verify_edge_with_llm)
            save_graph(artifact_dir, case.id, "verify_graph", graph)
            result = run_stage("aggregate", final_status, graph)
            output = compact_output(case, args.mode, graph, result.status)
        logging.info("case end id=%s status=%s", case.id, output["status"])
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    logging.info("run end cases=%s", case_count)
    return 0


def run_stage(name: str, action, *args):
    start = time.perf_counter()
    logging.info("stage start name=%s", name)
    try:
        result = action(*args)
    except Exception:
        logging.exception("stage error name=%s elapsed=%.2fs", name, time.perf_counter() - start)
        raise
    logging.info("stage end name=%s elapsed=%.2fs", name, time.perf_counter() - start)
    return result


def setup_run_outputs(cases: str, mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_dir = Path("runs") / f"graph-verifier-{stamp}-{os.getpid()}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"graph-verifier-{stamp}-{os.getpid()}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logging.info("run start cases=%s mode=%s artifacts=%s", cases, mode, artifact_dir)
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
        status = str(data.get("status", "verification_debt"))
    except LLMError as exc:
        status = "verification_debt"
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
    if graph.tool_debt:
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
    }


if __name__ == "__main__":
    raise SystemExit(main())
