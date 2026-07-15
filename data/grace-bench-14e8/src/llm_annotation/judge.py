
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

from src.utils.trace_loader import TraceRecord
from src.llm_annotation.prompts import (
    ANNOTATION_SYSTEM,
    format_all_steps_prompt,
    parse_all_steps_response,
    load_grounding_taxonomy,
    load_inference_taxonomy,
)

def _cache_path(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / f"{dataset_name}.jsonl"

def _cache_key(trace_id: str, rep_index: int, model: str, judge_id: str) -> str:
    return f"{trace_id}|{rep_index}|{model}|{judge_id}"

def load_cached_keys(output_dir: Path, dataset_name: str) -> set[str]:
    cache_file = _cache_path(output_dir, dataset_name)
    cached: set[str] = set()
    if not cache_file.exists():
        return cached
    with open(cache_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = _cache_key(
                    record["trace_id"], record["rep_index"],
                    record["model"], record["judge_id"],
                )
                cached.add(key)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARN] Skipping malformed cache line {line_num}: {e}")
    return cached

def append_to_cache(output_dir: Path, dataset_name: str, result: dict) -> None:
    cache_file = _cache_path(output_dir, dataset_name)
    with open(cache_file, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

async def annotate_one_trace(
    trace: TraceRecord,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    judge_id: str,
    grounding_taxonomy: dict,
    inference_taxonomy: dict | None,
    output_dir: Path,
    lock: asyncio.Lock,
    pbar: tqdm,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    max_tokens: int = 8192,
    enable_thinking: bool = False,
) -> dict | None:
    async with semaphore:
        prompt = format_all_steps_prompt(
            trace=trace,
            grounding_taxonomy=grounding_taxonomy,
            inference_taxonomy=inference_taxonomy,
        )

        messages = [
            {"role": "system", "content": ANNOTATION_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        extra_body = {
            "top_k": top_k,
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            },
        }

        try:
            response = await client.chat.completions.create(
                model=judge_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra_body,
            )

            raw_response = (
                response.choices[0].message.content.strip()
                if response.choices[0].message.content
                else ""
            )

        except Exception as e:
            print(
                f"[ERROR] Annotation failed for "
                f"{trace.trace_id} ({judge_id}): {e}"
            )
            return None

    step_annotations = parse_all_steps_response(
        raw_response,
        expected_steps=trace.num_steps,
        dataset=trace.dataset,
    )

    for ann in step_annotations:
        sid = ann["step_id"]
        if sid <= len(trace.steps):
            ann["step_text"] = trace.steps[sid - 1].text

    result = {
        "trace_id": trace.trace_id,
        "rep_index": trace.rep_index,
        "dataset": trace.dataset,
        "model": trace.model,
        "is_correct": trace.is_correct,
        "judge_model": judge_model,
        "judge_id": judge_id,
        "num_steps": trace.num_steps,
        "step_annotations": step_annotations,
        "raw_response": raw_response,
    }

    n_faithful = sum(1 for a in step_annotations if a["faithfulness"] == "faithful")
    n_unfaithful = sum(1 for a in step_annotations if a["faithfulness"] == "unfaithful")
    n_parse_error = sum(1 for a in step_annotations if a.get("parse_error", False))

    async with lock:
        append_to_cache(output_dir, trace.dataset, result)
        pbar.update(1)

        pbar.faithful += n_faithful
        pbar.unfaithful += n_unfaithful
        pbar.parse_fail += n_parse_error

        total = pbar.faithful + pbar.unfaithful + pbar.parse_fail
        uf_rate = (pbar.unfaithful / total * 100) if total else 0
        pbar.set_postfix(
            uf=f"{uf_rate:.1f}%",
            F=pbar.faithful,
            U=pbar.unfaithful,
            pfail=pbar.parse_fail,
            refresh=True,
        )

    return result

async def annotate_traces(
    traces: list[TraceRecord],
    judge_model: str,
    judge_id: str,
    taxonomy: dict,
    vllm_base_urls: list[str],
    output_dir: Path,
    api_key: str = "unused",
    max_concurrent: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    max_tokens: int = 8192,
    enable_thinking: bool = False,
    inference_taxonomy: dict | None = None,
) -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)

    if inference_taxonomy is None:
        try:
            inference_taxonomy = load_inference_taxonomy()
        except FileNotFoundError:
            print("[WARN] GRACE-Inference taxonomy not found — inference track prompts will fail")
            inference_taxonomy = None

    clients = [
        AsyncOpenAI(base_url=url, api_key=api_key)
        for url in vllm_base_urls
    ]
    print(f"vLLM servers ({len(clients)}): {vllm_base_urls}")

    cached_keys_by_dataset: dict[str, set[str]] = {}
    datasets_in_traces = set(t.dataset for t in traces)
    total_cached = 0
    for ds in datasets_in_traces:
        cached = load_cached_keys(output_dir, ds)
        cached_keys_by_dataset[ds] = cached
        total_cached += len(cached)

    if total_cached:
        print(f"Cache: {total_cached:,} trace annotations already done — will skip")

    pending: list[TraceRecord] = []
    total_traces = len(traces)
    total_steps = 0
    for trace in traces:
        total_steps += trace.num_steps
        key = _cache_key(
            trace.trace_id, trace.rep_index,
            trace.model, judge_id,
        )
        if key in cached_keys_by_dataset.get(trace.dataset, set()):
            continue
        pending.append(trace)

    print(f"Total traces: {total_traces:,} ({total_steps:,} steps)")
    print(f"Pending: {len(pending):,} / {total_traces:,} traces to annotate")
    print(f"API calls needed: {len(pending):,} (1 per trace, all-steps-at-once)")

    if not pending:
        print("Nothing to do — all traces are cached.")
        return []

    lock = asyncio.Lock()
    t_start = time.time()
    pbar = tqdm(total=len(pending), desc=f"Annotating [{judge_id}]", unit="trace")
    pbar.faithful = 0
    pbar.unfaithful = 0
    pbar.parse_fail = 0

    per_server = max(1, max_concurrent // len(clients))
    semaphores = [asyncio.Semaphore(per_server) for _ in clients]
    print(f"Max concurrent per server: {per_server} × {len(clients)} = {per_server * len(clients)} total")

    tasks = []
    for seq, trace in enumerate(pending):
        server_idx = seq % len(clients)
        tasks.append(
            annotate_one_trace(
                trace,
                clients[server_idx],
                semaphores[server_idx],
                judge_model,
                judge_id,
                taxonomy,
                inference_taxonomy,
                output_dir,
                lock,
                pbar,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
        )

    results = await asyncio.gather(*tasks)
    pbar.close()
    results = [r for r in results if r is not None]

    total_time = time.time() - t_start
    cat_counts: dict[str, int] = {}
    parse_fails = 0
    n_faithful = 0
    n_unfaithful = 0
    for r in results:
        for ann in r.get("step_annotations", []):
            if ann.get("parse_error"):
                parse_fails += 1
            faith = ann.get("faithfulness")
            if faith == "faithful":
                n_faithful += 1
            elif faith == "unfaithful":
                n_unfaithful += 1
                cat = ann.get("error_category")
                if cat:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1

    total_eval = n_faithful + n_unfaithful + parse_fails
    uf_rate = (n_unfaithful / total_eval * 100) if total_eval else 0

    print(f"\n{'━' * 60}")
    print(f"  ANNOTATION SUMMARY — {judge_id}")
    print(f"{'━' * 60}")
    print(f"  Judge model      : {judge_model}")
    print(f"  Judge ID         : {judge_id}")
    print(f"  Traces annotated : {len(results):,}")
    print(f"  Previously cached: {total_cached:,}")
    print(f"  Steps — faithful : {n_faithful:,}")
    print(f"  Steps — unfaithful: {n_unfaithful:,} ({uf_rate:.1f}%)")
    print(f"  Steps — parse fail: {parse_fails:,}")
    if n_unfaithful > 0:
        print(f"  Error categories :")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat:30s}: {count:,}")
    print(f"  API calls        : {len(results):,}")
    print(f"  Time             : {total_time:.1f}s")
    if results:
        print(f"  Throughput       : {len(results) / max(total_time, 0.1):.1f} traces/s")
    print(f"  Output           : {output_dir}")
    print(f"{'━' * 60}")

    return results
