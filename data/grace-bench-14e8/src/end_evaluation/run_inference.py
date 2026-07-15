

from __future__ import annotations

import asyncio
import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

from src.end_evaluation.prompts import (
    build_all_steps_prompt,
    build_trace_prediction,
    parse_all_steps_response,
)




_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = _PROJECT_ROOT / "resources" / "results" / "end_evaluation" / "inference"




def load_traces(input_dirs: list[str]) -> list[dict]:

    traces = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"  [WARN] {input_path} does not exist, skipping")
            continue
        for jsonl_file in sorted(input_path.glob("*.jsonl")):
            print(f"  Loading {jsonl_file}...")
            count = 0
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        traces.append(json.loads(line))
                        count += 1
            print(f"    → {count:,} traces")
    return traces




def _cache_path(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / f"{dataset_name}.jsonl"


def load_cached_trace_keys(output_dir: Path, dataset_name: str) -> set[tuple[str, str]]:

    path = _cache_path(output_dir, dataset_name)
    cached = set()
    if not path.exists():
        return cached
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                gen_model = record.get("generator_model") or record.get("model", "")
                cached.add((record["trace_id"], gen_model))
            except (json.JSONDecodeError, KeyError):
                pass
    return cached


def append_to_cache(output_dir: Path, dataset_name: str, record: dict) -> None:

    path = _cache_path(output_dir, dataset_name)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")




async def evaluate_trace(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    trace: dict,
    args,
) -> dict | None:

    steps = trace.get("steps", [])
    if not steps:
        return None

    prompt = build_all_steps_prompt(trace)
    messages = [{"role": "user", "content": prompt}]

    extra_body = {}
    _is_mistral = "mistral" in args.model_name.lower()
    if not _is_mistral:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": args.enable_thinking,
        }
    if args.top_k > 0:
        extra_body["top_k"] = args.top_k

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                extra_body=extra_body if extra_body else None,
            )

            raw_text = (
                response.choices[0].message.content.strip()
                if response.choices[0].message.content
                else ""
            )
        except Exception as e:
            print(f"[ERROR] Trace {trace['trace_id']}: {e}")
            raw_text = ""

    step_predictions = parse_all_steps_response(raw_text, expected_steps=len(steps))

    if step_predictions:
        step_predictions[0]["raw_response_full"] = raw_text

    record = build_trace_prediction(trace, step_predictions, args.model_name)
    return record


async def process_traces(
    traces: list[dict],
    clients: list[AsyncOpenAI],
    semaphores: list[asyncio.Semaphore],
    args,
    output_dir: Path,
) -> list[dict]:

    lock = asyncio.Lock()
    pbar = tqdm(total=len(traces), desc=f"Evaluating traces ({args.model_name})", unit="trace")
    seq_counter = [0]
    results = []

    async def _process_one(trace: dict) -> dict | None:
        idx = seq_counter[0] % len(clients)
        seq_counter[0] += 1

        record = await evaluate_trace(
            clients[idx], semaphores[idx], trace, args,
        )

        if record is not None:
            async with lock:
                append_to_cache(output_dir, trace["dataset"], record)
                results.append(record)
                pbar.update(1)

        return record

    tasks = [_process_one(trace) for trace in traces]
    await asyncio.gather(*tasks)
    pbar.close()
    return results




def parse_args():
    parser = ArgumentParser(
        description="Run GRACE faithfulness evaluation via vLLM API.",
    )

    parser.add_argument(
        "--input", type=str, nargs="+", required=True,
        help="Path(s) to assembled data directories containing JSONL files.",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="*", default=None,
        help="Filter to specific datasets (e.g., logiqa reclor). Default: all.",
    )

    parser.add_argument(
        "--model-name", type=str, required=True,
        help="Served model name (must match --served-model-name in vLLM).",
    )
    parser.add_argument(
        "--vllm-base-url", type=str, nargs="+",
        default=["http://localhost:8000/v1"],
        help="One or more vLLM OpenAI-compatible base URLs.",
    )
    parser.add_argument(
        "--api-key", type=str, default="unused",
        help="API key for the OpenAI-compatible server.",
    )

    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for deterministic).")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling threshold.")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Top-k sampling (default: -1 = disabled).")
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="Max tokens for the generated response.")
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                        help="Enable model thinking/reasoning mode.")

    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory.",
    )

    parser.add_argument("--max-concurrent", type=int, default=64,
                        help="Max concurrent async requests across all servers.")

    return parser.parse_args()




async def run(args):

    print("Loading traces...")
    all_traces = load_traces(args.input)
    print(f"Total traces loaded: {len(all_traces):,}")

    if args.datasets:
        all_traces = [t for t in all_traces if t["dataset"] in args.datasets]
        print(f"Filtered to datasets {args.datasets}: {len(all_traces):,} traces")

    model_dir = args.model_name.lower().replace("/", "_")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _DEFAULT_OUTPUT / model_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")

    clients = [
        AsyncOpenAI(base_url=url, api_key=args.api_key)
        for url in args.vllm_base_url
    ]
    print(f"vLLM servers ({len(clients)}): {args.vllm_base_url}")

    dataset_names = set(t["dataset"] for t in all_traces)
    cached_keys: set[tuple[str, str]] = set()
    for ds in dataset_names:
        cached_keys |= load_cached_trace_keys(output_dir, ds)
    if cached_keys:
        print(f"Cache: {len(cached_keys)} traces already done — will skip")

    pending = [t for t in all_traces if (t["trace_id"], t.get("model", "")) not in cached_keys]
    print(f"Pending: {len(pending):,} / {len(all_traces):,} traces")

    if not pending:
        print("Nothing to do — all traces are cached.")
        return

    per_server = max(1, args.max_concurrent // len(clients))
    semaphores = [asyncio.Semaphore(per_server) for _ in clients]
    print(f"Max concurrent per server: {per_server} × {len(clients)} = {per_server * len(clients)} total")

    t_start = time.time()

    total_steps = sum(len(t.get("steps", [])) for t in pending)
    print(f"Total steps across pending traces: {total_steps:,}")

    results = await process_traces(
        pending, clients, semaphores, args, output_dir,
    )

    total_time = time.time() - t_start
    total_step_preds = sum(len(r["step_predictions"]) for r in results)
    parsed_steps = sum(
        1 for r in results
        for sp in r["step_predictions"]
        if sp.get("pred_faithfulness") is not None
    )

    print(f"\n{'━' * 60}")
    print(f"  SUMMARY — {args.model_name}")
    print(f"{'━' * 60}")
    print(f"  Traces evaluated : {len(results):,}")
    print(f"  Steps evaluated  : {total_step_preds:,}")
    print(f"  Steps parsed     : {parsed_steps:,} / {total_step_preds:,} ({100 * parsed_steps / max(total_step_preds, 1):.1f}%)")
    print(f"  Previously cached: {len(cached_keys):,}")
    print(f"  Time             : {total_time:.1f}s")
    if results:
        print(f"  Throughput       : {len(results) / max(total_time, 0.1):.1f} traces/s")
    print(f"  Output           : {output_dir}")
    print(f"{'━' * 60}")


def main():
    args = parse_args()
    print("=" * 60)
    print("  GRACE Faithfulness Inference")
    print("=" * 60)
    print(json.dumps(vars(args), indent=4, ensure_ascii=False))
    print()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
