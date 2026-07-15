
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from argparse import ArgumentParser
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

from src.cot_generation.parse_traces import parse_trace
from src.cot_generation.prompts import SYSTEM_MESSAGE, build_cot_prompt
from src.dataset import GraceDataset, ACTIVE_DATASETS

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RESULTS_DIR = _PROJECT_ROOT / "resources" / "results" / "cot_traces"

_OPTION_LABELS = "ABCDEFGHIJ"

_PILOT_SIZES = {
    "musique": 100,
    "reclor": 80,
    "logiqa": 80,
    "wiki2multihop": 100,
}

def _sample_pilot_indices(ds: GraceDataset, seed: int = 42) -> list[int]:
    rng = random.Random(seed)
    target = _PILOT_SIZES.get(ds.name, 100)

    if ds.name == "musique":
        hop3 = [i for i in range(len(ds)) if ds.raw(i).get("id", "").startswith("3")]
        hop4 = [i for i in range(len(ds)) if ds.raw(i).get("id", "").startswith("4")]

        if len(hop3) < 70 or len(hop4) < 30:
            return sorted(rng.sample(range(len(ds)), min(target, len(ds))))

        selected = rng.sample(hop3, 70) + rng.sample(hop4, 30)
        return sorted(selected)

    elif ds.name == "wiki2multihop":
        type_targets = {
            "bridge_comparison": 40,
            "comparison": 20,
            "compositional": 20,
            "inference": 20,
        }
        type_groups: dict[str, list[int]] = {}
        for i in range(len(ds)):
            t = ds.raw(i).get("type", "unknown")
            type_groups.setdefault(t, []).append(i)

        selected = []
        for t, n in type_targets.items():
            pool = type_groups.get(t, [])
            selected.extend(rng.sample(pool, min(n, len(pool))))

        if len(selected) < target:
            remaining = [i for i in range(len(ds)) if i not in set(selected)]
            selected.extend(rng.sample(remaining, min(target - len(selected), len(remaining))))
        return sorted(selected)

    else:
        return sorted(rng.sample(range(len(ds)), min(target, len(ds))))

def _check_correctness(
    predicted: str,
    gold_answer: str,
    answer_type: str,
    gold_answer_index: int | None,
    options: list[str] | None,
    answer_aliases: list[str] | None = None,
) -> bool | None:
    if not predicted or not predicted.strip():
        return None

    pred = predicted.strip()

    if answer_type == "mcq" and gold_answer_index is not None and options:
        label = _OPTION_LABELS[gold_answer_index]
        pred_upper = pred.upper()

        if pred_upper.startswith(f"{label})") or pred_upper.startswith(f"{label}.") or pred_upper.startswith(f"{label} "):
            return True

        if pred_upper.strip() == label:
            return True

        if gold_answer.lower().strip() in pred.lower():
            return True

        for other_label in _OPTION_LABELS[:len(options)]:
            if other_label == label:
                continue
            if pred_upper.startswith(f"{other_label})") or pred_upper.startswith(f"{other_label}.") or pred_upper.startswith(f"{other_label} "):
                return False
            if pred_upper.strip() == other_label:
                return False

        return None

    else:
        pred_norm = pred.lower().strip().rstrip(".")

        candidates = [gold_answer]
        if answer_aliases:
            candidates.extend(answer_aliases)

        for candidate in candidates:
            cand_norm = candidate.lower().strip().rstrip(".")
            if not cand_norm:
                continue
            if pred_norm == cand_norm:
                return True
            if cand_norm in pred_norm:
                return True
            if pred_norm in cand_norm:
                return True

        return False

def _cache_path(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / f"{dataset_name}.jsonl"

def load_cached_keys(output_dir: Path, dataset_name: str) -> set[tuple[str, int]]:
    cache_file = _cache_path(output_dir, dataset_name)
    cached: set[tuple[str, int]] = set()
    if not cache_file.exists():
        return cached
    with open(cache_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                rep = record.get("rep_index", 0)
                cached.add((record["id"], rep))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARN] Skipping malformed cache line {line_num}: {e}")
    return cached

def append_to_cache(output_dir: Path, dataset_name: str, result: dict) -> None:
    cache_file = _cache_path(output_dir, dataset_name)
    with open(cache_file, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

async def generate_one_trace(
    example,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args,
    output_dir: Path,
    lock: asyncio.Lock,
    pbar: tqdm,
    rep_index: int = 0,
) -> dict | None:
    async with semaphore:
        prompt = build_cot_prompt(example)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]

        extra_body = {
            "top_k": args.top_k,
        }
        _is_mistral = "mistral" in args.model_name.lower()
        if not _is_mistral:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": args.enable_thinking,
            }

        try:
            response = await client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                extra_body=extra_body,
            )

            raw_text = (
                response.choices[0].message.content.strip()
                if response.choices[0].message.content
                else ""
            )

            reasoning_text = None
            msg = response.choices[0].message
            if hasattr(msg, "reasoning") and msg.reasoning:
                reasoning_text = msg.reasoning.strip()

        except Exception as e:
            print(f"[ERROR] Generation failed for {example.id}: {e}")
            return None

        parsed = parse_trace(raw_text)

        is_correct = _check_correctness(
            predicted=parsed.final_answer,
            gold_answer=example.gold_answer,
            answer_type=example.answer_type,
            gold_answer_index=example.gold_answer_index,
            options=example.options,
            answer_aliases=example.metadata.get("answer_aliases"),
        )

        result = {
            "id": example.id,
            "rep_index": rep_index,
            "dataset": example.dataset_name,
            "context": example.context,
            "question": example.question,
            "options": example.options,
            "gold_answer": example.gold_answer,
            "model": args.model_name,
            "temperature": args.temperature,
            "trace": parsed.to_dict(),
            "reasoning": reasoning_text,
            "is_correct": is_correct,
            "annotations": None,
        }

        async with lock:
            append_to_cache(output_dir, example.dataset_name, result)
            pbar.update(1)
            if is_correct is True:
                pbar.correct += 1
            elif is_correct is False:
                pbar.wrong += 1
            else:
                pbar.unknown += 1
            if not parsed.is_valid:
                pbar.parse_fail += 1
            total_done = pbar.correct + pbar.wrong + pbar.unknown
            acc = (pbar.correct / total_done * 100) if total_done else 0
            pbar.set_postfix(
                acc=f"{acc:.0f}%",
                ok=pbar.correct,
                wrong=pbar.wrong,
                unk=pbar.unknown,
                pfail=pbar.parse_fail,
                refresh=True,
            )

        return result

def parse_args():
    parser = ArgumentParser(
        description="Generate CoT reasoning traces for GRACE benchmark (Phase 2).",
    )

    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(ACTIVE_DATASETS),
        help="GRACE dataset to generate traces for.",
    )
    parser.add_argument(
        "--mode", type=str, default="pilot",
        choices=["pilot", "full"],
        help="Generation mode. 'pilot' samples a stratified subset; 'full' runs all examples.",
    )

    parser.add_argument(
        "--model-name", type=str, required=True,
        help="Served model name (must match --served-model-name in vLLM).",
    )
    parser.add_argument(
        "--vllm-base-url", type=str, nargs="+",
        default=["http://localhost:8000/v1"],
        help="One or more vLLM OpenAI-compatible base URLs. "
             "Requests are distributed round-robin across servers.",
    )
    parser.add_argument(
        "--api-key", type=str, default="unused",
        help="API key for the OpenAI-compatible server.",
    )

    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7 per DP-010).")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max tokens for the generated trace.")
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                        help="Enable model thinking mode (default: OFF per DP-010).")

    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory. Default: resources/results/cot_traces/{mode}/{model_name}/",
    )

    parser.add_argument("--max-concurrent", type=int, default=64,
                        help="Max concurrent async requests across all servers.")

    parser.add_argument("--n-repetition", type=int, default=1,
                        help="Number of independent traces to generate per example. "
                             "Each repetition uses the same prompt but temperature-based "
                             "sampling produces varied results. Useful for smaller "
                             "datasets to increase error coverage.")

    parser.add_argument("--pilot-seed", type=int, default=42,
                        help="Random seed for pilot sampling (reproducibility).")

    return parser.parse_args()

async def run(args):
    ds = GraceDataset(args.dataset)
    print(f"Loaded {args.dataset}: {len(ds):,} curated examples")

    if args.mode == "pilot":
        indices = _sample_pilot_indices(ds, seed=args.pilot_seed)
        print(f"Pilot mode: selected {len(indices)} examples (seed={args.pilot_seed})")
    else:
        indices = list(range(len(ds)))
        print(f"Full mode: all {len(indices):,} examples")

    model_dir_name = args.model_name.lower().replace("/", "_")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _RESULTS_DIR / args.mode / model_dir_name
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")

    clients = [
        AsyncOpenAI(base_url=url, api_key=args.api_key)
        for url in args.vllm_base_url
    ]
    print(f"vLLM servers ({len(clients)}): {args.vllm_base_url}")

    cached_keys = load_cached_keys(output_dir, args.dataset)
    if cached_keys:
        print(f"Cache: {len(cached_keys)} traces already done — will skip")

    n_rep = args.n_repetition
    if n_rep > 1:
        print(f"Repetitions: {n_rep} traces per example "
              f"(temperature={args.temperature} ensures diversity)")
    pending = []
    for idx in indices:
        ex = ds[idx]
        for rep in range(n_rep):
            if (ex.id, rep) not in cached_keys:
                pending.append((idx, ex, rep))
    total_expected = len(indices) * n_rep
    print(f"Pending: {len(pending):,} / {total_expected:,} traces to generate")

    if not pending:
        print("Nothing to do — all selected examples are cached.")
        return

    lock = asyncio.Lock()
    t_start = time.time()
    rep_label = f" (×{n_rep})" if n_rep > 1 else ""
    pbar = tqdm(total=len(pending), desc=f"Generating {args.dataset}{rep_label}", unit="trace")
    pbar.correct = 0
    pbar.wrong = 0
    pbar.unknown = 0
    pbar.parse_fail = 0

    per_server = max(1, args.max_concurrent // len(clients))
    semaphores = [asyncio.Semaphore(per_server) for _ in clients]
    print(f"Max concurrent per server: {per_server} × {len(clients)} = {per_server * len(clients)} total")

    tasks = []
    for seq, (idx, example, rep) in enumerate(pending):
        server_idx = seq % len(clients)
        tasks.append(
            generate_one_trace(
                example, clients[server_idx], semaphores[server_idx],
                args, output_dir, lock, pbar, rep_index=rep,
            )
        )

    results = await asyncio.gather(*tasks)
    pbar.close()
    results = [r for r in results if r is not None]

    total_time = time.time() - t_start
    n = len(results)
    cache_file = _cache_path(output_dir, args.dataset)

    total_done = pbar.correct + pbar.wrong + pbar.unknown
    acc = (pbar.correct / total_done * 100) if total_done else 0

    step_counts = [r["trace"]["num_steps"] for r in results if r["trace"]["is_valid"]]
    valid_count = len(step_counts)

    print(f"\n{'━' * 60}")
    print(f"  SUMMARY — {args.dataset} × {args.model_name}")
    print(f"{'━' * 60}")
    print(f"  Mode             : {args.mode}")
    if n_rep > 1:
        print(f"  Repetitions      : {n_rep} per example")
    print(f"  Newly generated  : {n:,}")
    print(f"  Previously cached: {len(cached_keys):,}")
    print(f"  Accuracy         : {pbar.correct}/{total_done} ({acc:.1f}%)")
    print(f"  Parse failures   : {pbar.parse_fail}")
    if step_counts:
        print(f"  Steps/trace      : avg={sum(step_counts)/len(step_counts):.1f}, "
              f"min={min(step_counts)}, max={max(step_counts)}")
    print(f"  Valid traces     : {valid_count}/{n}")
    print(f"  Time             : {total_time:.1f}s")
    if n > 0:
        print(f"  Throughput       : {n / max(total_time, 0.1):.1f} traces/s")
    print(f"  Output           : {cache_file}")
    print(f"{'━' * 60}")

def main():
    args = parse_args()

    print("=" * 60)
    print("  GRACE CoT Trace Generation")
    print("=" * 60)
    print(json.dumps(vars(args), indent=4, ensure_ascii=False))
    print()

    asyncio.run(run(args))

if __name__ == "__main__":
    main()
