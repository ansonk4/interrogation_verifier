from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from src.end_evaluation.prompts import load_predictions




def compute_binary_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    tp = fp = fn = tn = 0
    for gt, pred in zip(y_true, y_pred):
        gt_pos = gt == "unfaithful"
        pred_pos = pred == "unfaithful"
        if gt_pos and pred_pos:
            tp += 1
        elif not gt_pos and pred_pos:
            fp += 1
        elif gt_pos and not pred_pos:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
        "support_positive": tp + fn,
        "support_negative": tn + fp,
    }


def compute_multiclass_metrics(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    per_class = {}
    for cls in classes:
        tp = sum(1 for g, p in zip(y_true, y_pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(y_true, y_pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(y_true, y_pred) if g == cls and p != cls)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }


    active = [c for c in classes if per_class[c]["support"] > 0]
    macro_p = sum(per_class[c]["precision"] for c in active) / len(active) if active else 0.0
    macro_r = sum(per_class[c]["recall"] for c in active) / len(active) if active else 0.0
    macro_f1 = sum(per_class[c]["f1"] for c in active) / len(active) if active else 0.0

    accuracy = sum(1 for g, p in zip(y_true, y_pred) if g == p) / len(y_true) if y_true else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "total": len(y_true),
    }


def compute_confusion_matrix(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:

    matrix = {g: {p: 0 for p in classes} for g in classes}
    for gt, pred in zip(y_true, y_pred):
        if gt in matrix and pred in matrix[gt]:
            matrix[gt][pred] += 1
    return matrix




def extract_step_labels(records: list[dict]) -> dict:

    f_gold, f_pred = [], []
    ec_gold, ec_pred = [], []
    datasets, tracks = [], []

    for record in records:
        ds = record.get("dataset", "")
        track = record.get("track", "")

        for step in record.get("step_predictions", []):
            gold_f = step.get("gold_faithfulness")
            pred_f = step.get("pred_faithfulness")


            if not gold_f:
                continue


            if not pred_f:
                pred_f = "faithful"

            f_gold.append(gold_f)
            f_pred.append(pred_f)
            datasets.append(ds)
            tracks.append(track)


            if gold_f == "unfaithful" and pred_f == "unfaithful":
                gold_ec = step.get("gold_error_category") or "unknown"
                pred_ec = step.get("pred_error_category") or "unknown"
                ec_gold.append(gold_ec)
                ec_pred.append(pred_ec)

    return {
        "faithfulness_gold": f_gold,
        "faithfulness_pred": f_pred,
        "error_category_gold": ec_gold,
        "error_category_pred": ec_pred,
        "datasets": datasets,
        "tracks": tracks,
    }


def evaluate_model(records: list[dict], model_name: str = "") -> dict:

    labels = extract_step_labels(records)

    f_gold = labels["faithfulness_gold"]
    f_pred = labels["faithfulness_pred"]
    ec_gold = labels["error_category_gold"]
    ec_pred = labels["error_category_pred"]


    binary = compute_binary_metrics(f_gold, f_pred)
    faith_classes = ["faithful", "unfaithful"]
    faith_cm = compute_confusion_matrix(f_gold, f_pred, faith_classes)


    ec_classes = sorted(set(ec_gold + ec_pred))
    ec_metrics = compute_multiclass_metrics(ec_gold, ec_pred, ec_classes) if ec_gold else {}
    ec_cm = compute_confusion_matrix(ec_gold, ec_pred, ec_classes) if ec_gold else {}


    per_dataset = {}
    unique_datasets = sorted(set(labels["datasets"]))
    for ds in unique_datasets:
        mask = [i for i, d in enumerate(labels["datasets"]) if d == ds]
        ds_f_gold = [f_gold[i] for i in mask]
        ds_f_pred = [f_pred[i] for i in mask]

        ds_ec_gold, ds_ec_pred = [], []
        ec_idx = 0
        for i in range(len(f_gold)):
            if f_gold[i] == "unfaithful" and f_pred[i] == "unfaithful":
                if labels["datasets"][i] == ds:
                    ds_ec_gold.append(ec_gold[ec_idx])
                    ds_ec_pred.append(ec_pred[ec_idx])
                ec_idx += 1
        ds_ec_classes = sorted(set(ds_ec_gold + ds_ec_pred)) if ds_ec_gold else []
        per_dataset[ds] = {
            "binary": compute_binary_metrics(ds_f_gold, ds_f_pred),
            "error_category": compute_multiclass_metrics(ds_ec_gold, ds_ec_pred, ds_ec_classes) if ds_ec_gold else {},
            "n_steps": len(mask),
        }


    per_track = {}
    unique_tracks = sorted(set(labels["tracks"]))
    for track in unique_tracks:
        mask = [i for i, t in enumerate(labels["tracks"]) if t == track]
        t_f_gold = [f_gold[i] for i in mask]
        t_f_pred = [f_pred[i] for i in mask]
        per_track[track] = {
            "binary": compute_binary_metrics(t_f_gold, t_f_pred),
            "n_steps": len(mask),
        }


    ds_f1s = [per_dataset[ds]["binary"]["f1"] for ds in per_dataset]
    ds_ec_f1s = [per_dataset[ds]["error_category"]["macro_f1"]
                 for ds in per_dataset if per_dataset[ds].get("error_category")]
    macro_avg = {
        "step_f1": sum(ds_f1s) / len(ds_f1s) if ds_f1s else 0.0,
        "cat_f1": sum(ds_ec_f1s) / len(ds_ec_f1s) if ds_ec_f1s else 0.0,
        "n_datasets": len(ds_f1s),
        "n_datasets_with_ec": len(ds_ec_f1s),
    }

    return {
        "model": model_name,
        "n_traces": len(records),
        "n_steps": len(f_gold),
        "n_unfaithful_steps_both": len(ec_gold),
        "overall": {
            "faithfulness": binary,
            "faithfulness_confusion_matrix": faith_cm,
            "error_category": ec_metrics,
            "error_category_confusion_matrix": ec_cm,
        },
        "macro_avg": macro_avg,
        "per_dataset": per_dataset,
        "per_track": per_track,
    }




def format_results(results: dict) -> str:

    lines = []
    model = results.get("model", "unknown")
    lines.append(f"{'━' * 70}")
    lines.append(f"  Evaluation Results — {model}")
    lines.append(f"{'━' * 70}")
    n_both = results.get("n_unfaithful_steps_both", 0)
    lines.append(f"  Traces: {results['n_traces']:,}  |  Steps: {results['n_steps']:,}  |  Unfaithful (both): {n_both:,}")
    lines.append("")


    b = results["overall"]["faithfulness"]
    lines.append("  ── Step-level Faithfulness F1 (faithful vs. unfaithful) ──")
    lines.append(f"    Accuracy:  {b['accuracy']:.4f}")
    lines.append(f"    Precision: {b['precision']:.4f}")
    lines.append(f"    Recall:    {b['recall']:.4f}")
    lines.append(f"    F1:        {b['f1']:.4f}")
    lines.append(f"    TP={b['tp']}  FP={b['fp']}  FN={b['fn']}  TN={b['tn']}")
    lines.append("")


    ec = results["overall"].get("error_category")
    if ec:
        lines.append(f"  ── Error Category F1 (both-unfaithful, n={n_both}) ──")
        lines.append(f"    Accuracy:  {ec['accuracy']:.4f}")
        lines.append(f"    Macro F1:  {ec['macro_f1']:.4f}")
        for cls, metrics in ec["per_class"].items():
            if metrics["support"] > 0:
                lines.append(f"      {cls:25s}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  (n={metrics['support']})")
        lines.append("")


    lines.append("  ── Per-Dataset Breakdown ──")
    lines.append(f"    {'Dataset':20s} {'Steps':>6s} {'Acc':>7s} {'F1':>7s} {'P':>7s} {'R':>7s} {'EC-F1':>7s}")
    for ds, dm in results["per_dataset"].items():
        b = dm["binary"]
        ec = dm.get("error_category", {})
        ec_f1 = ec.get("macro_f1", 0.0) if ec else 0.0
        lines.append(f"    {ds:20s} {dm['n_steps']:>6d} {b['accuracy']:>7.3f} {b['f1']:>7.3f} {b['precision']:>7.3f} {b['recall']:>7.3f} {ec_f1:>7.3f}")
    ma = results.get("macro_avg", {})
    if ma:
        lines.append(f"    {'── Avg (macro) ──':20s} {'':>6s} {'':>7s} {ma['step_f1']:>7.3f} {'':>7s} {'':>7s} {ma['cat_f1']:>7.3f}")
    lines.append("")


    lines.append("  ── Per-Track Breakdown ──")
    lines.append(f"    {'Track':20s} {'Steps':>6s} {'Acc':>7s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    for track, tm in results["per_track"].items():
        b = tm["binary"]
        lines.append(f"    {track:20s} {tm['n_steps']:>6d} {b['accuracy']:>7.3f} {b['f1']:>7.3f} {b['precision']:>7.3f} {b['recall']:>7.3f}")

    lines.append(f"{'━' * 70}")
    return "\n".join(lines)




def parse_args():
    parser = ArgumentParser(
        description="Evaluate GRACE faithfulness prediction results.",
    )
    parser.add_argument(
        "--input", type=str, nargs="+", required=True,
        help="Path(s) to prediction directories. Each directory should "
             "contain dataset-level JSONL files.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write evaluation reports (JSON + text).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for input_path_str in args.input:
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"[WARN] {input_path} does not exist, skipping")
            continue


        records = []
        for jsonl_file in sorted(input_path.glob("*.jsonl")):
            records.extend(load_predictions(jsonl_file))

        if not records:
            print(f"[WARN] No predictions found in {input_path}")
            continue

        model_name = input_path.name or records[0].get("model", "unknown")

        print(f"\nEvaluating: {model_name} ({len(records):,} traces from {input_path})")
        results = evaluate_model(records, model_name)


        report = format_results(results)
        print(report)


        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            safe_name = model_name.lower().replace("/", "_").replace(" ", "_")

            json_path = out_dir / f"{safe_name}_results.json"
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  → JSON: {json_path}")

            txt_path = out_dir / f"{safe_name}_report.txt"
            with open(txt_path, "w") as f:
                f.write(report + "\n")
            print(f"  → Report: {txt_path}")


if __name__ == "__main__":
    main()
