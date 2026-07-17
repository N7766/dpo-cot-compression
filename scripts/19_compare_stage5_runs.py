#!/usr/bin/env python
"""Compare Stage 5 checkpoint evaluation summaries with Stage 3/4 baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_parent_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="outputs/results")
    parser.add_argument(
        "--output_file",
        default="outputs/results/stage5_model_comparison.json",
    )
    parser.add_argument(
        "--markdown_file",
        default="outputs/results/stage5_model_comparison.md",
    )
    parser.add_argument("--min_accuracy", type=float, default=0.93)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_summary_file"] = str(path)
    return data


def collect_summaries(results_root: Path) -> list[dict]:
    summaries: list[dict] = []
    baseline_paths = [
        results_root / "stage3_lora_sft" / "val_eval_full_b32.summary.json",
        results_root / "stage4_full_dpo_h200" / "val_eval_full_b16.summary.json",
    ]
    for path in baseline_paths:
        if path.exists():
            summaries.append(load_summary(path))

    for path in sorted(results_root.glob("stage5_*/*/checkpoint_evals/*.summary.json")):
        summaries.append(load_summary(path))
    for path in sorted(results_root.glob("stage5_*/*/*_eval.summary.json")):
        if "checkpoint_evals" not in str(path):
            summaries.append(load_summary(path))
    return summaries


def label_for(summary: dict) -> str:
    path = Path(summary["_summary_file"])
    if "stage3_lora_sft" in path.parts:
        return "stage3_lora_sft_baseline"
    if "stage4_full_dpo_h200" in path.parts:
        return "stage4_full_dpo_h200"
    checkpoint = path.stem.replace("_eval.summary", "")
    run = path.parents[1].name if path.parent.name == "checkpoint_evals" else path.parent.name
    return f"{run}/{checkpoint}"


def main() -> None:
    args = parse_args()
    summaries = collect_summaries(Path(args.results_root))
    rows = []
    for summary in summaries:
        accuracy = float(summary.get("accuracy", 0.0))
        avg_tokens = float(summary.get("avg_output_tokens", 0.0))
        rows.append(
            {
                "label": label_for(summary),
                "num_samples": summary.get("num_samples"),
                "correct": summary.get("correct"),
                "accuracy": accuracy,
                "avg_output_tokens": avg_tokens,
                "model_dir": summary.get("model_dir"),
                "summary_file": summary["_summary_file"],
                "meets_accuracy_target": accuracy >= args.min_accuracy,
            }
        )

    rows.sort(key=lambda row: (row["meets_accuracy_target"], row["accuracy"], -row["avg_output_tokens"]), reverse=True)
    ensure_parent_dir(args.output_file)
    write_json(args.output_file, {"min_accuracy": args.min_accuracy, "runs": rows})

    lines = [
        "# Stage 5 Model Comparison",
        "",
        f"Accuracy target: {args.min_accuracy * 100:.1f}%",
        "",
        "| Rank | Run | Accuracy | Correct | Avg tokens | Meets target |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | `{label}` | {acc:.2f}% | {correct} | {tokens:.2f} | {target} |".format(
                rank=idx,
                label=row["label"],
                acc=row["accuracy"] * 100,
                correct=row["correct"],
                tokens=row["avg_output_tokens"],
                target="yes" if row["meets_accuracy_target"] else "no",
            )
        )
    Path(args.markdown_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_file}")
    print(f"Wrote {args.markdown_file}")
    if rows:
        print("Best run:", rows[0]["label"])


if __name__ == "__main__":
    main()
