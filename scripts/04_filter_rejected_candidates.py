#!/usr/bin/env python
"""Analyze generations and filter rejected-candidate groups."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_yaml, read_jsonl, write_json
from src.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_baseline.yaml")
    return parser.parse_args()


def analysis_output_path(cfg: dict) -> str:
    paths = cfg.get("paths", {})
    if "analysis_output" in paths:
        return paths["analysis_output"]

    metrics_path = paths.get(
        "metrics_output",
        cfg.get("outputs", {}).get("metrics_path", "outputs/results/baseline_analysis.json"),
    )
    if metrics_path.endswith("_metrics.json"):
        return metrics_path.replace("_metrics.json", "_analysis.json")
    return str(Path(metrics_path).with_name(Path(metrics_path).stem + "_analysis.json"))


def result_path(filename: str) -> str:
    return str(PROJECT_ROOT / "outputs" / "results" / filename)


def mean_or_none(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def preview(text: str, limit: int = 220) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def build_analysis(records: list[dict]) -> dict:
    output_tokens = [int(row.get("output_tokens", 0)) for row in records]
    correct_rows = [row for row in records if bool(row.get("is_correct"))]
    incorrect_rows = [row for row in records if not bool(row.get("is_correct"))]
    correct_nontrivial_rows = [
        row for row in records if bool(row.get("is_correct")) and int(row.get("output_tokens", 0)) > 20
    ]
    answer_only_rows = [
        row for row in records if bool(row.get("is_correct")) and int(row.get("output_tokens", 0)) <= 5
    ]
    overthinking_failure_rows = [
        row
        for row in records
        if int(row.get("output_tokens", 0)) >= 500 and not bool(row.get("is_correct"))
    ]
    correct_tokens = [int(row.get("output_tokens", 0)) for row in correct_rows]
    incorrect_tokens = [int(row.get("output_tokens", 0)) for row in incorrect_rows]
    correct_nontrivial_tokens = [int(row.get("output_tokens", 0)) for row in correct_nontrivial_rows]
    answer_only_tokens = [int(row.get("output_tokens", 0)) for row in answer_only_rows]
    overthinking_failure_tokens = [int(row.get("output_tokens", 0)) for row in overthinking_failure_rows]

    suspicious = []
    for row in records:
        tokens = int(row.get("output_tokens", 0))
        is_correct = bool(row.get("is_correct"))
        reasons = []
        if tokens <= 5:
            reasons.append("output_tokens <= 5")
        if tokens >= 500:
            reasons.append("output_tokens >= 500")
        if not is_correct:
            reasons.append("is_correct == false")
        if reasons:
            suspicious.append(
                {
                    "id": row.get("id"),
                    "reasons": reasons,
                    "output_tokens": tokens,
                    "is_correct": is_correct,
                    "is_complete": row.get("is_complete"),
                    "num_retries": row.get("num_retries"),
                    "pred_answer": row.get("pred_answer"),
                    "gold_answer": row.get("gold_answer"),
                    "prediction_preview": preview(row.get("prediction", "")),
                }
            )

    total = len(records)
    correct = len(correct_rows)
    return {
        "total_samples": total,
        "correct_samples": correct,
        "incorrect_samples": len(incorrect_rows),
        "accuracy": correct / total if total else 0.0,
        "avg_tokens_all": mean_or_none(output_tokens),
        "avg_tokens_correct": mean_or_none(correct_tokens),
        "avg_tokens_incorrect": mean_or_none(incorrect_tokens),
        "avg_tokens_correct_nontrivial": mean_or_none(correct_nontrivial_tokens),
        "avg_tokens_answer_only": mean_or_none(answer_only_tokens),
        "avg_tokens_overthinking_failures": mean_or_none(overthinking_failure_tokens),
        "num_answer_only": len(answer_only_rows),
        "num_nontrivial_correct": len(correct_nontrivial_rows),
        "num_overthinking_failures": len(overthinking_failure_rows),
        "avg_output_tokens": mean_or_none(output_tokens),
        "median_output_tokens": statistics.median(output_tokens) if output_tokens else None,
        "min_output_tokens": min(output_tokens) if output_tokens else None,
        "max_output_tokens": max(output_tokens) if output_tokens else None,
        "num_output_tokens_lte_5": sum(tokens <= 5 for tokens in output_tokens),
        "num_output_tokens_lte_20": sum(tokens <= 20 for tokens in output_tokens),
        "num_output_tokens_gte_500": sum(tokens >= 500 for tokens in output_tokens),
        "avg_output_tokens_correct": mean_or_none(correct_tokens),
        "avg_output_tokens_incorrect": mean_or_none(incorrect_tokens),
        "suspicious_samples": suspicious,
    }


def build_filtered_groups(records: list[dict]) -> dict[str, list[dict]]:
    return {
        "correct_nontrivial": [
            row for row in records if bool(row.get("is_correct")) and int(row.get("output_tokens", 0)) > 20
        ],
        "answer_only": [
            row for row in records if bool(row.get("is_correct")) and int(row.get("output_tokens", 0)) <= 5
        ],
        "overthinking_failures": [
            row
            for row in records
            if int(row.get("output_tokens", 0)) >= 500 or not bool(row.get("is_correct"))
        ],
    }


def filtered_output_paths(cfg: dict) -> dict[str, str]:
    paths = cfg.get("paths", {})
    return {
        "correct_nontrivial": paths.get(
            "correct_nontrivial_output",
            result_path("baseline_correct_nontrivial.jsonl"),
        ),
        "answer_only": paths.get(
            "answer_only_output",
            result_path("baseline_answer_only.jsonl"),
        ),
        "overthinking_failures": paths.get(
            "overthinking_failures_output",
            result_path("baseline_overthinking_failures.jsonl"),
        ),
    }


def write_filtered_groups(groups: dict[str, list[dict]], cfg: dict) -> dict[str, str]:
    output_paths = filtered_output_paths(cfg)
    for name, rows in groups.items():
        write_jsonl(output_paths[name], rows)
    return output_paths


def print_summary(analysis: dict, input_path: str, output_path: str, filtered_paths: dict[str, str]) -> None:
    print("Baseline Generation Analysis")
    print("=" * 32)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print()
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Correct samples: {analysis['correct_samples']}")
    print(f"Incorrect samples: {analysis['incorrect_samples']}")
    print(f"Nontrivial correct samples: {analysis['num_nontrivial_correct']}")
    print(f"Answer-only samples: {analysis['num_answer_only']}")
    print(f"Overthinking failures: {analysis['num_overthinking_failures']}")
    print(f"Accuracy: {analysis['accuracy']:.4f}")
    print()
    print("Group average output tokens")
    print(f"  All samples: {analysis['avg_tokens_all']}")
    print(f"  Correct samples: {analysis['avg_tokens_correct']}")
    print(f"  Incorrect samples: {analysis['avg_tokens_incorrect']}")
    print(f"  Correct nontrivial: {analysis['avg_tokens_correct_nontrivial']}")
    print(f"  Answer-only: {analysis['avg_tokens_answer_only']}")
    print(f"  Overthinking failures: {analysis['avg_tokens_overthinking_failures']}")
    print()
    print("Output token length")
    print(f"  Median: {analysis['median_output_tokens']}")
    print(f"  Min: {analysis['min_output_tokens']}")
    print(f"  Max: {analysis['max_output_tokens']}")
    print(f"  <= 5 tokens: {analysis['num_output_tokens_lte_5']}")
    print(f"  <= 20 tokens: {analysis['num_output_tokens_lte_20']}")
    print(f"  >= 500 tokens: {analysis['num_output_tokens_gte_500']}")
    print()
    print("Filtered files")
    print(f"  Correct nontrivial: {filtered_paths['correct_nontrivial']}")
    print(f"  Answer-only: {filtered_paths['answer_only']}")
    print(f"  Overthinking failures: {filtered_paths['overthinking_failures']}")
    print()
    print(f"Suspicious samples: {len(analysis['suspicious_samples'])}")
    for item in analysis["suspicious_samples"]:
        print(
            f"  - {item['id']}: tokens={item['output_tokens']}, "
            f"correct={item['is_correct']}, complete={item['is_complete']}, "
            f"retries={item['num_retries']}, pred={item['pred_answer']}, "
            f"gold={item['gold_answer']}, reasons={'; '.join(item['reasons'])}"
        )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    generation_path = cfg.get("paths", {}).get(
        "generation_output",
        cfg.get("outputs", {}).get("generations_path"),
    )
    output_path = analysis_output_path(cfg)

    records = read_jsonl(generation_path)
    analysis = build_analysis(records)
    groups = build_filtered_groups(records)
    filtered_paths = write_filtered_groups(groups, cfg)
    write_json(output_path, analysis)
    print_summary(analysis, generation_path, output_path, filtered_paths)


if __name__ == "__main__":
    main()
