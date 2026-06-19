#!/usr/bin/env python
"""Evaluate saved baseline generations."""

from __future__ import annotations

import argparse
import tempfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.answer_extraction import answers_match, extract_final_answer, extract_gold_answer, extract_pred_answer
from src.evaluation.metrics import compute_baseline_metrics
from src.utils.io import load_yaml, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--self_test", action="store_true", help="Run lightweight sanity checks.")
    return parser.parse_args()


def run_self_test() -> None:
    examples = {
        "#### 72": "72",
        "Answer: 72": "72",
        "The answer is 72.": "72",
        "Therefore, the answer is 1,234.": "1234",
        "work... final answer: -3.5": "-3.5",
        "2 + 2 = 4. Answer: 5": "5",
    }
    for text, expected in examples.items():
        got = extract_pred_answer(text)
        assert got == expected, f"{text!r}: expected {expected}, got {got}"

    assert extract_gold_answer("some reasoning #### 72") == "72"
    assert answers_match("72.0", "72")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.jsonl"
        rows = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]
        write_jsonl(path, rows)
        assert read_jsonl(path) == rows

    print("Self-test passed: answer extraction and JSONL IO look good.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    cfg = load_yaml(args.config)
    generation_path = cfg.get("paths", {}).get(
        "generation_output",
        cfg.get("outputs", {}).get("generations_path"),
    )
    metrics_path = cfg.get("paths", {}).get(
        "metrics_output",
        cfg.get("outputs", {}).get("metrics_path"),
    )
    records = read_jsonl(generation_path)

    # Recompute correctness in case generation files were edited or produced by another runner.
    for row in records:
        row["pred_answer"] = row.get("pred_answer") or extract_final_answer(row.get("prediction", ""))
        is_complete = bool(row.get("is_complete", True))
        row["is_correct"] = is_complete and answers_match(row["pred_answer"], row.get("gold_answer"))

    metrics = compute_baseline_metrics(records)
    write_json(metrics_path, metrics)
    print(metrics)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
