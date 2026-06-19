"""Metric calculations for baseline GSM8K evaluation."""

from __future__ import annotations

from statistics import mean


def compute_baseline_metrics(records: list[dict]) -> dict:
    num_samples = len(records)
    if num_samples == 0:
        return {
            "num_samples": 0,
            "accuracy": 0.0,
            "avg_output_tokens": 0.0,
            "avg_reasoning_tokens": 0.0,
        }

    return {
        "num_samples": num_samples,
        "accuracy": sum(bool(row.get("is_correct")) for row in records) / num_samples,
        "avg_output_tokens": mean(int(row.get("output_tokens", 0)) for row in records),
        "avg_reasoning_tokens": mean(int(row.get("reasoning_tokens", 0)) for row in records),
    }
