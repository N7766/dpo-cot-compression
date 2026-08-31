#!/usr/bin/env python
"""Summarise fixed final-evaluation outputs and paired base-model comparisons."""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_parent_dir, load_yaml, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/final_controlled_eval.yaml")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def mcnemar_exact_pvalue(base_only: int, candidate_only: int) -> float:
    discordant = base_only + candidate_only
    if discordant == 0:
        return 1.0
    smaller = min(base_only, candidate_only)
    tail = sum(math.comb(discordant, count) for count in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def bootstrap_mean_ci(deltas: list[float], samples: int, seed: int = 42) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    rng = random.Random(seed)
    size = len(deltas)
    means = []
    for _ in range(samples):
        means.append(mean(deltas[rng.randrange(size)] for _ in range(size)))
    return [percentile(means, 0.025), percentile(means, 0.975)]


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [record for record in records if record.get("is_correct") is True]
    tokens = [int(record.get("output_tokens", 0)) for record in records]
    words = [int(record.get("output_words", 0)) for record in records]
    return {
        "num_samples": len(records),
        "correct": len(correct),
        "accuracy": len(correct) / len(records) if records else 0.0,
        "mean_output_tokens": mean(tokens) if tokens else 0.0,
        "median_output_tokens": median(tokens) if tokens else 0.0,
        "p95_output_tokens": percentile([float(value) for value in tokens], 0.95),
        "mean_output_words": mean(words) if words else 0.0,
        "answer_tag_rate": sum(bool(record.get("has_answer_tag")) for record in records) / len(records) if records else 0.0,
        "complete_rate": sum(bool(record.get("is_complete")) for record in records) / len(records) if records else 0.0,
        "cap_hit_count": sum(bool(record.get("hit_max_new_tokens")) for record in records),
    }


def paired_against_base(base_records: list[dict[str, Any]], candidate_records: list[dict[str, Any]], bootstrap_samples: int) -> dict[str, Any]:
    base_by_id = {str(record["id"]): record for record in base_records}
    candidate_by_id = {str(record["id"]): record for record in candidate_records}
    ids = sorted(set(base_by_id) & set(candidate_by_id))
    if not ids:
        raise ValueError("No overlapping IDs for paired analysis.")
    base_only = candidate_only = shorter = longer = ties = 0
    deltas: list[float] = []
    for row_id in ids:
        base = base_by_id[row_id]
        candidate = candidate_by_id[row_id]
        if base.get("is_correct") and not candidate.get("is_correct"):
            base_only += 1
        if candidate.get("is_correct") and not base.get("is_correct"):
            candidate_only += 1
        delta = int(candidate["output_tokens"]) - int(base["output_tokens"])
        deltas.append(delta)
        if delta < 0:
            shorter += 1
        elif delta > 0:
            longer += 1
        else:
            ties += 1
    return {
        "matched_samples": len(ids),
        "candidate_gain_base_loss": candidate_only,
        "candidate_loss_base_gain": base_only,
        "mcnemar_exact_pvalue": mcnemar_exact_pvalue(base_only, candidate_only),
        "mean_token_delta_candidate_minus_base": mean(deltas),
        "median_token_delta_candidate_minus_base": median(deltas),
        "mean_token_delta_95pct_bootstrap_ci": bootstrap_mean_ci(deltas, bootstrap_samples),
        "candidate_shorter": shorter,
        "candidate_longer": longer,
        "token_ties": ties,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    output_dir = Path(args.input_dir or cfg["paths"]["output_dir"])
    output_file = Path(args.output_file or cfg["paths"]["analysis_output"])
    markdown_file = Path(args.markdown_file or cfg["paths"]["analysis_markdown"])
    expected_models = [model["id"] for model in cfg["models"]]
    base_id = next(model["id"] for model in cfg["models"] if model["kind"] == "base")
    expected_prompts = [prompt["id"] for prompt in cfg["prompts"]]

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for prompt_id in expected_prompts:
        for model_id in expected_models:
            path = output_dir / prompt_id / f"{model_id}.jsonl"
            if path.exists():
                grouped[prompt_id][model_id] = read_jsonl(path)

    analyses: dict[str, Any] = {}
    lines = ["# Final Controlled Evaluation", "", "All outputs use fixed prompts and deterministic decoding.", ""]
    for prompt_id in expected_prompts:
        model_records = grouped.get(prompt_id, {})
        if not model_records:
            continue
        prompt_analysis: dict[str, Any] = {"models": {}, "paired_against_base": {}}
        lines.extend([f"## Prompt: `{prompt_id}`", "", "| Model | Accuracy | Correct | Mean completion tokens | Median | P95 | Answer-tag rate | Cap hits |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for model_id in expected_models:
            records = model_records.get(model_id)
            if not records:
                continue
            metrics = metric_summary(records)
            prompt_analysis["models"][model_id] = metrics
            lines.append(
                f"| {model_id} | {metrics['accuracy']:.2%} | {metrics['correct']}/{metrics['num_samples']} | "
                f"{metrics['mean_output_tokens']:.2f} | {metrics['median_output_tokens']:.0f} | "
                f"{metrics['p95_output_tokens']:.0f} | {metrics['answer_tag_rate']:.2%} | {metrics['cap_hit_count']} |"
            )
        base_records = model_records.get(base_id)
        if base_records:
            for model_id, records in model_records.items():
                if model_id == base_id:
                    continue
                paired = paired_against_base(base_records, records, args.bootstrap_samples)
                prompt_analysis["paired_against_base"][model_id] = paired
                lines.append("")
                lines.append(
                    f"`{model_id}` vs `{base_id}`: accuracy discordance candidate gain/base loss = "
                    f"{paired['candidate_gain_base_loss']}/{paired['candidate_loss_base_gain']}; "
                    f"McNemar p={paired['mcnemar_exact_pvalue']:.4f}; mean token delta = "
                    f"{paired['mean_token_delta_candidate_minus_base']:.2f} "
                    f"(95% bootstrap CI {paired['mean_token_delta_95pct_bootstrap_ci'][0]:.2f}, "
                    f"{paired['mean_token_delta_95pct_bootstrap_ci'][1]:.2f}); shorter/longer/tie = "
                    f"{paired['candidate_shorter']}/{paired['candidate_longer']}/{paired['token_ties']}."
                )
        lines.append("")
        analyses[prompt_id] = prompt_analysis

    payload = {
        "experiment": cfg["experiment"],
        "token_metric": "Tokenizer completion tokens excluding padding and EOS.",
        "word_metric": "Whitespace-separated visible units retained only for historic API comparison.",
        "prompt_analyses": analyses,
    }
    ensure_parent_dir(output_file)
    write_json(output_file, payload)
    markdown_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_file}")
    print(f"Wrote {markdown_file}")


if __name__ == "__main__":
    main()
