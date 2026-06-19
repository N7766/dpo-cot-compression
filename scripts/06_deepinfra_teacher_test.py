#!/usr/bin/env python
"""Smoke-test DeepInfra teacher generation for future DPO chosen responses."""

from __future__ import annotations

import argparse
import os
import sys
from statistics import mean
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.answer_extraction import answers_match, extract_final_answer
from src.utils.io import append_jsonl, load_yaml, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deepinfra_test.yaml")
    parser.add_argument("--max_samples", type=int, default=3)
    return parser.parse_args()


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem briefly.\n\n"
        "Use the minimum reasoning necessary.\n\n"
        "Do not skip essential calculations.\n\n"
        "End with the final answer exactly in this format:\n\n"
        "Answer: <number>\n\n"
        "Question:\n\n"
        f"{question}"
    )


def approximate_token_count(text: str) -> int:
    return len((text or "").split())


def compression_ratio(output_tokens: int, rejected_tokens: int) -> float | None:
    if rejected_tokens <= 0:
        return None
    return 1 - (output_tokens / rejected_tokens)


def call_deepinfra(client: OpenAI, model_cfg: dict, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model_cfg["name"],
        messages=[{"role": "user", "content": prompt}],
        temperature=float(model_cfg.get("temperature", 0.0)),
        top_p=float(model_cfg.get("top_p", 1.0)),
        max_tokens=int(model_cfg.get("max_new_tokens", 512)),
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]

    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPINFRA_API_KEY is required. Please run:\n\n"
            "export DEEPINFRA_API_KEY=your_key"
        )

    if model_cfg.get("backend") != "deepinfra":
        raise ValueError("This smoke-test script expects model.backend: deepinfra")

    input_file = paths_cfg["input_file"]
    output_file = paths_cfg["output_file"]
    model_name = model_cfg["name"]
    base_url = model_cfg["base_url"]

    print(f"Backend: deepinfra")
    print(f"Model: {model_name}")
    print(f"Base URL: {base_url}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    samples = read_jsonl(input_file)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    existing_outputs = read_jsonl(output_file)
    completed_ids = {row.get("id") for row in existing_outputs if row.get("id")}
    if completed_ids:
        print(f"Found {len(completed_ids)} existing teacher outputs; completed ids will be skipped.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    max_retries = int(cfg.get("generation", {}).get("max_retries", 2))

    new_results = []
    attempted = 0
    for sample in samples:
        sample_id = sample["id"]
        if sample_id in completed_ids:
            print(f"[{sample_id}] skipping existing output")
            continue

        prompt = build_prompt(sample["question"])
        prediction = ""
        error = None
        for attempt in range(max_retries + 1):
            try:
                prediction = call_deepinfra(client, model_cfg, prompt)
                break
            except Exception as exc:
                error = str(exc)
                if attempt >= max_retries:
                    print(f"[{sample_id}] DeepInfra call failed after {attempt + 1} attempts: {error}")
                else:
                    print(f"[{sample_id}] DeepInfra call failed on attempt {attempt + 1}; retrying: {error}")

        pred_answer = extract_final_answer(prediction)
        output_tokens = approximate_token_count(prediction)
        rejected_tokens = int(sample.get("output_tokens", 0))
        ratio = compression_ratio(output_tokens, rejected_tokens)
        is_correct = answers_match(pred_answer, sample.get("gold_answer"))

        result = {
            "id": sample_id,
            "question": sample["question"],
            "gold_answer": sample.get("gold_answer"),
            "teacher_model": model_name,
            "prompt": prompt,
            "prediction": prediction,
            "pred_answer": pred_answer,
            "is_correct": is_correct,
            "output_tokens": output_tokens,
            "rejected_tokens": rejected_tokens,
            "compression_ratio_vs_rejected": ratio,
        }
        if error and not prediction:
            result["error"] = error

        append_jsonl(output_file, result)
        completed_ids.add(sample_id)
        new_results.append(result)
        attempted += 1

        ratio_text = f"{ratio:.4f}" if ratio is not None else "n/a"
        print(
            f"[{sample_id}] output_tokens={output_tokens}, "
            f"rejected_tokens={rejected_tokens}, "
            f"compression={ratio_text}, "
            f"pred_answer={pred_answer}, "
            f"gold_answer={sample.get('gold_answer')}, "
            f"correct={is_correct}"
        )

    correct_count = sum(bool(row["is_correct"]) for row in new_results)
    avg_output_tokens = mean(row["output_tokens"] for row in new_results) if new_results else 0.0
    valid_ratios = [row["compression_ratio_vs_rejected"] for row in new_results if row["compression_ratio_vs_rejected"] is not None]
    avg_ratio = mean(valid_ratios) if valid_ratios else 0.0

    print()
    print("DeepInfra teacher smoke-test summary")
    print("=" * 40)
    print(f"Total attempted: {attempted}")
    print(f"Correct count: {correct_count}")
    print(f"Avg output tokens: {avg_output_tokens}")
    print(f"Avg compression ratio vs rejected: {avg_ratio}")


if __name__ == "__main__":
    main()
