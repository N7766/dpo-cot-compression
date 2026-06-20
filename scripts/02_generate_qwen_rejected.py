#!/usr/bin/env python
"""Generate Qwen3 GSM8K outputs for baseline or rejected-candidate data.

This script uses the Hugging Face remote inference API only. It never loads
transformers models or tokenizers locally, so it should not download model
weights.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.answer_extraction import answers_match, extract_final_answer
from src.utils.io import append_jsonl, load_yaml, read_jsonl


THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_baseline.yaml")
    parser.add_argument("--max_samples", type=int, default=None, help="Override config for smoke tests.")
    return parser.parse_args()


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem.\n\n"
        "Think carefully and explain every reasoning step in detail before giving the final answer.\n\n"
        "At the end, you must write the final result exactly in this format:\n\n"
        "Answer: <number>\n\n"
        f"Question:\n{question}"
    )


def approximate_token_count(text: str) -> int:
    return len((text or "").split())


def count_reasoning_tokens_approx(text: str, output_tokens: int) -> int:
    thinking_blocks = THINK_RE.findall(text or "")
    if not thinking_blocks:
        return output_tokens
    return sum(approximate_token_count(block) for block in thinking_blocks)


def response_to_text(response: Any) -> str:
    """Normalize Hugging Face API responses across client versions."""
    def clean(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text and text.lower() != "none" else None

    if isinstance(response, str):
        return response.strip()
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message"):
            for attr in ("content", "reasoning_content"):
                if hasattr(choice.message, attr):
                    text = clean(getattr(choice.message, attr))
                    if text is not None:
                        return text
        if hasattr(choice, "text"):
            text = clean(choice.text)
            if text is not None:
                return text
    if hasattr(response, "generated_text"):
        text = clean(response.generated_text)
        if text is not None:
            return text
    if isinstance(response, dict):
        choices = response.get("choices")
        if choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message", {})
                if isinstance(message, dict):
                    for key in ("content", "reasoning_content"):
                        if key in message:
                            text = clean(message[key])
                            if text is not None:
                                return text
                if "text" in first_choice:
                    text = clean(first_choice["text"])
                    if text is not None:
                        return text
        for key in ("generated_text", "text", "output_text"):
            if key in response:
                text = clean(response[key])
                if text is not None:
                    return text
    if isinstance(response, list) and response:
        return response_to_text(response[0])
    return ""


def run_hf_api(prompt: str, model_name: str, model_cfg: dict, hf_token: str) -> str:
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=model_name, token=hf_token)
    max_new_tokens = int(model_cfg.get("max_new_tokens", 512))
    temperature = float(model_cfg.get("temperature", 0.0))
    top_p = float(model_cfg.get("top_p", 1.0))

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return response_to_text(response)


def has_answer_tag(text: str) -> bool:
    return re.search(r"answer\s*:", text or "", flags=re.IGNORECASE) is not None


def check_completion(prediction: str) -> tuple[bool, bool, str | None]:
    answer_tag = has_answer_tag(prediction)
    pred_answer = extract_final_answer(prediction)
    return answer_tag, answer_tag and pred_answer is not None, pred_answer


def make_error_record(row: dict, prompt: str, error: str, num_retries: int) -> dict:
    return {
        "id": row["id"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "prompt": prompt,
        "prediction": "",
        "pred_answer": None,
        "is_correct": False,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "has_answer_tag": False,
        "num_retries": num_retries,
        "is_complete": False,
        "error": error,
    }


def generate_one(
    row: dict,
    idx: int,
    total: int,
    model_name: str,
    model_cfg: dict,
    hf_token: str,
    max_retries: int,
    retry_sleep_seconds: float,
) -> tuple[dict | None, bool]:
    prompt = build_prompt(row["question"])
    prediction = ""
    pred_answer = None
    answer_tag = False
    is_complete = False
    error = None
    num_retries = 0

    for attempt in range(max_retries + 1):
        num_retries = attempt
        try:
            prediction = run_hf_api(prompt, model_name, model_cfg, hf_token)
        except Exception as exc:
            error = str(exc)
            if attempt < max_retries:
                print(
                    f"[{idx}/{total}] API call failed for {row['id']} "
                    f"on attempt {attempt + 1}; retrying: {error}"
                )
                time.sleep(retry_sleep_seconds * (2**attempt))
                continue
            print(
                f"[{idx}/{total}] API call failed for {row['id']} "
                f"after {attempt + 1} attempts; not writing this sample so it can be retried later: {error}"
            )
            return None, True

        if not prediction:
            print(f"[{idx}/{total}] WARNING: empty API response for sample {row['id']}")
        answer_tag, is_complete, pred_answer = check_completion(prediction)
        if is_complete:
            break
        if attempt < max_retries:
            print(
                f"[{idx}/{total}] incomplete output for {row['id']} "
                f"(has_answer_tag={answer_tag}, pred_answer={pred_answer}); retrying"
            )
            time.sleep(retry_sleep_seconds * (2**attempt))

    output_tokens = approximate_token_count(prediction)
    reasoning_tokens = count_reasoning_tokens_approx(prediction, output_tokens)
    is_correct = is_complete and answers_match(pred_answer, row["gold_answer"])
    return (
        {
            "id": row["id"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "prompt": prompt,
            "prediction": prediction,
            "pred_answer": pred_answer,
            "is_correct": is_correct,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "has_answer_tag": answer_tag,
            "num_retries": num_retries,
            "is_complete": is_complete,
            "error": error,
        },
        False,
    )


def print_result_log(result: dict, idx: int, total: int) -> None:
    print(
        f"[{idx}/{total}] {result['id']} "
        f"output_tokens={result['output_tokens']} "
        f"has_answer_tag={result['has_answer_tag']} "
        f"pred_answer={result['pred_answer']} "
        f"gold_answer={result['gold_answer']} "
        f"correct={result['is_correct']}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]

    backend = model_cfg.get("backend", "hf_api")
    model_name = model_cfg["name"]
    input_file = paths_cfg["input_file"]
    output_file = paths_cfg["generation_output"]

    print(f"Backend: {backend}")
    print(f"Model: {model_name}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    records = read_jsonl(input_file)
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("data", {}).get("max_samples")
    if max_samples is not None:
        records = records[:max_samples]

    hf_token = os.environ.get("HF_TOKEN")
    if backend == "hf_api":
        if not hf_token:
            raise RuntimeError("HF_TOKEN is required for hf_api backend. Please run: export HF_TOKEN=your_token")
    else:
        raise ValueError("Stage 1 baseline inference supports only backend: hf_api")

    existing_outputs = read_jsonl(output_file)
    completed_ids = {row.get("id") for row in existing_outputs if row.get("id")}
    if completed_ids:
        print(f"Found {len(completed_ids)} existing generations; completed ids will be skipped.")

    pending_records = [row for row in records if row.get("id") not in completed_ids]
    print(f"Total requested samples: {len(records)}")
    print(f"Pending samples: {len(pending_records)}")

    generation_cfg = cfg.get("generation", {})
    max_retries = int(generation_cfg.get("max_retries", 2))
    concurrency = max(1, int(generation_cfg.get("concurrency", 1)))
    retry_sleep_seconds = float(generation_cfg.get("retry_sleep_seconds", 2))
    max_consecutive_api_failures = int(generation_cfg.get("max_consecutive_api_failures", 20))
    print(f"Concurrency: {concurrency}")
    print(f"Max retries: {max_retries}")
    print(f"Max consecutive API failures before stopping: {max_consecutive_api_failures}")

    index_by_id = {row["id"]: idx for idx, row in enumerate(records, start=1)}
    pending_iter = iter(pending_records)
    consecutive_api_failures = 0

    def submit_next(executor: ThreadPoolExecutor, futures: dict) -> bool:
        try:
            row = next(pending_iter)
        except StopIteration:
            return False
        idx = index_by_id[row["id"]]
        print(f"[{idx}/{len(records)}] Running sample {row['id']}")
        futures[
            executor.submit(
                generate_one,
                row,
                idx,
                len(records),
                model_name,
                model_cfg,
                hf_token,
                max_retries,
                retry_sleep_seconds,
            )
        ] = (row, idx)
        return True

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for _ in range(concurrency):
            if not submit_next(executor, futures):
                break

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                row, idx = futures.pop(future)
                try:
                    result, api_failed = future.result()
                except Exception as exc:
                    print(f"Unexpected failure. model={model_name} sample_id={row['id']} error={exc}")
                    result, api_failed = None, True

                if api_failed:
                    consecutive_api_failures += 1
                    if consecutive_api_failures >= max_consecutive_api_failures:
                        print(
                            f"Stopping early after {consecutive_api_failures} consecutive API failures. "
                            "Already-written samples are preserved; rerun later to resume."
                        )
                        for pending_future in futures:
                            pending_future.cancel()
                        futures.clear()
                        break
                else:
                    consecutive_api_failures = 0

                if result is not None:
                    append_jsonl(output_file, result)
                    completed_ids.add(result["id"])
                    print_result_log(result, idx, len(records))

                if consecutive_api_failures < max_consecutive_api_failures:
                    submit_next(executor, futures)

    print(f"Saved/resumed generations in {output_file}")


if __name__ == "__main__":
    main()
