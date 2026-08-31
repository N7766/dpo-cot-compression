#!/usr/bin/env python
"""Run a fixed, resume-safe final comparison for base and adapted Qwen models.

This script deliberately keeps prompt wording, questions, decoding and token
accounting fixed across models. It is intended for a post-selection GSM8K test
evaluation, not for choosing another training checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.answer_extraction import answers_match, extract_final_answer
from src.models.stage2_load import load_model_and_tokenizer
from src.utils.io import append_jsonl, ensure_parent_dir, load_yaml, read_jsonl, write_json


ANSWER_TAG_RE = re.compile(r"answer\s*:", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/final_controlled_eval.yaml")
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None, help="Optional model IDs from the config.")
    parser.add_argument("--prompt_modes", nargs="+", default=None, help="Optional prompt IDs from the config.")
    parser.add_argument("--overwrite", action="store_true", help="Discard resume data and overwrite selected output files.")
    parser.add_argument("--dry_run", action="store_true", help="Validate configuration without loading any model weights.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_rows(rows: list[dict[str, Any]], skip_first_n: int, max_samples: int | None) -> list[dict[str, Any]]:
    selected = rows[skip_first_n:]
    if max_samples is not None:
        selected = selected[:max_samples]
    missing = [row.get("id") for row in selected if not row.get("question") or row.get("gold_answer") is None]
    if missing:
        raise ValueError(f"Selected evaluation data has {len(missing)} rows missing question or gold_answer.")
    ids = [str(row["id"]) for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Selected evaluation data contains duplicate ids.")
    return selected


def select_by_id(items: list[dict[str, Any]], selected_ids: set[str], label: str) -> list[dict[str, Any]]:
    selected = [item for item in items if item.get("id") in selected_ids]
    if not selected:
        raise ValueError(f"No {label} selected. Check --{label} values against the config.")
    return selected


def prompt_text(template: str, question: str) -> str:
    try:
        return template.format(question=question)
    except KeyError as exc:
        raise ValueError("Each prompt template may only use the {question} placeholder.") from exc


def output_paths(output_dir: Path, prompt_id: str, model_id: str) -> tuple[Path, Path]:
    run_dir = output_dir / prompt_id
    return run_dir / f"{model_id}.jsonl", run_dir / f"{model_id}.summary.json"


def completion_token_count(token_ids: Any, tokenizer: Any) -> int:
    """Count visible completion tokens, excluding padding and EOS tokens."""
    stop_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    count = 0
    for token_id in token_ids.tolist():
        if token_id in stop_ids:
            break
        count += 1
    return count


def existing_ids(path: Path) -> set[str]:
    return {str(row["id"]) for row in read_jsonl(path) if row.get("id") is not None}


def summary_for_records(
    records: list[dict[str, Any]],
    *,
    experiment: str,
    model_spec: dict[str, Any],
    prompt_spec: dict[str, Any],
    input_file: Path,
    selected_count: int,
    generation_cfg: dict[str, Any],
) -> dict[str, Any]:
    correct = [record for record in records if record.get("is_correct") is True]
    token_counts = [int(record.get("output_tokens", 0)) for record in records]
    word_counts = [int(record.get("output_words", 0)) for record in records]
    return {
        "experiment": experiment,
        "model_id": model_spec["id"],
        "model_kind": model_spec["kind"],
        "model_source": model_spec["source"],
        "model_revision": model_spec.get("revision"),
        "prompt_id": prompt_spec["id"],
        "prompt_template": prompt_spec["template"],
        "input_file": str(input_file),
        "input_file_sha256": sha256_file(input_file),
        "selected_row_count": selected_count,
        "completed_row_count": len(records),
        "correct": len(correct),
        "accuracy": len(correct) / len(records) if records else 0.0,
        "avg_output_tokens": mean(token_counts) if token_counts else 0.0,
        "avg_output_words": mean(word_counts) if word_counts else 0.0,
        "answer_tag_rate": sum(bool(record.get("has_answer_tag")) for record in records) / len(records) if records else 0.0,
        "complete_rate": sum(bool(record.get("is_complete")) for record in records) / len(records) if records else 0.0,
        "max_new_tokens": int(generation_cfg["max_new_tokens"]),
        "temperature": float(generation_cfg["temperature"]),
        "top_p": float(generation_cfg["top_p"]),
        "do_sample": False,
        "cap_hit_count": sum(bool(record.get("hit_max_new_tokens")) for record in records),
        "token_count_definition": "Tokenizer completion tokens excluding padding and EOS.",
        "word_count_definition": "Whitespace-separated visible output units; retained only for comparison with older API artifacts.",
    }


def load_one_model(model_spec: dict[str, Any], defaults: dict[str, Any]):
    model_cfg = dict(defaults)
    model_cfg["base_model_name_or_path"] = model_spec.get(
        "base_model_name_or_path", defaults["base_model_name_or_path"]
    )
    if model_spec["kind"] == "adapter":
        model_cfg["adapter_name_or_path"] = model_spec["source"]
        model_cfg["base_model_revision"] = defaults.get("revision")
        model_cfg["adapter_revision"] = model_spec.get("revision")
    elif model_spec["kind"] != "base":
        raise ValueError(f"Unsupported model kind: {model_spec['kind']}")
    else:
        model_cfg["revision"] = model_spec.get("revision", defaults.get("revision"))
    return load_model_and_tokenizer(model_spec["source"], model_cfg)


def run_model_prompt(
    *,
    rows: list[dict[str, Any]],
    model_spec: dict[str, Any],
    prompt_spec: dict[str, Any],
    defaults: dict[str, Any],
    generation_cfg: dict[str, Any],
    output_file: Path,
    summary_file: Path,
    input_file: Path,
    experiment: str,
    overwrite: bool,
) -> None:
    if overwrite and output_file.exists():
        output_file.unlink()
    completed_ids = existing_ids(output_file)
    pending = [row for row in rows if str(row["id"]) not in completed_ids]
    print(f"\n[{prompt_spec['id']} | {model_spec['id']}] completed={len(completed_ids)} pending={len(pending)}")

    if pending:
        import torch

        model, tokenizer = load_one_model(model_spec, defaults)
        tokenizer.padding_side = "left"
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            print("WARNING: CUDA is unavailable; final evaluation will be very slow on CPU.")
        model.to(device)

        batch_size = int(generation_cfg["batch_size"])
        for start in tqdm(range(0, len(pending), batch_size), desc=f"{prompt_spec['id']}:{model_spec['id']}"):
            batch_rows = pending[start : start + batch_size]
            prompts = [prompt_text(prompt_spec["template"], row["question"]) for row in batch_rows]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            prompt_width = encoded["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=int(generation_cfg["max_new_tokens"]),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completions = generated[:, prompt_width:]
            predictions = tokenizer.batch_decode(completions, skip_special_tokens=True)
            for row, prompt, completion_ids, prediction in zip(batch_rows, prompts, completions, predictions):
                output_tokens = completion_token_count(completion_ids, tokenizer)
                pred_answer = extract_final_answer(prediction)
                has_answer_tag = bool(ANSWER_TAG_RE.search(prediction))
                record = {
                    "id": row["id"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "model_id": model_spec["id"],
                    "model_source": model_spec["source"],
                    "model_revision": model_spec.get("revision"),
                    "prompt_id": prompt_spec["id"],
                    "prompt": prompt,
                    "prediction": prediction,
                    "pred_answer": pred_answer,
                    "has_answer_tag": has_answer_tag,
                    "is_complete": has_answer_tag and pred_answer is not None,
                    "is_correct": answers_match(pred_answer, row["gold_answer"]),
                    "output_tokens": output_tokens,
                    "output_words": len(prediction.split()),
                    "hit_max_new_tokens": output_tokens >= int(generation_cfg["max_new_tokens"]),
                }
                append_jsonl(output_file, record)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_ids = {str(item["id"]) for item in rows}
    records = [row for row in read_jsonl(output_file) if str(row.get("id")) in selected_ids]
    record_ids = [str(row["id"]) for row in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"Duplicate ids found in controlled-evaluation output: {output_file}")
    if set(record_ids) != selected_ids:
        missing = len(selected_ids - set(record_ids))
        raise RuntimeError(f"Controlled evaluation is incomplete for {output_file}: {missing} selected ids are missing.")
    records.sort(key=lambda item: str(item["id"]))
    summary = summary_for_records(
        records,
        experiment=experiment,
        model_spec=model_spec,
        prompt_spec=prompt_spec,
        input_file=input_file,
        selected_count=len(rows),
        generation_cfg=generation_cfg,
    )
    ensure_parent_dir(summary_file)
    write_json(summary_file, summary)
    print(
        f"[{prompt_spec['id']} | {model_spec['id']}] "
        f"accuracy={summary['accuracy']:.2%} "
        f"mean_completion_tokens={summary['avg_output_tokens']:.2f} "
        f"cap_hits={summary['cap_hit_count']}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    input_file = Path(args.input_file or data_cfg["input_file"])
    if not input_file.exists():
        raise FileNotFoundError(
            f"Evaluation input does not exist: {input_file}. "
            "Prepare it first with: python scripts/01_prepare_gsm8k.py --config configs/final_gsm8k_test.yaml"
        )

    rows = select_rows(
        read_jsonl(input_file),
        int(data_cfg.get("skip_first_n", 0)),
        args.max_samples if args.max_samples is not None else data_cfg.get("max_samples"),
    )
    model_specs = select_by_id(cfg["models"], set(args.models or [model["id"] for model in cfg["models"]]), "models")
    prompt_specs = select_by_id(cfg["prompts"], set(args.prompt_modes or [prompt["id"] for prompt in cfg["prompts"]]), "prompt_modes")
    output_dir = Path(args.output_dir or cfg["paths"]["output_dir"])
    generation_cfg = dict(cfg["generation"])
    if args.batch_size is not None:
        generation_cfg["batch_size"] = args.batch_size

    print(f"Experiment: {cfg['experiment']['name']}")
    print(f"Input file: {input_file}")
    print(f"Input SHA256: {sha256_file(input_file)}")
    print(f"Rows after excluding first {data_cfg.get('skip_first_n', 0)}: {len(rows)}")
    print("Models:", ", ".join(model["id"] for model in model_specs))
    print("Prompt modes:", ", ".join(prompt["id"] for prompt in prompt_specs))
    print(f"Decoding: temperature=0.0, max_new_tokens={generation_cfg['max_new_tokens']}, batch_size={generation_cfg['batch_size']}")

    if args.dry_run:
        print("Dry run complete. No model weights were loaded and no outputs were written.")
        return

    for prompt_spec in prompt_specs:
        for model_spec in model_specs:
            output_file, summary_file = output_paths(output_dir, prompt_spec["id"], model_spec["id"])
            run_model_prompt(
                rows=rows,
                model_spec=model_spec,
                prompt_spec=prompt_spec,
                defaults=cfg["model_defaults"],
                generation_cfg=generation_cfg,
                output_file=output_file,
                summary_file=summary_file,
                input_file=input_file,
                experiment=cfg["experiment"]["name"],
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
