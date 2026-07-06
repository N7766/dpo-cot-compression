#!/usr/bin/env python
"""Build Stage 3 SFT and stricter DPO datasets from Stage 1 preference data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.answer_extraction import answers_match, extract_final_answer
from src.utils.io import load_yaml, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3_build_data.yaml")
    return parser.parse_args()


def has_clean_answer_tag(text: str) -> bool:
    return extract_final_answer(text) is not None and "answer:" in text.lower()


def is_correct_pair(row: dict) -> bool:
    gold = row.get("gold_answer")
    return answers_match(row.get("chosen_answer"), gold) and answers_match(row.get("rejected_answer"), gold)


def sft_record(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "prompt": row.get("prompt"),
        "response": row.get("chosen"),
        "response_answer": row.get("chosen_answer"),
        "response_tokens": row.get("chosen_tokens"),
        "teacher_model": row.get("chosen_model"),
    }


def build_split(rows: list[dict], filters: dict) -> tuple[list[dict], list[dict], dict]:
    require_answer_tag = bool(filters.get("require_answer_tag", True))
    max_chosen_tokens = int(filters.get("max_chosen_tokens", 80))
    min_rejected_tokens = int(filters.get("min_rejected_tokens", 80))
    min_compression_ratio = float(filters.get("min_compression_ratio", 0.30))

    sft_rows = []
    strict_dpo_rows = []
    counts = {
        "input": len(rows),
        "sft_chosen_incorrect": 0,
        "sft_missing_answer_tag": 0,
        "sft_chosen_too_long": 0,
        "dpo_pair_incorrect": 0,
        "dpo_chosen_too_long": 0,
        "dpo_rejected_too_short": 0,
        "dpo_low_compression": 0,
    }

    for row in rows:
        chosen = str(row.get("chosen", ""))
        chosen_correct = answers_match(row.get("chosen_answer"), row.get("gold_answer"))
        chosen_has_answer = has_clean_answer_tag(chosen)
        chosen_tokens = int(row.get("chosen_tokens") or 0)

        if not chosen_correct:
            counts["sft_chosen_incorrect"] += 1
        elif require_answer_tag and not chosen_has_answer:
            counts["sft_missing_answer_tag"] += 1
        elif chosen_tokens > max_chosen_tokens:
            counts["sft_chosen_too_long"] += 1
        else:
            sft_rows.append(sft_record(row))

        pair_correct = is_correct_pair(row)
        rejected_tokens = int(row.get("rejected_tokens") or 0)
        compression_ratio = float(row.get("compression_ratio") or 0.0)
        if not pair_correct:
            counts["dpo_pair_incorrect"] += 1
        elif require_answer_tag and not chosen_has_answer:
            counts["sft_missing_answer_tag"] += 1
        elif chosen_tokens > max_chosen_tokens:
            counts["dpo_chosen_too_long"] += 1
        elif rejected_tokens < min_rejected_tokens:
            counts["dpo_rejected_too_short"] += 1
        elif compression_ratio < min_compression_ratio:
            counts["dpo_low_compression"] += 1
        else:
            strict_dpo_rows.append(row)

    return sft_rows, strict_dpo_rows, counts


def summarize_records(sft_rows: list[dict], dpo_rows: list[dict], counts: dict) -> dict:
    return {
        **counts,
        "sft_rows": len(sft_rows),
        "strict_dpo_rows": len(dpo_rows),
        "avg_sft_response_tokens": mean([int(r.get("response_tokens") or 0) for r in sft_rows]) if sft_rows else 0.0,
        "avg_dpo_chosen_tokens": mean([int(r.get("chosen_tokens") or 0) for r in dpo_rows]) if dpo_rows else 0.0,
        "avg_dpo_rejected_tokens": mean([int(r.get("rejected_tokens") or 0) for r in dpo_rows]) if dpo_rows else 0.0,
        "avg_dpo_compression_ratio": mean([float(r.get("compression_ratio") or 0.0) for r in dpo_rows]) if dpo_rows else 0.0,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    inputs = cfg["inputs"]
    outputs = cfg["outputs"]
    filters = cfg.get("filters", {})

    train_rows = read_jsonl(inputs["dpo_train_file"])
    val_rows = read_jsonl(inputs["dpo_val_file"])
    train_sft, train_dpo, train_counts = build_split(train_rows, filters)
    val_sft, val_dpo, val_counts = build_split(val_rows, filters)

    write_jsonl(outputs["sft_train_file"], train_sft)
    write_jsonl(outputs["sft_val_file"], val_sft)
    write_jsonl(outputs["dpo_train_file"], train_dpo)
    write_jsonl(outputs["dpo_val_file"], val_dpo)

    summary = {
        "filters": filters,
        "train": summarize_records(train_sft, train_dpo, train_counts),
        "validation": summarize_records(val_sft, val_dpo, val_counts),
        "outputs": outputs,
    }
    write_json(outputs["summary_file"], summary)

    print("Stage 3 data build summary")
    print("=" * 40)
    for split in ["train", "validation"]:
        s = summary[split]
        print(
            f"{split}: input={s['input']} sft={s['sft_rows']} strict_dpo={s['strict_dpo_rows']} "
            f"avg_sft_tokens={s['avg_sft_response_tokens']:.2f} "
            f"avg_dpo_compression={s['avg_dpo_compression_ratio']:.4f}"
        )
    print(f"Summary file: {outputs['summary_file']}")


if __name__ == "__main__":
    main()
