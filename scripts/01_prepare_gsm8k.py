#!/usr/bin/env python
"""Download and preprocess GSM8K for baseline or preference-data generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gsm8k import load_gsm8k_split, process_gsm8k_examples
from src.utils.io import load_yaml, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_baseline.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]

    max_samples = args.max_samples if args.max_samples is not None else data_cfg.get("max_samples")
    dataset = load_gsm8k_split(
        dataset_name=data_cfg.get("dataset_name", "openai/gsm8k"),
        dataset_config=data_cfg.get("dataset_config", "main"),
        split=data_cfg.get("split", "test"),
    )
    output_path = cfg.get("paths", {}).get("input_file", data_cfg.get("processed_path"))
    records = process_gsm8k_examples(dataset, split=data_cfg.get("split", "test"), max_samples=max_samples)
    existing_records = read_jsonl(output_path)
    existing_by_id = {row.get("id"): row for row in existing_records}
    if existing_by_id:
        merged_records = []
        num_new = 0
        for record in records:
            existing = existing_by_id.get(record["id"])
            if existing is not None:
                merged_records.append(existing)
            else:
                merged_records.append(record)
                num_new += 1
        records = merged_records
        print(f"Found {len(existing_records)} existing processed examples; preserving them and adding {num_new} new examples.")

    write_jsonl(output_path, records)
    print(f"Saved {len(records)} GSM8K examples to {output_path}")


if __name__ == "__main__":
    main()
