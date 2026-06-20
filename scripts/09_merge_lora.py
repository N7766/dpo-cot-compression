#!/usr/bin/env python
"""Merge a trained LoRA adapter into the base model for inference/export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trl_dpo_utils import load_stage2_config, model_init_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_lora_dpo.yaml")
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_stage2_config(args.config)
    adapter_dir = Path(args.adapter_dir or cfg["paths"]["output_dir"])
    output_dir = Path(args.output_dir or cfg["paths"]["merged_output_dir"])
    base_model = cfg["model"]["base_model_name_or_path"]

    print(f"Base model: {base_model}")
    print(f"Adapter dir: {adapter_dir}")
    print(f"Merged output dir: {output_dir}")
    if args.dry_run:
        print("Dry run complete. Model was not loaded and merge was not started.")
        return
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_init_kwargs(cfg))
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)))
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged model to {output_dir}")


if __name__ == "__main__":
    main()
