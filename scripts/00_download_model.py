#!/usr/bin/env python
"""Download/cache the base model on a GPU machine without starting training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trl_dpo_utils import load_stage2_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_lora_dpo.yaml")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--local_dir", default=None, help="Optional explicit download directory outside the repo.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_stage2_config(args.config)
    model_name = args.model_name_or_path or cfg["model"]["base_model_name_or_path"]

    print(f"Model: {model_name}")
    print(f"HF_HOME: {os.environ.get('HF_HOME', '(default)')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', '(default)')}")
    print(f"Local dir: {args.local_dir or '(HF cache)'}")

    if args.dry_run:
        print("Dry run complete. No files were downloaded.")
        return

    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN is not set. Public models may still download; gated models will fail.")

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=model_name,
        local_dir=args.local_dir,
        token=os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
    )
    print(f"Downloaded/cached model at: {path}")


if __name__ == "__main__":
    main()
