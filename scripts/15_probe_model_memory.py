#!/usr/bin/env python
"""Load a model or LoRA adapter once and report CUDA memory usage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.stage2_load import load_model_and_tokenizer
from src.training.trl_dpo_utils import load_stage2_config
from src.utils.io import ensure_parent_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_lora_dpo.yaml")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve(path: str, model_dir: str | None) -> tuple[str, dict]:
    raw = load_yaml(path)
    if "experiment" in raw:
        cfg = load_stage2_config(path)
        return model_dir or cfg["model"]["base_model_name_or_path"], cfg["model"]
    model_cfg = raw["model"]
    return model_dir or model_cfg.get("model_dir") or model_cfg.get("base_model_name_or_path"), model_cfg


def cuda_report() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(device),
        "total_gb": round(props.total_memory / 1024**3, 2),
        "allocated_gb": round(torch.cuda.memory_allocated(device) / 1024**3, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(device) / 1024**3, 2),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1024**3, 2),
    }


def main() -> None:
    args = parse_args()
    model_path, model_cfg = resolve(args.config, args.model_dir)
    print(f"Model path: {model_path}")
    print(f"Base model: {model_cfg.get('base_model_name_or_path', model_cfg.get('base_model', '(same as model path)'))}")
    if args.dry_run:
        print("Dry run complete. Model was not loaded.")
        return

    import torch

    model, tokenizer = load_model_and_tokenizer(model_path, model_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    report = cuda_report()
    report["model_path"] = str(model_path)
    report["vocab_size"] = len(tokenizer)
    print(json.dumps(report, indent=2))
    if args.output:
        ensure_parent_dir(args.output)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
