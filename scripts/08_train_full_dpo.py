#!/usr/bin/env python
"""Train the Stage 2 full-parameter DPO model with TRL.

This route is GPU-heavy. It is designed for distributed training with FSDP,
gradient checkpointing, gradient accumulation, and reference log-prob
precomputation so the reference model does not remain resident during training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trl_dpo_utils import (
    common_dpo_config_kwargs,
    fail_with_dependency_hint,
    filter_config_kwargs,
    gpu_memory_log_path,
    load_preference_datasets,
    load_stage2_config,
    load_tokenizer,
    model_init_kwargs,
    prepare_output_dirs,
    print_training_banner,
    require_training_files,
)
from src.training.callbacks import GpuMemoryCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_full_dpo.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Load config/data only; do not load the model or train.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_stage2_config(args.config)

    if cfg["experiment"].get("method") != "full_dpo":
        raise ValueError("This script expects experiment.method: full_dpo")

    require_training_files(cfg)
    prepare_output_dirs(cfg)
    print_training_banner(cfg, "full-parameter")

    dataset = load_preference_datasets(cfg)
    print(f"Train rows: {len(dataset['train'])}")
    print(f"Validation rows: {len(dataset['validation'])}")
    print(f"FSDP: {cfg['training'].get('fsdp')}")
    print(f"Gradient checkpointing: {cfg['training'].get('gradient_checkpointing')}")
    print(f"Gradient accumulation steps: {cfg['training'].get('gradient_accumulation_steps')}")
    print(f"Precompute reference log-probs: {cfg['dpo'].get('precompute_ref_log_probs')}")

    if args.dry_run:
        print("Dry run complete. Model was not loaded and training was not started.")
        return

    try:
        from transformers import AutoModelForCausalLM
        from trl import DPOConfig, DPOTrainer
    except Exception as exc:  # pragma: no cover - depends on training env
        fail_with_dependency_hint(exc)

    tokenizer = load_tokenizer(cfg)
    init_kwargs = model_init_kwargs(cfg)
    dpo_kwargs = common_dpo_config_kwargs(cfg)
    dpo_kwargs["fsdp"] = cfg["training"].get("fsdp")
    dpo_kwargs["fsdp_config"] = cfg["training"].get("fsdp_config")
    dpo_kwargs["activation_offloading"] = bool(cfg["dpo"].get("activation_offloading", False))
    dpo_args = DPOConfig(**filter_config_kwargs(DPOConfig, dpo_kwargs))

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model_name_or_path"],
        **init_kwargs,
    )
    if cfg["training"].get("gradient_checkpointing", True):
        model.config.use_cache = False

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=[GpuMemoryCallback(gpu_memory_log_path(cfg))],
    )
    trainer.train()
    trainer.save_model(cfg["paths"]["output_dir"])
    tokenizer.save_pretrained(cfg["paths"]["output_dir"])


if __name__ == "__main__":
    main()
