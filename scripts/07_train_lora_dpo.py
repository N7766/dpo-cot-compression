#!/usr/bin/env python
"""Train the Stage 2 LoRA+DPO model with TRL.

This is the first recommended Stage 2 route. It keeps base model weights frozen,
trains LoRA adapters, and relies on TRL's PEFT-aware DPOTrainer behavior for the
reference policy instead of keeping a separate full reference model in memory.
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
    parser.add_argument("--config", default="configs/stage2_lora_dpo.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Load config/data only; do not load the model or train.")
    return parser.parse_args()


def build_peft_config(cfg: dict):
    try:
        from peft import LoraConfig, TaskType
    except Exception as exc:  # pragma: no cover - depends on training env
        fail_with_dependency_hint(exc)

    lora = cfg["lora"]
    return LoraConfig(
        r=int(lora.get("r", 8)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora["target_modules"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def main() -> None:
    args = parse_args()
    cfg = load_stage2_config(args.config)

    if cfg["experiment"].get("method") != "lora_dpo":
        raise ValueError("This script expects experiment.method: lora_dpo")

    require_training_files(cfg)
    prepare_output_dirs(cfg)
    print_training_banner(cfg, "LoRA")

    dataset = load_preference_datasets(cfg)
    print(f"Train rows: {len(dataset['train'])}")
    print(f"Validation rows: {len(dataset['validation'])}")
    print(f"LoRA rank: {cfg['lora']['r']}")

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
    dpo_args = DPOConfig(**filter_config_kwargs(DPOConfig, dpo_kwargs))
    peft_config = build_peft_config(cfg)

    adapter_path = cfg["model"].get("adapter_name_or_path")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model_name_or_path"],
        **init_kwargs,
    )
    if adapter_path:
        from peft import PeftModel

        print(f"Loading trainable LoRA adapter for DPO warm-start: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    if cfg["training"].get("gradient_checkpointing", True):
        model.config.use_cache = False

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=None if adapter_path else peft_config,
        callbacks=[GpuMemoryCallback(gpu_memory_log_path(cfg))],
    )
    trainer.train()
    trainer.save_model(cfg["paths"]["output_dir"])
    tokenizer.save_pretrained(cfg["paths"]["output_dir"])


if __name__ == "__main__":
    main()
