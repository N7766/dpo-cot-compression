#!/usr/bin/env python
"""Train the Stage 3 LoRA SFT warm-up model with Transformers + PEFT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset

from src.training.callbacks import GpuMemoryCallback
from src.training.trl_dpo_utils import (
    dtype_from_config,
    fail_with_dependency_hint,
    filter_config_kwargs,
    gpu_memory_log_path,
    load_stage2_config,
    load_tokenizer,
    model_init_kwargs,
    prepare_output_dirs,
    require_training_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3_lora_sft.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Load config/data only; do not load the model or train.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps for smoke testing.")
    return parser.parse_args()


def build_peft_config(cfg: dict[str, Any]):
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


def load_sft_datasets(cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    dataset = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_file"],
            "validation": data_cfg["val_file"],
        },
    )
    return dataset


def tokenize_sft_dataset(dataset, tokenizer, cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    prompt_field = data_cfg.get("prompt_field", "prompt")
    response_field = data_cfg.get("response_field", "response")
    max_length = int(data_cfg.get("max_length", 1024))
    eos = tokenizer.eos_token or ""

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        prompt = str(row[prompt_field]).rstrip()
        response = str(row[response_field]).strip()
        full_text = f"{prompt}\n\n{response}{eos}"

        prompt_ids = tokenizer(prompt + "\n\n", add_special_tokens=True)["input_ids"]
        full = tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=max_length)
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if all(label == -100 for label in labels):
            labels[-1] = input_ids[-1]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    remove_columns = dataset["train"].column_names
    return dataset.map(tokenize, remove_columns=remove_columns)


class SftDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padded_labels.append(label + [-100] * (max_len - len(label)))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def training_args_kwargs(cfg: dict[str, Any], max_steps: int | None = None) -> dict[str, Any]:
    training = cfg["training"]
    paths = cfg["paths"]
    kwargs = {
        "output_dir": paths["output_dir"],
        "logging_dir": paths["logging_dir"],
        "run_name": cfg["experiment"]["name"],
        "seed": int(cfg["experiment"].get("seed", 42)),
        "num_train_epochs": float(training.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training.get("learning_rate", 1e-5)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.03)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "lr_scheduler_type": training.get("lr_scheduler_type", "cosine"),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": training.get("gradient_checkpointing_kwargs"),
        "bf16": bool(training.get("bf16", True)),
        "fp16": bool(training.get("fp16", False)),
        "optim": training.get("optim", "adamw_torch"),
        "logging_steps": int(training.get("logging_steps", 10)),
        "eval_steps": int(training.get("eval_steps", 100)),
        "save_steps": int(training.get("save_steps", 100)),
        "save_total_limit": int(training.get("save_total_limit", 3)),
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "report_to": training.get("report_to", []),
    }
    if max_steps is not None:
        kwargs["max_steps"] = int(max_steps)
    return kwargs


def print_banner(cfg: dict[str, Any]) -> None:
    print("=" * 72)
    print("Stage 3 LoRA SFT warm-up training")
    print("=" * 72)
    print(f"Experiment: {cfg['experiment']['name']}")
    print(f"Base model: {cfg['model']['base_model_name_or_path']}")
    print(f"Train file: {cfg['data']['train_file']}")
    print(f"Val file: {cfg['data']['val_file']}")
    print(f"Max length: {cfg['data'].get('max_length', 1024)}")
    print(f"Output dir: {cfg['paths']['output_dir']}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    cfg = load_stage2_config(args.config)
    if cfg["experiment"].get("method") != "lora_sft":
        raise ValueError("This script expects experiment.method: lora_sft")

    require_training_files(cfg)
    prepare_output_dirs(cfg)
    print_banner(cfg)

    dataset = load_sft_datasets(cfg)
    print(f"Train rows: {len(dataset['train'])}")
    print(f"Validation rows: {len(dataset['validation'])}")
    print(f"LoRA rank: {cfg['lora']['r']}")

    if args.dry_run:
        print("Dry run complete. Model was not loaded and training was not started.")
        return

    try:
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    except Exception as exc:  # pragma: no cover - depends on training env
        fail_with_dependency_hint(exc)

    tokenizer = load_tokenizer(cfg)
    tokenizer.padding_side = "right"
    tokenized = tokenize_sft_dataset(dataset, tokenizer, cfg)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model_name_or_path"],
        **model_init_kwargs(cfg),
    )
    if cfg["training"].get("gradient_checkpointing", True):
        model.config.use_cache = False
    model = get_peft_model(model, build_peft_config(cfg))
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        **filter_config_kwargs(TrainingArguments, training_args_kwargs(cfg, args.max_steps))
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=SftDataCollator(tokenizer),
        callbacks=[GpuMemoryCallback(gpu_memory_log_path(cfg))],
    )
    trainer.train()
    trainer.save_model(cfg["paths"]["output_dir"])
    tokenizer.save_pretrained(cfg["paths"]["output_dir"])


if __name__ == "__main__":
    main()
