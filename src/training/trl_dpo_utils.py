"""Utilities shared by Stage 2 TRL DPO training scripts."""

from __future__ import annotations

import os
import sys
from dataclasses import fields, is_dataclass
from inspect import signature
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.utils.io import load_yaml


def load_stage2_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    for section in ["experiment", "model", "data", "dpo", "training", "paths"]:
        if section not in cfg:
            raise ValueError(f"Missing required config section: {section}")
    return cfg


def require_training_files(cfg: dict[str, Any]) -> None:
    data_cfg = cfg["data"]
    missing = [
        path
        for path in [data_cfg["train_file"], data_cfg["val_file"]]
        if not Path(path).exists()
    ]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Stage 2 preference files are missing. Generate or copy them first:\n"
            f"{joined}"
        )


def prepare_output_dirs(cfg: dict[str, Any]) -> None:
    paths = cfg["paths"]
    for key in ["output_dir", "logging_dir", "metrics_dir", "eval_generation_dir"]:
        if paths.get(key):
            Path(paths[key]).mkdir(parents=True, exist_ok=True)
    if paths.get("merged_output_dir"):
        Path(paths["merged_output_dir"]).mkdir(parents=True, exist_ok=True)


def load_preference_datasets(cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    fields = [
        data_cfg.get("prompt_field", "prompt"),
        data_cfg.get("chosen_field", "chosen"),
        data_cfg.get("rejected_field", "rejected"),
    ]

    dataset = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_file"],
            "validation": data_cfg["val_file"],
        },
    )

    def normalize(row: dict[str, Any]) -> dict[str, str]:
        return {
            "prompt": str(row[fields[0]]),
            "chosen": str(row[fields[1]]),
            "rejected": str(row[fields[2]]),
        }

    remove_columns = dataset["train"].column_names
    return dataset.map(normalize, remove_columns=remove_columns)


def dtype_from_config(dtype_name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from exc


def model_init_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = cfg["model"]
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "torch_dtype": dtype_from_config(model_cfg.get("torch_dtype", "bfloat16")),
    }
    attn_impl = model_cfg.get("attn_implementation")
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    return kwargs


def load_tokenizer(cfg: dict[str, Any]):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model_name_or_path"],
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def config_value(cfg: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    return cfg.get(section, {}).get(key, default)


def common_dpo_config_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    training = cfg["training"]
    dpo = cfg["dpo"]
    data = cfg["data"]
    paths = cfg["paths"]
    hub = cfg.get("hub", {})

    return {
        "output_dir": paths["output_dir"],
        "logging_dir": paths["logging_dir"],
        "run_name": cfg["experiment"]["name"],
        "seed": int(cfg["experiment"].get("seed", 42)),
        "num_train_epochs": float(training.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training.get("learning_rate", 5e-6)),
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
        "save_total_limit": int(training.get("save_total_limit", 2)),
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "report_to": training.get("report_to", ["tensorboard"]),
        "push_to_hub": bool(hub.get("push_to_hub", False)),
        "hub_model_id": hub.get("repo_id"),
        "beta": float(dpo.get("beta", 0.1)),
        "loss_type": dpo.get("loss_type", "sigmoid"),
        "max_prompt_length": int(data.get("max_prompt_length", 512)),
        "max_length": int(data.get("max_length", 2048)),
        "precompute_ref_log_probs": bool(dpo.get("precompute_ref_log_probs", False)),
        "precompute_ref_batch_size": dpo.get("precompute_ref_batch_size"),
        "model_init_kwargs": model_init_kwargs(cfg),
        "trust_remote_code": bool(cfg["model"].get("trust_remote_code", True)),
    }


def filter_config_kwargs(config_cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by the installed TRL DPOConfig version."""
    if is_dataclass(config_cls):
        accepted = {field.name for field in fields(config_cls)}
    else:
        accepted = set(signature(config_cls).parameters)

    filtered = {key: value for key, value in kwargs.items() if key in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(
            "Warning: installed TRL DPOConfig does not accept these config keys; "
            f"ignoring them: {', '.join(dropped)}"
        )

    # Older Transformers versions used evaluation_strategy instead of
    # eval_strategy. Handle that rename without requiring a repo-wide pin.
    if "eval_strategy" in kwargs and "eval_strategy" not in accepted and "evaluation_strategy" in accepted:
        filtered["evaluation_strategy"] = kwargs["eval_strategy"]
    return filtered


def print_training_banner(cfg: dict[str, Any], route: str) -> None:
    print("=" * 72)
    print(f"Stage 2 {route} DPO training")
    print("=" * 72)
    print(f"Experiment: {cfg['experiment']['name']}")
    print(f"Base model: {cfg['model']['base_model_name_or_path']}")
    print(f"Train file: {cfg['data']['train_file']}")
    print(f"Val file: {cfg['data']['val_file']}")
    print(f"Output dir: {cfg['paths']['output_dir']}")
    print(f"Logging dir: {cfg['paths']['logging_dir']}")
    print(f"HF_HOME: {os.environ.get('HF_HOME', '(default)')}")
    print("=" * 72)


def fail_with_dependency_hint(exc: Exception) -> None:
    print("Missing or incompatible Stage 2 training dependency.", file=sys.stderr)
    print("Install dependencies with:", file=sys.stderr)
    print("  uv pip install -r requirements.txt", file=sys.stderr)
    print(f"Original error: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc
