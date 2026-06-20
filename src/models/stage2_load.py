"""Loading helpers for Stage 2 trained models and LoRA adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.training.trl_dpo_utils import dtype_from_config


def model_kwargs_from_cfg(model_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "torch_dtype": dtype_from_config(model_cfg.get("torch_dtype", "bfloat16")),
    }
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    return kwargs


def load_model_and_tokenizer(model_dir: str | Path, model_cfg: dict[str, Any]):
    """Load either a full model directory or a PEFT LoRA adapter directory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(model_dir)
    base_model = model_cfg.get("base_model_name_or_path") or model_cfg.get("base_model")
    kwargs = model_kwargs_from_cfg(model_cfg)

    if (model_dir / "adapter_config.json").exists():
        if not base_model:
            raise ValueError("LoRA adapter loading requires model.base_model_name_or_path in config.")
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=kwargs["trust_remote_code"])
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        model = PeftModel.from_pretrained(model, str(model_dir))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=kwargs["trust_remote_code"])
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
