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
    if model_cfg.get("revision"):
        kwargs["revision"] = model_cfg["revision"]
    return kwargs


def load_model_and_tokenizer(model_dir: str | Path, model_cfg: dict[str, Any]):
    """Load a full model or PEFT LoRA adapter from a local path or Hub id.

    ``adapter_name_or_path`` can be supplied explicitly for a remote adapter.
    Keeping this explicit avoids guessing whether a Hub repository is a full
    checkpoint or a PEFT adapter.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = str(model_dir)
    local_model_dir = Path(model_source)
    base_model = model_cfg.get("base_model_name_or_path") or model_cfg.get("base_model")
    explicit_adapter = model_cfg.get("adapter_name_or_path")
    kwargs = model_kwargs_from_cfg(model_cfg)

    is_local_adapter = local_model_dir.is_dir() and (local_model_dir / "adapter_config.json").exists()
    if explicit_adapter or is_local_adapter:
        if not base_model:
            raise ValueError("LoRA adapter loading requires model.base_model_name_or_path in config.")
        from peft import PeftModel

        base_revision = model_cfg.get("base_model_revision") or kwargs.get("revision")
        base_kwargs = dict(kwargs)
        if base_revision:
            base_kwargs["revision"] = base_revision
        else:
            base_kwargs.pop("revision", None)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=base_kwargs["trust_remote_code"],
            revision=base_kwargs.get("revision"),
        )
        model = AutoModelForCausalLM.from_pretrained(base_model, **base_kwargs)
        adapter_source = str(explicit_adapter or model_source)
        model = PeftModel.from_pretrained(
            model,
            adapter_source,
            revision=model_cfg.get("adapter_revision"),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs.get("revision"),
        )
        model = AutoModelForCausalLM.from_pretrained(model_source, **kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
