#!/usr/bin/env python
"""Estimate rough Stage 2 GPU memory needs before renting/using GPUs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trl_dpo_utils import load_stage2_config
from src.utils.io import ensure_parent_dir


MODEL_PARAM_ESTIMATES = {
    "Qwen/Qwen3-8B": 8.0e9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_lora_dpo.yaml")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--params_b", type=float, default=None, help="Override model parameter count in billions.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def dtype_bytes(dtype: str) -> int:
    return 2 if dtype.lower() in {"bf16", "bfloat16", "fp16", "float16"} else 4


def main() -> None:
    args = parse_args()
    cfg = load_stage2_config(args.config)
    method = cfg["experiment"]["method"]
    model_name = cfg["model"]["base_model_name_or_path"]
    params = (args.params_b * 1e9) if args.params_b else MODEL_PARAM_ESTIMATES.get(model_name, 8.0e9)
    bytes_per_param = dtype_bytes(cfg["model"].get("torch_dtype", "bfloat16"))
    weight_gb = params * bytes_per_param / 1024**3

    if method == "lora_dpo":
        trainable_gb = 0.15
        optimizer_gb = trainable_gb * 2
        gradient_gb = trainable_gb
        activation_gb = 4.0
        ref_overhead_gb = 0.0
        total_gb = weight_gb + optimizer_gb + gradient_gb + activation_gb + ref_overhead_gb
    else:
        # AdamW in mixed precision is still expensive. FSDP shards parameters,
        # grads, and optimizer states; activations remain workload-dependent.
        sharded_weight_gb = weight_gb / args.num_gpus
        optimizer_gb = (weight_gb * 2) / args.num_gpus
        gradient_gb = weight_gb / args.num_gpus
        activation_gb = 8.0
        ref_overhead_gb = 0.5 if cfg["dpo"].get("precompute_ref_log_probs", True) else weight_gb
        total_gb = sharded_weight_gb + optimizer_gb + gradient_gb + activation_gb + ref_overhead_gb

    result = {
        "experiment": cfg["experiment"]["name"],
        "method": method,
        "model": model_name,
        "num_gpus": args.num_gpus,
        "estimated_params_b": round(params / 1e9, 2),
        "weight_memory_gb_unsharded": round(weight_gb, 2),
        "estimated_memory_per_gpu_gb": round(total_gb, 2),
        "notes": [
            "This is a rough planning estimate, not a profiler result.",
            "Actual memory depends on sequence lengths, attention kernel, TRL version, FSDP behavior, and CUDA fragmentation.",
            "Start with per_device_train_batch_size=1 and gradient_accumulation_steps>=16.",
        ],
    }

    print(json.dumps(result, indent=2))
    if args.output:
        ensure_parent_dir(args.output)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
