#!/usr/bin/env python
"""Evaluate all saved checkpoints for one training config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trl_dpo_utils import load_stage2_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--eval_file",
        default="data/preference/stage1_gsm8k_qwen3_rejected_glm52_chosen_val.jsonl",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--include_final", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    if path.name.startswith("checkpoint-"):
        try:
            return (int(path.name.split("-", 1)[1]), path.name)
        except ValueError:
            pass
    return (10**12, path.name)


def find_model_dirs(output_dir: Path, include_final: bool) -> list[Path]:
    checkpoints = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=checkpoint_sort_key,
    )
    model_dirs = checkpoints
    if include_final and ((output_dir / "adapter_config.json").exists() or (output_dir / "config.json").exists()):
        model_dirs.append(output_dir)
    return model_dirs


def main() -> None:
    args = parse_args()
    cfg = load_stage2_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"]) / "checkpoint_evals"
    model_dirs = find_model_dirs(output_dir, include_final=args.include_final)

    print(f"Config: {args.config}")
    print(f"Output dir: {output_dir}")
    print(f"Eval file: {args.eval_file}")
    print(f"Found model dirs: {len(model_dirs)}")
    for model_dir in model_dirs:
        print(f"  - {model_dir}")

    if args.dry_run:
        print("Dry run complete. No evaluations were started.")
        return
    if not model_dirs:
        raise FileNotFoundError(f"No checkpoint directories found under {output_dir}")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    for model_dir in model_dirs:
        suffix = model_dir.name if model_dir != output_dir else "final"
        output_file = metrics_dir / f"{suffix}_eval.json"
        cmd = [
            sys.executable,
            "scripts/10_eval_stage2_model.py",
            "--config",
            args.config,
            "--model_dir",
            str(model_dir),
            "--eval_file",
            args.eval_file,
            "--output_file",
            str(output_file),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--batch_size",
            str(args.batch_size),
        ]
        if args.max_samples is not None:
            cmd.extend(["--max_samples", str(args.max_samples)])
        print("\n" + "=" * 80)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
