#!/usr/bin/env python
"""Plot loss and GPU memory curves from a Stage 2 training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_state", required=True, help="Path to trainer_state.json inside a checkpoint/output dir.")
    parser.add_argument("--gpu_memory", default=None, help="Path to *_gpu_memory.jsonl.")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_trainer_logs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("log_history", [])


def read_jsonl(path: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    logs = read_trainer_logs(Path(args.trainer_state))
    train = [(row["step"], row["loss"]) for row in logs if "loss" in row and "step" in row]
    evals = [(row["step"], row["eval_loss"]) for row in logs if "eval_loss" in row and "step" in row]

    if train or evals:
        plt.figure(figsize=(8, 4.5))
        if train:
            plt.plot([x for x, _ in train], [y for _, y in train], label="train loss")
        if evals:
            plt.plot([x for x, _ in evals], [y for _, y in evals], label="eval loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = output_dir / "loss_curve.png"
        plt.savefig(out, dpi=160)
        print(f"Saved {out}")

    memory_rows = read_jsonl(Path(args.gpu_memory)) if args.gpu_memory else []
    if memory_rows:
        plt.figure(figsize=(8, 4.5))
        plt.plot([r["step"] for r in memory_rows], [r["allocated_gb"] for r in memory_rows], label="allocated GB")
        plt.plot([r["step"] for r in memory_rows], [r["reserved_gb"] for r in memory_rows], label="reserved GB")
        plt.xlabel("step")
        plt.ylabel("GPU memory (GB)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = output_dir / "gpu_memory_curve.png"
        plt.savefig(out, dpi=160)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
