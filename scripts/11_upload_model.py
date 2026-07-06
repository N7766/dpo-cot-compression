#!/usr/bin/env python
"""Upload a trained Stage 2 model or adapter directory to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trl_dpo_utils import load_stage2_config
from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_upload.yaml")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit_message", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_yaml(args.config)
    if "experiment" in raw:
        cfg = load_stage2_config(args.config)
        model_dir = Path(args.model_dir or cfg["paths"]["output_dir"])
        hub_cfg = cfg.get("hub", {})
    else:
        model_dir = Path(args.model_dir or raw["model"]["model_dir"])
        hub_cfg = raw.get("hub", {})
    repo_id = args.repo_id or hub_cfg.get("repo_id")
    private = args.private or bool(hub_cfg.get("private", False))
    commit_message = args.commit_message or hub_cfg.get("commit_message", "Upload Stage 2 model")
    ignore_patterns = hub_cfg.get("ignore_patterns", [])

    if not repo_id:
        raise ValueError("Set --repo_id or hub.repo_id in the config.")
    print(f"Model dir: {model_dir}")
    print(f"Repo id: {repo_id}")
    print(f"Private: {private}")
    if ignore_patterns:
        print("Ignore patterns:")
        for pattern in ignore_patterns:
            print(f"  - {pattern}")

    if args.dry_run:
        print("Dry run complete. No files were uploaded.")
        return
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required. Please run: export HF_TOKEN=your_token")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
    )
    print(f"Uploaded {model_dir} to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
