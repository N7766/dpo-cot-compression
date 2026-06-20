#!/usr/bin/env bash
set -euo pipefail

# Quick setup helper for a fresh rented GPU box.
# Usage:
#   bash scripts/remote_gpu_quickstart.sh

echo "== System =="
uname -a || true
python --version || true
nvidia-smi || true

echo "== Python env =="
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it first if your provider image does not include it:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

echo "== Recommended cache vars =="
echo "export HF_HOME=${HF_HOME:-/workspace/hf_cache}"
echo "export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/workspace/hf_cache/transformers}"

echo "== Stage 2 dry runs =="
python scripts/07_train_lora_dpo.py --config configs/stage2_lora_dpo.yaml --dry_run
python scripts/08_train_full_dpo.py --config configs/stage2_full_dpo.yaml --dry_run

echo "== Memory estimates =="
python scripts/12_estimate_memory.py --config configs/stage2_lora_dpo.yaml --num_gpus 1
python scripts/12_estimate_memory.py --config configs/stage2_full_dpo.yaml --num_gpus "${NUM_GPUS:-4}"

echo "Quickstart checks complete."
