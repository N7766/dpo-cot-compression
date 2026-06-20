# Stage 2 Plan: DPO Training Layout and Pipeline

Stage 2 will train models from the Stage 1 preference dataset. This document fixes the naming and directory conventions before training code, SSH setup, or GPU runs are added.

## 1. Training Variants

Two training variants are planned:

| Variant | Config | Purpose |
| --- | --- | --- |
| LoRA+DPO | `configs/stage2_lora_dpo.yaml` | First, cheaper training path for iteration and dissertation experiments |
| Full-parameter DPO | `configs/stage2_full_dpo.yaml` | Heavier comparison run if GPU budget allows |

The recommended first run is LoRA+DPO. Full DPO should be treated as a later compute-heavy experiment.

## 2. Input Dataset

Stage 2 consumes the filtered Stage 1 DPO files:

```text
data/preference/stage1_gsm8k_qwen3_rejected_glm52_chosen_train.jsonl
data/preference/stage1_gsm8k_qwen3_rejected_glm52_chosen_val.jsonl
```

Expected fields:

```text
prompt
chosen
rejected
gold_answer
chosen_answer
rejected_answer
chosen_tokens
rejected_tokens
compression_ratio
```

These JSONL files are generated artifacts and are ignored by git. On a remote GPU machine, they should either be regenerated with Stage 1 scripts or copied explicitly with `rsync/scp`.

## 3. Model Placement

Base model weights should not be stored in this repository.

Recommended remote layout:

```text
/workspace/dpo-cot-compression/       # git checkout
/workspace/hf_cache/                  # Hugging Face model cache
/workspace/artifacts/                 # optional external backup/export area
```

Recommended environment variables on the GPU machine:

```bash
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
export HF_TOKEN=xxxxx
```

The base model should be referenced by name unless a local path is intentionally used:

```text
Qwen/Qwen3-8B
```

If a local base model copy is used later, keep it outside the repository, for example:

```text
/workspace/models/Qwen3-8B
```

## 4. Output Layout

All generated Stage 2 outputs stay under `outputs/` and are ignored by git.

LoRA+DPO:

```text
outputs/checkpoints/stage2_lora_dpo/qwen3_8b_gsm8k_cot_compression_lora/
outputs/checkpoints/stage2_lora_dpo/qwen3_8b_gsm8k_cot_compression_lora_merged/
outputs/logs/stage2_lora_dpo/qwen3_8b_gsm8k_cot_compression_lora/
outputs/results/stage2_lora_dpo/
outputs/generations/stage2_eval/lora_dpo/
```

Full-parameter DPO:

```text
outputs/checkpoints/stage2_full_dpo/qwen3_8b_gsm8k_cot_compression_full/
outputs/logs/stage2_full_dpo/qwen3_8b_gsm8k_cot_compression_full/
outputs/results/stage2_full_dpo/
outputs/generations/stage2_eval/full_dpo/
```

Only `.gitkeep` placeholder files should be tracked inside these directories. Checkpoints, adapters, merged models, logs, metrics, and evaluation generations should remain untracked.

## 5. Naming Convention

Use this pattern for experiment names:

```text
<base_model>_<dataset>_<objective>_<method>
```

Current names:

```text
qwen3_8b_gsm8k_cot_compression_lora_dpo
qwen3_8b_gsm8k_cot_compression_full_dpo
```

Use this pattern for Hugging Face Hub model repos if uploading later:

```text
N7766/qwen3-8b-gsm8k-cot-compression-lora-dpo
N7766/qwen3-8b-gsm8k-cot-compression-full-dpo
```

Do not enable Hub upload in config until the model card and license notes are ready.

## 6. Planned Script Layout

Suggested Stage 2 scripts:

| Script | Purpose |
| --- | --- |
| `scripts/07_train_lora_dpo.py` | Train LoRA adapter with DPO |
| `scripts/08_train_full_dpo.py` | Train full model with DPO or FSDP/DeepSpeed |
| `scripts/09_merge_lora.py` | Merge LoRA adapter into base model for inference/export |
| `scripts/10_eval_stage2_model.py` | Evaluate trained model on held-out GSM8K |
| `scripts/11_upload_model.py` | Optional Hugging Face upload after review |

Stage 2 should begin with `07_train_lora_dpo.py`.

## 7. Planned Training Pipeline

LoRA+DPO first:

```text
Stage 1 DPO train/val JSONL
  -> load Qwen/Qwen3-8B
  -> attach LoRA adapters
  -> DPO training
  -> save adapter checkpoints
  -> evaluate validation loss
  -> optionally merge adapter
  -> held-out GSM8K evaluation
```

LoRA+DPO initial design:

```text
LoRA rank: r=8
LoRA alpha: 16
Reference policy: handled by TRL/PEFT without keeping a second full model copy
Gradient checkpointing: enabled
Gradient accumulation: enabled
```

Full DPO later:

```text
Stage 1 DPO train/val JSONL
  -> load Qwen/Qwen3-8B
  -> full-parameter DPO with FSDP/DeepSpeed
  -> save full checkpoints
  -> held-out GSM8K evaluation
  -> compare against LoRA+DPO
```

Full DPO memory plan:

```text
FSDP: full_shard auto_wrap
Gradient checkpointing: enabled
Gradient accumulation: enabled
Reference log-probs: precomputed before policy updates
Reference model during training: not kept resident after precomputation
```

## 8. Remote SSH Workflow

Initial remote setup should follow this shape:

```bash
git clone https://github.com/N7766/dpo-cot-compression.git
cd dpo-cot-compression

uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

export HF_HOME=/workspace/hf_cache
export HF_TOKEN=xxxxx
```

Copy local generated Stage 1 preference data if not regenerating remotely:

```bash
rsync -av data/preference/stage1_gsm8k_qwen3_rejected_glm52_chosen_train.jsonl <remote>:/workspace/dpo-cot-compression/data/preference/
rsync -av data/preference/stage1_gsm8k_qwen3_rejected_glm52_chosen_val.jsonl <remote>:/workspace/dpo-cot-compression/data/preference/
```

Then run LoRA+DPO training once the script exists:

```bash
python scripts/07_train_lora_dpo.py --config configs/stage2_lora_dpo.yaml
```

For a no-model dry run:

```bash
python scripts/07_train_lora_dpo.py --config configs/stage2_lora_dpo.yaml --dry_run
python scripts/08_train_full_dpo.py --config configs/stage2_full_dpo.yaml --dry_run
```

For full DPO, use a distributed launcher. Example:

```bash
torchrun --nproc_per_node=4 scripts/08_train_full_dpo.py --config configs/stage2_full_dpo.yaml
```

## 9. Evaluation Policy

Validation data:

- Use Stage 1 DPO validation split for training diagnostics.
- Track validation DPO loss and response length.

Final evaluation:

- Use held-out GSM8K test data.
- Report answer accuracy and average output length.
- Compare against the Stage 1 Qwen3-8B baseline.

The final dissertation comparison should include at least:

| Model | Accuracy | Avg output tokens | Compression vs baseline |
| --- | ---: | ---: | ---: |
| Qwen3-8B baseline | TBD | TBD | 0% |
| Qwen3-8B LoRA+DPO | TBD | TBD | TBD |
| Qwen3-8B full DPO | TBD | TBD | TBD |

## 10. Safety Notes

- Do not commit checkpoints, adapters, merged models, logs, or generated evaluation outputs.
- Do not commit `.env`, API keys, SSH keys, or Hugging Face tokens.
- Keep base model weights in Hugging Face cache or an external model directory, not in the repository.
- Prefer LoRA+DPO first to validate the pipeline before spending full DPO GPU time.
