# Stage 5 Experiment Plan: Conservative DPO Sweep

Stage 5 tests whether DPO can improve reasoning compression without damaging the strong free-form accuracy obtained by Stage 3 LoRA SFT.

## Current Baselines

| Stage | Method | Accuracy | Avg output tokens |
| --- | --- | ---: | ---: |
| Stage 3 | LoRA SFT | 95.02% | 32.42 |
| Stage 4 | Full DPO FSDP | 33.12% | 79.68 |

Stage 5 success should be judged by free-form validation generation, not by DPO loss alone.

Target:

```text
accuracy >= 93%
avg_output_tokens < 32.42
```

## Main Route

Prioritize LoRA DPO initialized from the Stage 3 SFT adapter:

```text
Qwen/Qwen3-8B
  -> load Stage 3 LoRA SFT adapter as train adapter
  -> load the same Stage 3 adapter as frozen reference adapter
  -> conservative LoRA DPO
  -> evaluate every saved checkpoint
```

This is different from Stage 2. Stage 2 initialized a fresh LoRA adapter and failed. Stage 5 continues from the successful Stage 3 SFT adapter.

## Configs

Priority order:

| Priority | Config | Purpose |
| ---: | --- | --- |
| 1 | `configs/stage5/stage5_lora_dpo_sft_lr1e-6_beta003_steps50.yaml` | Short aggressive-but-small LoRA DPO |
| 2 | `configs/stage5/stage5_lora_dpo_sft_lr5e-7_beta003_steps100.yaml` | Main conservative LoRA DPO |
| 3 | `configs/stage5/stage5_lora_dpo_sft_lr5e-7_beta005_steps100.yaml` | Slightly stronger beta |
| 4 | `configs/stage5/stage5_lora_dpo_sft_lr1e-7_beta01_steps100.yaml` | Very low LR, higher beta control |
| 5 | `configs/stage5/stage5_full_dpo_base_lr1e-7_beta003_steps50.yaml` | Conservative full DPO check |
| 6 | `configs/stage5/stage5_full_dpo_base_lr5e-8_beta003_steps100.yaml` | Lower LR full DPO check |

Run the LoRA configs first. Full DPO is optional and should only be run if H200 time is available.

## Remote Setup

After connecting to the GPU server:

```bash
cd /workspace
git clone https://github.com/N7766/dpo-cot-compression.git
cd dpo-cot-compression
source /venv/main/bin/activate
uv pip install -r requirements.txt
```

Sync local preference data if it is not already on the server:

```bash
rsync -av data/preference/stage3_dpo_strict_qwen3_rejected_glm52_chosen_train.jsonl \
  data/preference/stage3_dpo_strict_qwen3_rejected_glm52_chosen_val.jsonl \
  data/preference/stage1_gsm8k_qwen3_rejected_glm52_chosen_val.jsonl \
  root@SERVER:/workspace/dpo-cot-compression/data/preference/
```

Download the base model and Stage 3 adapter through Hugging Face cache:

```bash
HF_HOME=/workspace/.hf_home python scripts/00_download_model.py --model Qwen/Qwen3-8B
```

The Stage 3 adapter is referenced from:

```text
N7766/qwen3-8b-gsm8k-cot-compression-lora-sft-stage3
```

## Smoke Test

Before each run:

```bash
HF_HOME=/workspace/.hf_home python scripts/07_train_lora_dpo.py \
  --config configs/stage5/stage5_lora_dpo_sft_lr5e-7_beta003_steps100.yaml \
  --max_train_samples 128 \
  --max_eval_samples 64 \
  --max_steps 5 \
  --skip_save
```

For full DPO configs use:

```bash
HF_HOME=/workspace/.hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 scripts/08_train_full_dpo.py \
  --config configs/stage5/stage5_full_dpo_base_lr1e-7_beta003_steps50.yaml \
  --max_train_samples 128 \
  --max_eval_samples 64 \
  --max_steps 5 \
  --skip_save
```

## Training Commands

LoRA DPO:

```bash
HF_HOME=/workspace/.hf_home python scripts/07_train_lora_dpo.py \
  --config configs/stage5/stage5_lora_dpo_sft_lr5e-7_beta003_steps100.yaml
```

Full DPO:

```bash
HF_HOME=/workspace/.hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 scripts/08_train_full_dpo.py \
  --config configs/stage5/stage5_full_dpo_base_lr1e-7_beta003_steps50.yaml
```

## Checkpoint Evaluation

Evaluate every checkpoint from one run:

```bash
HF_HOME=/workspace/.hf_home python scripts/18_eval_checkpoints.py \
  --config configs/stage5/stage5_lora_dpo_sft_lr5e-7_beta003_steps100.yaml \
  --batch_size 32 \
  --max_new_tokens 128 \
  --include_final
```

Then compare all Stage 5 results with Stage 3 and Stage 4:

```bash
python scripts/19_compare_stage5_runs.py
```

Primary output:

```text
outputs/results/stage5_model_comparison.json
outputs/results/stage5_model_comparison.md
```

## Decision Rule

Keep a checkpoint only if it is on the accuracy/token Pareto frontier.

Good outcome:

```text
accuracy >= 93%
avg_output_tokens < 32.42
```

Reject even if the DPO objective looks good when:

```text
accuracy drops sharply
outputs become longer
answers lose the required final Answer: format
```
