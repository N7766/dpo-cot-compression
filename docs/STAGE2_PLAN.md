# Stage 2 Plan and Training Record

Stage 2 trains models from the Stage 1 preference dataset. This document records the naming conventions, actual Stage 2 hyperparameters/results, and the planned Stage 3 route.

## 1. Stage 2 Training Variants

Two training variants are planned:

| Variant | Config | Purpose |
| --- | --- | --- |
| LoRA+DPO | `configs/stage2_lora_dpo.yaml` | First, cheaper training path for iteration and dissertation experiments |
| Full-parameter DPO | `configs/stage2_full_dpo.yaml` | Heavier comparison run if GPU budget allows |

The first completed run is LoRA+DPO. Full DPO remains a later compute-heavy comparison and was not run in Stage 2.

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
| `scripts/00_download_model.py` | Pre-download/cache the base model on a GPU machine |
| `scripts/07_train_lora_dpo.py` | Train LoRA adapter with DPO |
| `scripts/08_train_full_dpo.py` | Train full model with DPO or FSDP/DeepSpeed |
| `scripts/09_merge_lora.py` | Merge LoRA adapter into base model for inference/export |
| `scripts/10_eval_stage2_model.py` | Evaluate trained model on held-out GSM8K |
| `scripts/11_upload_model.py` | Optional Hugging Face upload after review |
| `scripts/12_estimate_memory.py` | Rough memory estimate before training |
| `scripts/13_plot_training_curves.py` | Plot loss and GPU memory curves |
| `scripts/14_serve_fastapi.py` | Local FastAPI inference server |
| `scripts/15_probe_model_memory.py` | Load model once and report actual CUDA memory |
| `scripts/remote_gpu_quickstart.sh` | Remote GPU setup and dry-run helper |

Stage 2 should begin with `07_train_lora_dpo.py`.

## 7. Stage 2 Completed Run: Direct LoRA DPO

Stage 2 tested direct preference optimization from a newly initialized LoRA adapter.

```text
Qwen/Qwen3-8B frozen base model
  -> initialize new LoRA adapter
  -> train adapter directly with DPO
  -> evaluate free-form GSM8K-style generation
```

The base model weights were not updated. Only LoRA parameters were trained. This means the Stage 2 adapter started from a randomly/near-zero initialized LoRA state, without an SFT warm-up.

### Stage 2 Hardware and Runtime

| Item | Value |
| --- | ---: |
| GPU machine | Vast.ai |
| GPU used | 1 x RTX 5090 |
| Available GPUs | 2 x RTX 5090 |
| GPU memory per card | 32607 MiB |
| PyTorch | 2.11.0+cu128 |
| Attention implementation | `sdpa` |
| Training runtime | 2809 s / 46 min 49 s |
| Peak allocated GPU memory | 23.97 GB |
| Peak reserved GPU memory | 26.35 GB |
| Inference adapter load memory | 15.34 GB allocated |
| Batched generation eval memory | about 16.7 GB |

`flash_attention_2` was replaced with `sdpa` because FlashAttention2 was not installed in the remote environment. This made the run more portable and avoided an additional compiled dependency.

### Stage 2 LoRA DPO Hyperparameters

| Group | Hyperparameter | Value |
| --- | --- | ---: |
| Model | Base model | `Qwen/Qwen3-8B` |
| Model | dtype | `bfloat16` |
| Model | attention | `sdpa` |
| Data | train pairs | 4163 |
| Data | validation pairs | 462 |
| Data | `max_prompt_length` | 512 |
| Data | `max_length` | 2048 |
| LoRA | rank `r` | 8 |
| LoRA | alpha | 16 |
| LoRA | dropout | 0.05 |
| LoRA | target modules | q/k/v/o + gate/up/down projections |
| DPO | beta | 0.1 |
| DPO | loss type | `sigmoid` |
| DPO | reference | TRL/PEFT reference behavior, `ref_model=None` |
| Training | epochs | 1 |
| Training | per-device batch size | 1 |
| Training | gradient accumulation | 16 |
| Training | effective batch size | 16 |
| Training | learning rate | 5e-6 |
| Training | warmup ratio | 0.03 |
| Training | scheduler | cosine |
| Training | optimizer | `adamw_torch` |
| Training | gradient checkpointing | true |
| Training | eval steps | 100 |
| Training | save steps | 100 |
| Training | logging steps | 10 |
| Training | `report_to` | `[]` |

TensorBoard reporting was disabled because the installed Transformers/TRL stack attempted to JSON-serialize a `torch.dtype` object at train startup. The project GPU-memory callback still logged memory to JSONL, and training metrics were printed to the run log.

### Stage 2 Training Metrics

| Metric | Value |
| --- | ---: |
| Final train loss | 0.1523 |
| Final eval loss | 0.0176 |
| Eval rewards accuracy | 0.9913 |
| Eval reward margin | 9.721 |
| Eval mean token accuracy | 0.8177 |
| Eval samples per second | 4.795 |

These DPO validation metrics indicate that the adapter learned to prefer the teacher chosen responses over the rejected responses in the pairwise objective.

### Stage 2 Free-Form Generation Evaluation

The trained adapter was then evaluated by free-form generation on the 462 held-out Stage 1 validation questions.

| Metric | Value |
| --- | ---: |
| Eval samples | 462 |
| Generation `max_new_tokens` | 128 |
| Generation batch size | 4 |
| Accuracy | 26.62% |
| Correct samples | 123 |
| Avg output tokens | 82.09 |
| Runtime | about 18 min 36 s |

A small base-model comparison was run on the first 40 validation samples:

| Model | Samples | Accuracy | Avg output tokens |
| --- | ---: | ---: | ---: |
| Base Qwen3-8B | 40 | 32.5% | 76.05 |
| Stage 2 LoRA DPO | 40 | 20.0% | 82.8 |

Qualitative inspection showed output-format drift: some generations produced option lists, repeated templates, early incorrect `Answer:` lines, or continued with unrelated problem-like text. Therefore, Stage 2 should be treated as a direct-DPO baseline and a negative result for free-form accuracy.

Stage 2 conclusion:

```text
Direct LoRA DPO from an empty adapter optimizes the pairwise DPO objective,
but it does not preserve free-form GSM8K answer accuracy.
```

This motivates Stage 3.

## 8. Stage 3 Plan: SFT Warm-Up + LoRA DPO

Stage 3 changes the training route:

```text
Qwen/Qwen3-8B frozen base model
  -> initialize LoRA adapter
  -> SFT on GLM-5.2 chosen concise correct responses
  -> continue DPO from the SFT adapter
  -> evaluate free-form GSM8K accuracy and output length
```

The purpose of SFT is to stabilize the answer format and short reasoning behavior before preference optimization. DPO then compresses and sharpens the preference rather than learning from an empty LoRA adapter.

### Planned Hyperparameter Changes

| Area | Stage 2 Direct DPO | Stage 3 Planned Change | Reason |
| --- | --- | --- | --- |
| Training route | Direct LoRA DPO | LoRA SFT warm-up, then LoRA DPO | Stabilize generation before preference optimization |
| LoRA initialization | Empty/new LoRA adapter | Empty LoRA adapter for SFT, then continue same adapter with DPO | DPO starts from a teacher-imitation adapter |
| SFT data | Not used | GLM chosen responses only | Teach concise correct answer format |
| DPO data filtering | Existing filtered pairs | Stricter pair filtering | Remove weak compression pairs |
| Minimum compression | Not enforced strongly | Prefer `compression_ratio >= 0.3` | Ensure rejected is meaningfully longer |
| Rejected length | Correct nontrivial | Prefer `rejected_tokens >= 80` | Avoid training on already-compressed rejected answers |
| Chosen length | GLM chosen | Prefer `chosen_tokens <= 80` | Keep chosen concise |
| Sequence length | `max_length: 2048` | SFT likely `max_length: 1024`; DPO keep 2048 initially | Chosen responses are short; DPO may still need long rejected responses |
| GPU use | Single GPU | Smoke test single GPU, then 2-GPU DDP | Use both RTX 5090s while keeping memory stable |
| DPO batch | 1 GPU x batch 1 x grad acc 16 | 2 GPUs x per-device batch 1 x grad acc 8 | Preserve effective batch size 16 while improving speed |
| Full DPO | Planned but not run | Defer | 2 x 5090 is not enough for comfortable full DPO |

Suggested Stage 3 SFT starting point:

| Hyperparameter | Value |
| --- | ---: |
| Base model | `Qwen/Qwen3-8B` |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| SFT max length | 1024 |
| SFT per-device batch size | 1 or 2 |
| SFT gradient accumulation | 8 or 16 |
| SFT learning rate | 1e-5 to 2e-5 |
| SFT epochs | 1 |
| DPO beta | 0.1 initially |
| DPO learning rate | 5e-6 initially |

The first Stage 3 run should be a small smoke test before full training:

```text
SFT smoke test -> free-form eval sample -> full SFT -> free-form eval -> DPO from SFT adapter
```

Stage 3 data/config/script files:

| Purpose | Path |
| --- | --- |
| Build Stage 3 data | `configs/stage3_build_data.yaml` |
| LoRA SFT config | `configs/stage3_lora_sft.yaml` |
| LoRA DPO-from-SFT config | `configs/stage3_lora_dpo.yaml` |
| Build Stage 3 data script | `scripts/16_build_stage3_data.py` |
| Train Stage 3 LoRA SFT script | `scripts/17_train_lora_sft.py` |

Current Stage 3 data build:

| Split | SFT rows | Strict DPO rows | Avg SFT tokens | Avg rejected tokens | Avg compression |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 4048 | 2894 | 38.69 | 160.69 | 0.7168 |
| Validation | 451 | 320 | 38.18 | 171.47 | 0.7322 |

Stage 3 commands:

```bash
python scripts/16_build_stage3_data.py --config configs/stage3_build_data.yaml
python scripts/17_train_lora_sft.py --config configs/stage3_lora_sft.yaml --dry_run
python scripts/17_train_lora_sft.py --config configs/stage3_lora_sft.yaml --max_steps 5
python scripts/17_train_lora_sft.py --config configs/stage3_lora_sft.yaml
python scripts/07_train_lora_dpo.py --config configs/stage3_lora_dpo.yaml
```

## 9. Original Training Pipeline Notes

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

## 10. Remote SSH Workflow

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

The quickstart helper performs dependency installation, dry-runs, and memory estimates:

```bash
bash scripts/remote_gpu_quickstart.sh
```

Copy local generated Stage 1 preference data if not regenerating remotely:

```bash
rsync -av data/preference/stage1_gsm8k_qwen3_rejected_glm52_chosen_train.jsonl <remote>:/workspace/dpo-cot-compression/data/preference/
rsync -av data/preference/stage1_gsm8k_qwen3_rejected_glm52_chosen_val.jsonl <remote>:/workspace/dpo-cot-compression/data/preference/
```

Then run LoRA+DPO training once the script exists:

```bash
python scripts/00_download_model.py --config configs/stage2_lora_dpo.yaml
python scripts/12_estimate_memory.py --config configs/stage2_lora_dpo.yaml --num_gpus 1
python scripts/15_probe_model_memory.py --config configs/stage2_lora_dpo.yaml
python scripts/07_train_lora_dpo.py --config configs/stage2_lora_dpo.yaml
```

For a no-model dry run:

```bash
python scripts/07_train_lora_dpo.py --config configs/stage2_lora_dpo.yaml --dry_run
python scripts/08_train_full_dpo.py --config configs/stage2_full_dpo.yaml --dry_run
```

For full DPO, use a distributed launcher. Example:

```bash
python scripts/00_download_model.py --config configs/stage2_full_dpo.yaml
python scripts/12_estimate_memory.py --config configs/stage2_full_dpo.yaml --num_gpus 4
torchrun --nproc_per_node=4 scripts/08_train_full_dpo.py --config configs/stage2_full_dpo.yaml
```

After training, plot loss and GPU memory:

```bash
python scripts/13_plot_training_curves.py \
  --trainer_state outputs/checkpoints/stage2_lora_dpo/qwen3_8b_gsm8k_cot_compression_lora/trainer_state.json \
  --gpu_memory outputs/results/stage2_lora_dpo/qwen3_8b_gsm8k_cot_compression_lora_dpo_gpu_memory.jsonl \
  --output_dir outputs/results/stage2_lora_dpo/plots
```

Evaluate the LoRA model on the Stage 1 validation split:

```bash
python scripts/10_eval_stage2_model.py --config configs/stage2_eval.yaml --max_samples 100
```

Start a small FastAPI inference server:

```bash
python scripts/14_serve_fastapi.py --config configs/stage2_serve.yaml
```

Upload an adapter or model directory to Hugging Face after review:

```bash
python scripts/11_upload_model.py --config configs/stage2_upload.yaml --dry_run
python scripts/11_upload_model.py --config configs/stage2_upload.yaml
```

## 11. Evaluation Policy

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

## 12. Safety Notes

- Do not commit checkpoints, adapters, merged models, logs, or generated evaluation outputs.
- Do not commit `.env`, API keys, SSH keys, or Hugging Face tokens.
- Keep base model weights in Hugging Face cache or an external model directory, not in the repository.
- Prefer LoRA+DPO first to validate the pipeline before spending full DPO GPU time.
