# DPO-based Chain-of-Thought Compression

MSc dissertation project studying whether Direct Preference Optimization (DPO) can compress chain-of-thought style mathematical reasoning while preserving answer accuracy.

## Project Goal

The long-term goal is to construct preference data where verbose but correct reasoning is treated as the rejected response and shorter correct reasoning is treated as the chosen response. The project will then use DPO to encourage more concise reasoning without sacrificing GSM8K answer accuracy.

## Stage 1: Data Construction

Stage 1 builds the evaluation baseline and the DPO preference dataset. Stage 2 is reserved for DPO training and model evaluation.

Full Stage 1 analysis report: [`docs/STAGE1_REPORT.md`](docs/STAGE1_REPORT.md)

Stage 1A baseline objective:

- Evaluate `Qwen/Qwen3-8B` on a 100-sample GSM8K subset.
- Use Hugging Face API inference to avoid local model weight downloads.
- Save model generations as JSONL.
- Compute answer accuracy and output length statistics.
- Identify correct nontrivial reasoning outputs.

Stage 1B/1C preference-data objective:

- Generate Qwen3-8B train-set responses as rejected candidates.
- Filter for correct nontrivial rejected responses.
- Generate shorter correct teacher responses with GLM-5.2 as chosen candidates.
- Build train/validation DPO JSONL files.

Pipeline:

```text
GSM8K -> Prompt -> Qwen3-8B HF API -> JSONL generations -> evaluation -> filtering analysis
GSM8K train -> Qwen3 rejected -> GLM-5.2 chosen -> filtered DPO train/val data
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Export your Hugging Face token before running API inference:

```bash
export HF_TOKEN=xxxxx
```

The token is read from the `HF_TOKEN` environment variable. Do not put tokens in config files or source code.

## Stage 1A Baseline Commands

Download and preprocess GSM8K:

```bash
python scripts/01_prepare_gsm8k.py --config configs/stage1_baseline.yaml
```

Run baseline inference:

```bash
python scripts/02_generate_qwen_rejected.py --config configs/stage1_baseline.yaml --max_samples 100
```

Inference is append-only and resume-safe. If `outputs/generations/baseline_qwen3_8b_gsm8k.jsonl` already contains completed sample ids, the script skips those ids and appends only new results. This allows larger runs to continue from the existing 100-sample baseline without overwriting it.

Evaluate saved generations:

```bash
python scripts/03_evaluate_generations.py --config configs/stage1_baseline.yaml
```

Analyze generation lengths and create filtering files:

```bash
python scripts/04_filter_rejected_candidates.py --config configs/stage1_baseline.yaml
```

## Stage 1 Results

Dataset: GSM8K

Model: `Qwen/Qwen3-8B`

Backend: Hugging Face API

| Metric | Value |
| --- | ---: |
| Total samples | 100 |
| Accuracy | 94.0% |
| Avg tokens, all | 176.49 |
| Avg tokens, correct | 96.23 |
| Avg tokens, correct nontrivial | 118.55 |
| Answer-only samples | 18 |
| Overthinking failures | 6 |
| Correct nontrivial samples | 76 |

Correct nontrivial samples are used as candidate rejected responses for future DPO preference construction. Answer-only outputs are excluded because they are already compressed. Incorrect or very long overthinking failures are kept for separate analysis and excluded from preference data for now.

Generated Stage 1 files are intentionally ignored by git:

- `outputs/generations/baseline_qwen3_8b_gsm8k.jsonl`
- `outputs/results/baseline_qwen3_8b_gsm8k_metrics.json`
- `outputs/results/baseline_qwen3_8b_gsm8k_analysis.json`
- `outputs/results/baseline_correct_nontrivial.jsonl`
- `outputs/results/baseline_answer_only.jsonl`
- `outputs/results/baseline_overthinking_failures.jsonl`

## Stage 1B/1C Preference Data

Preference construction uses GSM8K train data, not the 100-sample Stage 1A test baseline. Qwen3 train generations are used as candidate rejected responses, and GLM-5.2 teacher generations are used as concise chosen responses.

Stage 1 preference-data result:

| Metric | Value |
| --- | ---: |
| Qwen rejected candidates analyzed | 6998 |
| Correct nontrivial rejected candidates | 4829 |
| Matched GLM chosen candidates | 4829 |
| Final DPO pairs | 4625 |
| Train pairs | 4163 |
| Validation pairs | 462 |
| Avg chosen tokens | 40.04 |
| Avg rejected tokens | 136.16 |
| Avg compression ratio | 63.68% |

## Workflow Reference

| Step | Purpose | Script | Config |
| --- | --- | --- | --- |
| 1 | Prepare GSM8K JSONL | `scripts/01_prepare_gsm8k.py` | `configs/stage1_baseline.yaml` or `configs/stage1_generate_rejected.yaml` |
| 2 | Generate Qwen3 outputs/rejected candidates | `scripts/02_generate_qwen_rejected.py` | `configs/stage1_baseline.yaml` or `configs/stage1_generate_rejected.yaml` |
| 3 | Evaluate generations | `scripts/03_evaluate_generations.py` | matching generation config |
| 4 | Filter correct nontrivial rejected candidates | `scripts/04_filter_rejected_candidates.py` | matching generation config |
| 5 | Generate GLM-5.2 chosen responses | `scripts/05_generate_glm_chosen.py` | `configs/stage1_generate_chosen.yaml` |
| 6 | Build DPO preference dataset | `scripts/06_build_dpo_dataset.py` | `configs/stage1_build_dpo.yaml` |

The rejected-candidate config is:

```text
configs/stage1_generate_rejected.yaml
```

It uses:

- split: `train`
- max samples: `7000`
- processed input: `data/processed/gsm8k_train.jsonl`
- generation output: `outputs/generations/qwen3_8b_gsm8k_train_rejected_candidates.jsonl`

Prepare the train subset:

```bash
python scripts/01_prepare_gsm8k.py --config configs/stage1_generate_rejected.yaml
```

Generate Qwen3 rejected candidates with resume-safe append behavior:

```bash
python scripts/02_generate_qwen_rejected.py --config configs/stage1_generate_rejected.yaml
```

This does not start DPO training and does not generate teacher chosen responses.

Generate GLM-5.2 teacher chosen responses after the train correct-nontrivial file exists:

```bash
python scripts/05_generate_glm_chosen.py --config configs/stage1_generate_chosen.yaml
```

The production chosen output is:

```text
outputs/generations/glm52_gsm8k_train_chosen.jsonl
```

For a small DeepInfra smoke test, use:

```bash
python scripts/05_generate_glm_chosen.py --config configs/stage1_smoke_deepinfra_teacher.yaml --max_samples 3
```

Smoke-test output is kept separate:

```text
outputs/generations/deepinfra_glm52_teacher_smoke_test.jsonl
```

For chosen generation, use `--max_samples N` only for staged partial runs; omitting it processes all available input samples.

Build DPO preference pairs after rejected and chosen files are ready:

```bash
python scripts/06_build_dpo_dataset.py --config configs/stage1_build_dpo.yaml
```

This writes:

```text
data/preference/stage1_gsm8k_qwen3_rejected_glm52_chosen_train.jsonl
data/preference/stage1_gsm8k_qwen3_rejected_glm52_chosen_val.jsonl
outputs/results/stage1_gsm8k_qwen3_glm52_dpo_build_summary.json
```

The validation split is a held-out slice of the filtered preference data for Stage 2 training diagnostics and checkpoint selection. Final accuracy should still be measured on a separate GSM8K evaluation set.

## Stage 2: DPO Training

Stage 2 will train and evaluate the compression model using the Stage 1 DPO train/validation files. No Stage 2 training scripts are run as part of Stage 1 data construction.

Stage 2 training layout and naming plan: [`docs/STAGE2_PLAN.md`](docs/STAGE2_PLAN.md)

Planned Stage 2 configs:

- `configs/stage2_lora_dpo.yaml`
- `configs/stage2_full_dpo.yaml`
- `configs/stage2_eval.yaml`
- `configs/stage2_serve.yaml`
- `configs/stage2_upload.yaml`

Planned training outputs remain ignored by git under:

- `outputs/checkpoints/stage2_lora_dpo/`
- `outputs/checkpoints/stage2_full_dpo/`
- `outputs/logs/stage2_lora_dpo/`
- `outputs/logs/stage2_full_dpo/`
- `outputs/results/stage2_lora_dpo/`
- `outputs/results/stage2_full_dpo/`

Initial Stage 2 commands:

```bash
python scripts/00_download_model.py --config configs/stage2_lora_dpo.yaml --dry_run
python scripts/12_estimate_memory.py --config configs/stage2_lora_dpo.yaml --num_gpus 1
python scripts/07_train_lora_dpo.py --config configs/stage2_lora_dpo.yaml --dry_run
python scripts/08_train_full_dpo.py --config configs/stage2_full_dpo.yaml --dry_run
python scripts/10_eval_stage2_model.py --config configs/stage2_eval.yaml --dry_run
```

Full DPO is expected to run with a distributed launcher, for example:

```bash
torchrun --nproc_per_node=4 scripts/08_train_full_dpo.py --config configs/stage2_full_dpo.yaml
```

After training, use:

- `scripts/13_plot_training_curves.py` for loss/GPU memory plots
- `scripts/10_eval_stage2_model.py` for validation/test evaluation
- `scripts/14_serve_fastapi.py` for quick inference testing
- `scripts/11_upload_model.py` for optional Hugging Face upload

## Repository Layout

```text
configs/       Experiment configs
data/          Raw, processed, and preference data placeholders
scripts/       Stage scripts for data, inference, evaluation, and analysis
src/           Reusable data, evaluation, model, training, and utility code
outputs/       Ignored generated outputs and checkpoints
notebooks/     Lightweight analysis notebooks
```

## Optional Hugging Face Cache Cleanup

Stage 1 baseline inference uses the Hugging Face remote inference API and should not download local model weights.

If local `transformers` inference was accidentally started and interrupted, inspect the macOS Hugging Face cache with:

```bash
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B* 2>/dev/null
```

If those are incomplete or unwanted Qwen3-8B cache directories, remove them manually with:

```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B*
```

No project Python script deletes Hugging Face cache files automatically.

## Lightweight Sanity Checks

These checks do not load the 8B model:

```bash
python scripts/03_evaluate_generations.py --self_test
```

They test GSM8K/model answer extraction and JSONL reading/writing.
