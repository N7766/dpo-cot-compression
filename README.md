# DPO-based Chain-of-Thought Compression

MSc dissertation project studying whether Direct Preference Optimization (DPO) can compress chain-of-thought style mathematical reasoning while preserving answer accuracy.

## Project Goal

The long-term goal is to construct preference data where verbose but correct reasoning is treated as the rejected response and shorter correct reasoning is treated as the chosen response. The project will then use DPO to encourage more concise reasoning without sacrificing GSM8K answer accuracy.

## Stage 1: Baseline

Stage 1 establishes a baseline inference and evaluation pipeline before any preference construction or DPO training.

Objective:

- Evaluate `Qwen/Qwen3-8B` on a 100-sample GSM8K subset.
- Use Hugging Face API inference to avoid local model weight downloads.
- Save model generations as JSONL.
- Compute answer accuracy and output length statistics.
- Identify correct nontrivial reasoning outputs for future DPO rejected candidates.

Pipeline:

```text
GSM8K -> Prompt -> Qwen3-8B HF API -> JSONL generations -> evaluation -> filtering analysis
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

## Stage 1 Commands

Download and preprocess GSM8K:

```bash
python scripts/01_download_data.py --config configs/baseline.yaml
```

Run baseline inference:

```bash
python scripts/02_baseline_inference.py --config configs/baseline.yaml --max_samples 100
```

Inference is append-only and resume-safe. If `outputs/generations/baseline_qwen3_8b_gsm8k.jsonl` already contains completed sample ids, the script skips those ids and appends only new results. This allows larger runs to continue from the existing 100-sample baseline without overwriting it.

Evaluate saved generations:

```bash
python scripts/03_baseline_eval.py --config configs/baseline.yaml
```

Analyze generation lengths and create filtering files:

```bash
python scripts/04_analyze_generations.py --config configs/baseline.yaml
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

## Stage 2 Preparation: Rejected Candidates

Stage 2 preference construction should use GSM8K train data, not the Stage 1 test baseline. The Qwen3 train generations will be used as candidate rejected responses; chosen responses will be generated later by a teacher model.

The rejected-candidate config is:

```text
configs/generate_data.yaml
```

It uses:

- split: `train`
- max samples: `3000`
- processed input: `data/processed/gsm8k_train.jsonl`
- generation output: `outputs/generations/qwen3_8b_gsm8k_train_rejected_candidates.jsonl`

Prepare the train subset:

```bash
python scripts/01_download_data.py --config configs/generate_data.yaml
```

Generate Qwen3 rejected candidates with resume-safe append behavior:

```bash
python scripts/02_baseline_inference.py --config configs/generate_data.yaml
```

This does not start DPO training and does not generate teacher chosen responses.

Generate GLM-5.2 teacher chosen responses after the train correct-nontrivial file exists:

```bash
python scripts/06_deepinfra_teacher_test.py --config configs/deepinfra_test.yaml
```

Use `--max_samples N` only for smoke tests; omitting it processes all available input samples.

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
python scripts/03_baseline_eval.py --self_test
```

They test GSM8K/model answer extraction and JSONL reading/writing.
