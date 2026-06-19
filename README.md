# DPO-based Chain-of-Thought Compression

MSc dissertation project for studying DPO-based compression of chain-of-thought reasoning.

## Stage 1: Baseline Inference Pipeline

Stage 1 evaluates `Qwen/Qwen3-8B` on a small GSM8K subset and records:

- answer accuracy
- average output token length
- average reasoning token length

Reasoning tokens are counted inside Qwen-style `<think>...</think>` blocks when present. If no thinking tags are found, the full output token count is used as the reasoning length fallback.

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

## Optional Hugging Face Cache Cleanup

Stage 1 baseline inference uses the Hugging Face remote inference API by default and should not download local model weights.

If you accidentally started local `transformers` inference and interrupted a Qwen3-8B download, inspect the local macOS Hugging Face cache with:

```bash
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B* 2>/dev/null
```

If you confirm these are incomplete/unwanted Qwen3-8B cache directories, remove them manually with:

```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B*
```

No project Python script deletes Hugging Face cache files automatically.

## Data Download

Download and preprocess GSM8K from Hugging Face datasets:

```bash
python scripts/01_download_data.py --config configs/baseline.yaml
```

This writes:

```text
data/processed/gsm8k_test.jsonl
```

Each line has this format:

```json
{
  "id": "test_0",
  "question": "...",
  "answer": "...",
  "gold_answer": "72"
}
```

## Baseline Inference

The default backend in `configs/baseline.yaml` is:

```yaml
model:
  backend: hf_api
```

This calls Hugging Face's remote inference API through `huggingface_hub.InferenceClient` using the chat/conversational endpoint; it does not call `AutoTokenizer.from_pretrained` or `AutoModelForCausalLM.from_pretrained`, so it should not download local model weights.

Smoke test on three examples:

```bash
python scripts/02_baseline_inference.py --config configs/baseline.yaml --max_samples 3
```

Run the configured subset:

```bash
python scripts/02_baseline_inference.py --config configs/baseline.yaml
```

Generations are saved to:

```text
outputs/generations/baseline_qwen3_8b_gsm8k.jsonl
```

## Evaluation

Compute metrics from saved generations:

```bash
python scripts/03_baseline_eval.py --config configs/baseline.yaml
```

Metrics are saved to:

```text
outputs/results/baseline_qwen3_8b_gsm8k_metrics.json
```

Expected metric format:

```json
{
  "num_samples": 100,
  "accuracy": 0.82,
  "avg_output_tokens": 230.5,
  "avg_reasoning_tokens": 190.2
}
```

## Lightweight Sanity Checks

These checks do not load the 8B model:

```bash
python scripts/03_baseline_eval.py --self_test
```

They test GSM8K/model answer extraction and JSONL reading/writing.
