# Stage 1 Report: DPO Preference Data Construction for CoT Compression

## 1. Project Context

This project investigates whether Direct Preference Optimization (DPO) can reduce verbose chain-of-thought style mathematical reasoning while preserving answer accuracy. The target task is GSM8K arithmetic word-problem solving.

The central preference-data idea is:

- Rejected response: a correct but relatively verbose reasoning response.
- Chosen response: a correct and shorter reasoning response for the same question.

Stage 1 focuses on building and validating this data pipeline. Stage 2 will use the resulting preference dataset for DPO training.

## 2. Stage 1 Objectives

Stage 1 had four practical objectives:

1. Establish a small baseline evaluation pipeline for `Qwen/Qwen3-8B` on GSM8K.
2. Generate Qwen3-8B train-set responses to serve as candidate rejected responses.
3. Generate concise teacher responses with `zai-org/GLM-5.2` to serve as candidate chosen responses.
4. Build a filtered DPO preference dataset with train/validation splits.

All model calls were made through API backends. No local model weights are required for Stage 1.

## 3. Pipeline Overview

```text
GSM8K
  -> JSONL preprocessing
  -> Qwen3-8B generation
  -> answer extraction and correctness check
  -> filter correct nontrivial Qwen responses as rejected candidates
  -> GLM-5.2 concise teacher generation
  -> answer extraction and correctness check
  -> filter correct and shorter chosen responses
  -> DPO train/validation JSONL files
```

The final DPO format stores each example as:

- `prompt`: the GSM8K problem prompt.
- `chosen`: concise correct teacher response.
- `rejected`: verbose correct Qwen3-8B response.
- metadata: gold answer, extracted answers, token counts, compression ratio, model names.

## 4. Baseline Evaluation

The initial Stage 1A baseline used a 100-sample GSM8K subset with `Qwen/Qwen3-8B` through the Hugging Face API.

| Metric | Value |
| --- | ---: |
| Samples | 100 |
| Accuracy | 94.0% |
| Average output tokens, all samples | 176.49 |
| Average output tokens, correct samples | 96.23 |
| Average output tokens, correct nontrivial samples | 118.55 |
| Answer-only samples | 18 |
| Overthinking failures | 6 |
| Correct nontrivial samples | 76 |

Observation: the small baseline showed high answer accuracy, but responses split into several behavior types: concise answer-only outputs, useful multi-step reasoning, and long overthinking failures. This motivated filtering rather than using all outputs directly for preference construction.

## 5. Rejected Candidate Generation

For preference-data construction, Qwen3-8B was run on the GSM8K train split. The target was to collect correct but nontrivial reasoning responses.

| Metric | Value |
| --- | ---: |
| Generated samples analyzed | 6998 |
| Correct samples | 5200 |
| Incorrect samples | 1798 |
| Accuracy | 74.31% |
| Average output tokens, all samples | 277.99 |
| Average output tokens, correct samples | 127.12 |
| Average output tokens, incorrect samples | 714.34 |
| Correct nontrivial samples | 4829 |
| Answer-only correct samples | 371 |
| Overthinking failures | 774 |

Filtering rule for rejected candidates:

```text
is_correct == true
and output_tokens > 20
```

This keeps correct reasoning responses while excluding answer-only outputs that are already compressed. Incorrect or very long failure cases are excluded because the Stage 2 DPO goal is compression of correct reasoning, not learning from wrong reasoning.

## 6. Chosen Candidate Generation

The chosen responses were generated with `zai-org/GLM-5.2` through DeepInfra's OpenAI-compatible API. The prompt asked for brief reasoning with the final answer in a fixed `Answer: <number>` format.

The teacher generation was run against the 4829 Qwen3 correct nontrivial rejected candidates.

| Metric | Value |
| --- | ---: |
| Rejected input rows | 4829 |
| Chosen input rows | 4829 |
| Matched ids | 4829 |
| Filtered because chosen was incorrect | 113 |
| Filtered because chosen was not shorter | 91 |
| Final DPO pairs | 4625 |
| Average chosen tokens | 40.04 |
| Average rejected tokens | 136.16 |
| Average compression ratio | 63.68% |

Filtering rule for final DPO pairs:

```text
rejected is correct
chosen is correct
chosen is non-empty
rejected_tokens >= 21
chosen_tokens >= 1
chosen_tokens < rejected_tokens
```

Compression ratio is computed as:

```text
1 - chosen_tokens / rejected_tokens
```

The final average compression ratio of 63.68% indicates that the teacher chosen responses are substantially shorter than the Qwen rejected responses while preserving the numeric answer.

## 7. Final Stage 1 Dataset

The final preference dataset is split into train and validation sets using a fixed random seed.

| Split | Examples |
| --- | ---: |
| Train | 4163 |
| Validation | 462 |
| Total | 4625 |

Split configuration:

```yaml
split:
  val_ratio: 0.1
  seed: 42
```

Local generated output files:

```text
data/preference/stage1_gsm8k_qwen3_rejected_glm52_chosen_train.jsonl
data/preference/stage1_gsm8k_qwen3_rejected_glm52_chosen_val.jsonl
```

These files are generated locally and are intentionally excluded from git. The repository tracks the code, configuration files, and documentation needed to reproduce the pipeline, rather than committing generated JSONL experiment outputs.

## 8. Reproducibility and GitHub Repository

The GitHub repository contains:

- source code for data preparation, inference, evaluation, filtering, teacher generation, and DPO dataset construction;
- YAML configuration files for each Stage 1 step;
- project documentation and this Stage 1 report.

Generated data and experiment outputs are not uploaded to GitHub. This includes processed GSM8K JSONL files, model generation JSONL files, filtered candidate files, DPO train/validation JSONL files, metrics JSON files, logs, checkpoints, local caches, and API tokens.

This is intentional for three reasons:

1. Generated JSONL files can be large and are better treated as reproducible artifacts.
2. API keys and local environment files must never be committed.
3. The important reproducibility record is the script/config combination used to regenerate the artifacts.

The repository can therefore be cited as the implementation and reproduction record, while the numerical results in this report summarize the locally generated Stage 1 artifacts.

## 9. Implementation Notes

Stage 1 scripts are organized as:

| Step | Script | Purpose |
| --- | --- | --- |
| 1 | `scripts/01_prepare_gsm8k.py` | Download and preprocess GSM8K |
| 2 | `scripts/02_generate_qwen_rejected.py` | Generate Qwen3 baseline or rejected candidates |
| 3 | `scripts/03_evaluate_generations.py` | Evaluate answer accuracy and token length |
| 4 | `scripts/04_filter_rejected_candidates.py` | Analyze and filter Qwen generations |
| 5 | `scripts/05_generate_glm_chosen.py` | Generate concise GLM-5.2 chosen responses |
| 6 | `scripts/06_build_dpo_dataset.py` | Build filtered DPO train/validation data |

Important engineering decisions:

- API keys are read from environment variables only.
- Generated JSONL files are append/resume safe where API calls are used.
- Generated data, outputs, logs, checkpoints, and caches are ignored by git.
- The train/validation preference split is separate from final held-out evaluation.

## 10. Limitations and Risks

1. Token counts are approximate word-based counts for API outputs, not exact tokenizer counts.
2. Correctness is based on numerical answer extraction and exact numeric matching, which is suitable for GSM8K but not a general semantic evaluator.
3. Teacher chosen responses may contain subtle reasoning omissions even when the final answer is correct.
4. The validation split is drawn from filtered training data and should be used for Stage 2 training diagnostics only. Final model quality should be evaluated on held-out GSM8K test data.
5. The rejected data is conditioned on Qwen3-8B being correct; this intentionally narrows the task to compression of correct reasoning rather than error correction.

## 11. Stage 2 Plan

Stage 2 will train a DPO model using the Stage 1 preference dataset.

Planned next steps:

1. Select a base model and training strategy, likely LoRA+DPO first for cost efficiency.
2. Train on `stage1_gsm8k_qwen3_rejected_glm52_chosen_train.jsonl`.
3. Monitor validation loss and generation behavior on `stage1_gsm8k_qwen3_rejected_glm52_chosen_val.jsonl`.
4. Evaluate the trained model on held-out GSM8K test samples.
5. Compare against Qwen3-8B baseline accuracy and output length.

The key research question for Stage 2 is whether DPO can reduce reasoning length while maintaining answer accuracy.
