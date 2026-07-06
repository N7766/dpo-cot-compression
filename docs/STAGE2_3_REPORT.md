# Stage 2/3 Report: LoRA DPO Baseline and SFT Warm-Up

## 1. Project Context

This project investigates whether preference optimization can compress mathematical reasoning while preserving answer accuracy. Stage 1 constructed a GSM8K preference dataset where:

- `chosen`: concise correct GLM-5.2 teacher response.
- `rejected`: longer correct Qwen3-8B response.

Stages 2 and 3 test training routes on this dataset.

## 2. Stage 2 Objective

Stage 2 tested whether a LoRA adapter can be trained directly with DPO from the Stage 1 preference dataset.

Training route:

```text
Qwen/Qwen3-8B frozen base model
  -> initialize new LoRA adapter
  -> train adapter directly with DPO
  -> evaluate free-form GSM8K-style generation
```

## 3. Stage 2 Setup

| Item | Value |
| --- | ---: |
| Base model | `Qwen/Qwen3-8B` |
| Method | LoRA + DPO |
| Train pairs | 4163 |
| Validation pairs | 462 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| DPO beta | 0.1 |
| Learning rate | 5e-6 |
| Epochs | 1 |
| Effective batch size | 16 |
| GPU | 1 x RTX 5090 |
| Training runtime | 46 min 49 s |
| Peak reserved GPU memory | 26.35 GB |

## 4. Stage 2 Results

The DPO training objective looked strong:

| Metric | Value |
| --- | ---: |
| Final train loss | 0.1523 |
| Final eval loss | 0.0176 |
| Eval rewards accuracy | 0.9913 |
| Eval reward margin | 9.721 |

However, free-form validation generation was poor:

| Metric | Value |
| --- | ---: |
| Eval samples | 462 |
| Correct samples | 123 |
| Accuracy | 26.62% |
| Avg output tokens | 82.09 |
| Generation batch size | 4 |

Qualitative inspection showed output-format drift, including repeated templates, incorrect early `Answer:` lines, and unrelated continuation text.

Stage 2 conclusion:

```text
Direct LoRA DPO from a newly initialized adapter optimizes the pairwise DPO objective,
but it does not preserve reliable free-form GSM8K answer behavior.
```

This motivated a revised SFT-first route.

## 5. Stage 3 Objective

Stage 3 introduces supervised fine-tuning before DPO. The goal is to first teach the model the concise teacher-answer style, then later use DPO to sharpen compression preferences.

Training route:

```text
Qwen/Qwen3-8B
  -> LoRA SFT on concise GLM-5.2 chosen responses
  -> evaluate free-form generation
  -> use this SFT adapter as initialization for later DPO
```

This report covers the Stage 3 SFT warm-up only. Stage 3 DPO has not been run yet.

## 6. Stage 3 Data

Stage 3 data is derived from the Stage 1 preference dataset with stricter filtering.

| Split | SFT examples | Strict DPO examples |
| --- | ---: | ---: |
| Train | 4048 | 2894 |
| Validation | 451 | 320 |

Filtering rules:

| Rule | Value |
| --- | ---: |
| Require final `Answer:` tag | true |
| Maximum chosen tokens | 80 |
| Minimum rejected tokens | 80 |
| Minimum compression ratio | 30% |

Average statistics:

| Split | Avg SFT response tokens | Avg strict DPO rejected tokens | Avg strict DPO compression |
| --- | ---: | ---: | ---: |
| Train | 38.69 | 160.69 | 71.68% |
| Validation | 38.18 | 171.47 | 73.22% |

## 7. Stage 3 Setup

| Item | Value |
| --- | ---: |
| Base model | `Qwen/Qwen3-8B` |
| Method | LoRA SFT |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Max sequence length | 1024 |
| Learning rate | 1e-5 |
| Epochs | 1 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| GPUs | 2 x RTX 5090 |
| Effective batch size | 16 |
| Training runtime | 16 min 45 s |
| Peak allocated memory per GPU | about 16.78 GB |

Training command:

```bash
torchrun --nproc_per_node=2 scripts/17_train_lora_sft.py --config configs/stage3_lora_sft.yaml
```

## 8. Stage 3 Results

Training metrics:

| Metric | Value |
| --- | ---: |
| Final train loss | 0.3577 |
| Final eval loss | 0.2007 |

Free-form validation generation:

| Run | Samples | Batch size | Accuracy | Avg output tokens |
| --- | ---: | ---: | ---: | ---: |
| Quick check | 80 | 16 | 96.25% | 33.73 |
| Batch check | 80 | 32 | 96.25% | 33.63 |
| Full validation | 462 | 32 | 95.02% | 32.42 |

The full validation result is the main Stage 3 SFT result:

```text
accuracy = 95.02%
average output tokens = 32.42
```

## 9. Sample Quality Check

Manual sampling showed that most Stage 3 SFT outputs follow the desired pattern:

```text
short calculation steps
Answer: <number>
```

Observed validation behavior:

| Metric | Value |
| --- | ---: |
| Total validation outputs | 462 |
| Correct outputs | 439 |
| Incorrect outputs | 23 |
| Median output tokens | 31 |
| Maximum output tokens | 86 |
| Outputs with >=100 tokens | 0 |
| Missing `Answer:` tag | 8 |

The main remaining errors are ordinary arithmetic or problem-interpretation mistakes. A small number of outputs calculate the correct number in the reasoning but copy the wrong final number after `Answer:`.

## 10. Comparison

| Model / Stage | Training route | Eval samples | Accuracy | Avg output tokens |
| --- | --- | ---: | ---: | ---: |
| Stage 1 Qwen3 baseline subset | API baseline | 100 | 94.0% | 176.49 |
| Stage 2 LoRA DPO | Direct DPO | 462 | 26.62% | 82.09 |
| Stage 3 LoRA SFT | SFT warm-up | 462 | 95.02% | 32.42 |

The key finding is that the SFT warm-up stabilizes answer format and preserves accuracy while producing short responses. Direct DPO alone is not sufficient in this setup.

## 11. Artifacts

Stage 3 LoRA SFT adapter:

```text
https://huggingface.co/N7766/qwen3-8b-gsm8k-cot-compression-lora-sft-stage3
```

Repository:

```text
https://github.com/N7766/dpo-cot-compression
```

Generated datasets, logs, checkpoints, and evaluation JSON files are not committed to GitHub. They are ignored by git and treated as reproducible experiment artifacts.

## 12. Next Step

The next planned stage is Stage 4: full-parameter DPO on a more capable GPU setup.

Stage 4 should compare:

1. Stage 3 LoRA SFT adapter behavior.
2. Full-parameter DPO or FSDP-based DPO.
3. Accuracy and average output length on a held-out GSM8K evaluation set.

The main Stage 4 research question is whether full-parameter DPO can improve compression beyond the SFT warm-up while maintaining answer accuracy.
