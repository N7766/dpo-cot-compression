# Stage 5 Conservative LoRA-DPO Sweep

Stage 5 tested conservative LoRA-DPO updates initialized from the Stage 3 LoRA-SFT adapter.
The objective was to keep Stage 3 accuracy while reducing output length.

Validation set: `stage1_gsm8k_qwen3_rejected_glm52_chosen_val.jsonl`  
Validation samples: 462  
Generation setting: `max_new_tokens=128`, `batch_size=32`

## Baselines

| Model | Accuracy | Avg output tokens |
|---|---:|---:|
| Stage 3 LoRA-SFT | 95.02% | 32.42 |
| Stage 4 full DPO | 33.12% | 79.68 |

## Stage 5 Runs

| Run | Best checkpoint | LR | Beta | Max steps | Accuracy | Avg output tokens |
|---|---:|---:|---:|---:|---:|---:|
| `stage5_lora_dpo_sft_lr1e-6_beta003_steps50` | 50 | 1e-6 | 0.03 | 50 | 95.67% | 31.53 |
| `stage5_lora_dpo_sft_lr5e-7_beta003_steps100` | 75 | 5e-7 | 0.03 | 100 | 95.45% | 31.35 |
| `stage5_lora_dpo_sft_lr5e-7_beta005_steps100` | 100/final | 5e-7 | 0.05 | 100 | 95.02% | 31.50 |

## Best Result

Best checkpoint: `stage5_lora_dpo_sft_lr1e-6_beta003_steps50/checkpoint-50`

Compared with Stage 3:

| Metric | Stage 3 | Stage 5 best | Change |
|---|---:|---:|---:|
| Accuracy | 95.02% | 95.67% | +0.65 pp |
| Avg output tokens | 32.42 | 31.53 | -0.89 |

Conclusion: a short conservative DPO pass from the Stage 3 SFT adapter improves accuracy slightly while reducing output length. Longer or higher-beta DPO runs did not improve the accuracy-length tradeoff in this sweep.
