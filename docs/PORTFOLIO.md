# DPO 推理链压缩：项目与结果概览

本项目研究如何通过偏好优化缩短数学推理输出，并保留答题准确率。实现包括偏好数据构建、SFT / DPO 配置、检查点评估及结果比较。

## 可核对的结果

下表来自个人实验的原始 summary 文件。数值摘要与源文件 SHA-256 已保存到 [experiment-summary.json](portfolio/experiment-summary.json)。

| 模型 | 验证集正确题数 | 准确率 | 平均输出长度（按空格切分的词） |
| --- | --- | --- | --- |
| LoRA-SFT | 439 / 462 | 95.02% | 32.4221 |
| SFT 初始化 LoRA-DPO，checkpoint-50 | 442 / 462 | 95.67% | 31.5281 |

准确率增加约 0.649 个百分点，平均词数减少约 2.7572%。这是这一组验证集比较的观察结果。

## 结论边界

- 462 题验证集参与检查点选择，不能当作独立测试集成绩。
- 原字段 `avg_output_tokens` 实际来自 `len(prediction.split())`；这里明确改称按空格切分的词数。
- 摘要中的正确题数不足以证明统计显著性。
- 词数变化不能直接证明推理时延、吞吐或显存收益；需要单独测量。
- 此处保存经过整理的数值证据，不包含模型权重或逐题生成内容。

## 实现入口

- [训练与评测配置](../configs)
- [检查点评估](../scripts/18_eval_checkpoints.py)
- [Stage 5 比较脚本](../scripts/19_compare_stage5_runs.py)
- [原始评估器与长度计算](../scripts/10_eval_stage2_model.py)

完整阶段说明与运行命令见 [README](../README.md)。

## 后续工作

保留独立评测题，增加同设置下的速度测量，并记录更多失败案例。配套的 [LabLens](https://github.com/N7766/lablens-agent) 将这些摘要接入只读工具，提供实验对比、方法检索与来源追溯；LabLens 当前为离线已验证原型。
