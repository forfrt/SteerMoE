# SteerMoE 评估规范

## 评估指标

| 指标 | 脚本 | 适用任务 |
|------|------|---------|
| WER (Word Error Rate) | `scripts/wer.py` | 英文 ASR (LibriSpeech) |
| CER (Character Error Rate) | `scripts/cer.py` | 中文 ASR (AISHELL) |
| Accuracy | 手动计算 | 音频 QA (ClothoAQA, MMAU) |

## 评估流程

### 通过训练脚本评估

```bash
# Conformer 模型评估
python scripts/train_layer_wise_conformer.py \
  --config configs/layer_wise_conformer_qwen7b_libri_eval_aed.yaml \
  --mode eval \
  --eval_dataset librispeech_test_clean \
  --model_path results/<experiment>/checkpoint-<step>

# Whisper 模型评估
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_test.yaml \
  --mode eval \
  --eval_dataset librispeech_test_clean \
  --model_path results/<experiment>/checkpoint-<step>
```

### 评估函数内部流程

`evaluate_layer_wise_model()`:
1. 加载模型和 checkpoint 权重
2. 构建评估 DataLoader
3. 逐 batch 调用 `model.generate()` 自回归生成
4. Decode 生成的 token IDs → 文本
5. 去除 textual prompt 前缀 (匹配并剥离)
6. 使用 `load_metric('./scripts/cer.py')` 或 `wer.py` 计算指标 (来自 `datasets` 库)

### 关键注意事项

- 评估时 `model.eval()` 和 `torch.no_grad()`
- 生成时 textual prompt 需与训练时一致
- 去除 prompt 前缀后才计算 WER/CER
- 中文 CER 评估需要字级别分词

## 主要基准数据集

| 数据集 | 语言 | 任务 | 指标 | 测试集路径 |
|--------|------|------|------|-----------|
| LibriSpeech test-clean | EN | ASR | WER | `~/data/librispeech_asr/all/test.clean/` |
| LibriSpeech test-other | EN | ASR | WER | `~/data/librispeech_asr/all/test.other/` |
| AISHELL-1 test | ZH | ASR | CER | `~/data/AISHELL-DEV-TEST-SET/iOS/test/` |
| ClothoAQA test | EN | Audio QA | Accuracy | `~/data/ClothoAQA/` |
| MMAU-Pro test | EN | Multi-task AU | Accuracy | `~/data/MMAU-Pro/` |
| OpenAudioBench | EN | Audio QA | Accuracy | `~/data/OpenAudioBench/` |

## 参考性能

| 模型配置 | LibriSpeech test-clean WER | 可训练参数 |
|---------|---------------------------|-----------|
| SteerMoE (C7B, 8 experts) | 2.42% | ~1.8M |
| SteerMoE (C7B, 16 experts) | 2.43% | ~2.1M |
| SteerMoE (C7B, 4 experts) | 3.10% | ~1.5M |
| SteerMoE (C7B, 2 experts) | 6.22% | ~1.3M |
| Static Adapter (无 MoE) | >100% (崩溃) | ~1.1M |
| Whisper-large-v3 单独 | 2.7% | 1550M |

## 训练监控

### Callbacks 输出

**SteeringAnalysisCallback** 每 N 步 (默认 100) 记录:
- 各层 `steering_vectors` 范数
- `gating_scores` 分布 (专家利用率)
- `layer_scales` 值

**GradientClippingCallback**: 对 steering 参数独立裁剪梯度，防止训练不稳定。

### 训练收敛观察

- 前 ~1000 步: loss 快速下降 (router 学习分配模式)
- ~1000-5000 步: 稳定下降 (steering vectors 精细调整)
- 5000+ 步: 收敛平稳

### 常见问题

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| Loss 不降 | steering_learning_rate 太低 | 提高到 1e-2 |
| Loss NaN | 梯度爆炸 | 启用 gradient_clipping, 减小 learning_rate |
| WER 很高 | textual_prompt 不匹配 | 确保 train/eval prompt 一致 |
| 所有输出相同 | 专家坍塌 | 增加 num_experts, 检查 router 梯度 |
| OOM | batch_size 过大 | 减小 batch_size, 启用 DeepSpeed |
