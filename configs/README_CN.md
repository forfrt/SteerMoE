# 配置文件

本文件夹包含控制 SteerMoE 模型训练和评估各方面的 YAML 配置文件。

## 📁 文件组织

配置文件遵循以下命名规范：
```
{方法}_{编码器}_{解码器}_{数据集}_{模式}_{变体}.yaml
```

其中：
- **方法**：`layer_wise`（SteerMoE）或 `linear`（消融基线）或 `lora`（LoRA 基线）
- **编码器**：`whisper` 或 `conformer`
- **解码器**：`qwen3b`、`qwen7b` 等
- **数据集**：`libri`、`aishell`、`clothoaqa` 等
- **模式**：`train` 或 `eval`/`test`
- **变体**：`aed`（音频事件检测）、`linear`（仅线性）等

## 🎯 主要配置文件

### DeepSpeed 配置（JSON）

| 文件 | ZeRO 阶段 | 说明 |
|------|-----------|------|
| `stage0.json` | Stage 0 | 无优化，基础分布式训练 |
| `stage2.json` | Stage 2 | 梯度和优化器状态分区，**最常用** |
| `stage2_simple.json` | Stage 2 | 简化版，较小的桶大小，适合内存受限场景 |
| `stage3.json` | Stage 3 | 全参数分区 + CPU 卸载，适合超大模型 |
| `stage3_wo_offload.json` | Stage 3 | 无 CPU 卸载，需要更多 GPU 显存但更快 |

### 环境配置

| 文件 | 说明 |
|------|------|
| `rt_mo.yml` | Conda 环境定义（Python 3.10、PyTorch 2.1.1、CUDA 12.4） |
| `whisper_train.yml` | Whisper 训练环境定义 |

### 基础配置模板

| 文件 | 说明 |
|------|------|
| `default.yaml` | 模板配置（Whisper + GPT-2，8 专家） |
| `layer_wise.yaml` | 逐层 SteerMoE 基础配置（Whisper-large-v3 + Qwen2.5-7B） |
| `layer_wise_en.yaml` | 英文变体（LibriSpeech + Qwen2.5-0.5B） |

### 训练配置

#### 英文 ASR（LibriSpeech）

| 文件 | 编码器 | 解码器 | 说明 |
|------|--------|--------|------|
| `layer_wise_whisper_qwen7b_libri_train.yaml` | Whisper-large-v3 | Qwen2.5-7B | **英文推荐** |
| `layer_wise_whisper_qwen3b_libri_train.yaml` | Whisper-large-v3 | Qwen2.5-3B | 更快训练，更低显存 |
| `layer_wise_conformer_qwen7b_libri_train.yaml` | Conformer | Qwen2.5-7B | Conformer 编码器 |
| `layer_wise_conformer_qwen7b_libri_train_aed.yaml` | Conformer | Qwen2.5-7B | 带音频事件检测 |
| `layer_wise_conformer_qwen3b_libri_train_aed.yaml` | Conformer | Qwen2.5-3B | Qwen3B 变体 |

#### 中文 ASR（AISHELL）

| 文件 | 编码器 | 解码器 | 说明 |
|------|--------|--------|------|
| `layer_wise_whisper_qwen7b_aishell_train.yaml` | Whisper-large-v3 | Qwen2.5-7B | 标准中文 ASR |

#### 音频问答（ClothoAQA）

| 文件 | 编码器 | 解码器 | 说明 |
|------|--------|--------|------|
| `layer_wise_whisper_qwen7b_clothoaqa_train.yaml` | Whisper-large-v3 | Qwen2.5-7B | 音频问答任务 |
| `layer_wise_conformer_qwen7b_clothoaqa_train_aed.yaml` | Conformer | Qwen2.5-7B | Conformer 变体 |
| `layer_wise_conformer_qwen3b_clothoaqa_train_aed.yaml` | Conformer | Qwen2.5-3B | Qwen3B 变体 |

#### 消融实验基线

| 文件 | 说明 |
|------|------|
| `layer_wise_whisper_qwen7b_libri_train_linear.yaml` | 仅线性投影（无 SteerMoE），2 epochs |
| `linear_conformer_qwen3b_libri_train_aed.yaml` | Conformer + 线性基线，5 epochs |
| `linear_conformer_qwen7b_libri_train_aed.yaml` | Conformer + 线性基线，5 epochs |
| `lora_whisper_qwen7b_libri_train.yaml` | LoRA 训练基线（rank=8, alpha=1.0） |

### 评估/测试配置

评估配置使用相同命名，后缀为 `_test` 或 `_eval`：

| 文件 | 说明 |
|------|------|
| `layer_wise_whisper_qwen7b_libri_test.yaml` | LibriSpeech 测试 |
| `layer_wise_whisper_qwen7b_aishell_test.yaml` | AISHELL 测试 |
| `layer_wise_whisper_qwen7b_clothoaqa_test.yaml` | ClothoAQA 测试 |
| `layer_wise_whisper_qwen7b_openaudiobench_test.yaml` | OpenAudioBench 测试 |
| `layer_wise_conformer_qwen7b_libri_eval_aed.yaml` | Conformer LibriSpeech 评估 |
| `layer_wise_conformer_qwen3b_libri_eval_aed.yaml` | Conformer Qwen3B 评估 |
| `layer_wise_conformer_qwen7b_clothoaqa_eval_aed.yaml` | Conformer ClothoAQA 评估 |
| `layer_wise_conformer_qwen3b_clothoaqa_eval_aed.yaml` | Conformer Qwen3B 评估 |
| `linear_conformer_qwen3b_libri_eval_aed.yaml` | 线性基线评估 |
| `linear_conformer_qwen7b_libri_eval_aed.yaml` | 线性基线评估 |
| `lora_whisper_qwen7b_libri_test.yaml` | LoRA 评估 |
| `layer_wise_whisper_qwen7b_libri_test_linear.yaml` | 线性 Whisper 测试 |

## 📝 配置结构

### 示例配置

```yaml
# ============= 音频编码器 =============
whisper_encoder:
  model_path: "/mnt/models/whisper-large-v3/"
  feature_dim: 1280              # Whisper-large 输出维度
  num_layers: 32                 # 编码器层数

# ============= 语言解码器 =============
llm_decoder:
  model_name: "/mnt/models/Qwen2.5-7B-Instruct/"
  max_length: 512
  use_cache: false               # 训练时禁用 KV 缓存

# ============= 引导配置 =============
steering:
  num_experts: 8                 # MoE 专家数量
  steering_scale: 0.1            # 引导强度
  use_layer_scales: true         # 启用逐层缩放
  steering_gradient_clip: 1.0    # 引导梯度裁剪

# ============= 训练参数 =============
training:
  batch_size: 4
  learning_rate: 1e-4            # 基础学习率
  steering_learning_rate: 1e-3   # 引导参数学习率（更高）
  num_epochs: 10
  warmup_steps: 1000
  fp16: true
  deepspeed: "configs/stage2.json"
```

### 关键参数说明

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `num_experts` | 8 或 16 | MoE 专家数量 |
| `steering_scale` | 0.1 | 引导强度缩放因子 |
| `use_layer_scales` | true | 是否启用逐层缩放 |
| `steering_gradient_clip` | 1.0 | 引导梯度裁剪阈值 |
| `batch_size` | 3-8 | 每设备批大小 |
| `learning_rate` | 1e-4 ~ 1e-3 | 基础学习率 |
| `steering_learning_rate` | 1e-3 ~ 1e-2 | 引导参数学习率 |
| `warmup_steps` | 1000 | 预热步数 |
| `max_prompt_tokens` | 2048 | 最大音频序列长度 |
| `max_text_length` | 448 | 最大文本长度 |
| `sample_rate` | 16000 | 音频采样率 (Hz) |

## 🔧 常见问题排查

### 显存不足 (OOM)

1. 减小批大小：`batch_size: 1` 或 `2`
2. 启用梯度检查点：`gradient_checkpointing: true`
3. 限制序列长度：降低 `max_prompt_tokens` 和 `max_text_length`
4. 使用 DeepSpeed ZeRO-3：切换到 `stage3.json`

### 收敛困难

1. 检查学习率：确保 `steering_learning_rate >> learning_rate`
2. 增加预热步数：`warmup_steps: 2000`
3. 验证数据：检查 `textual_prompt` 是否匹配语言/任务
4. 监控引导：启用 `log_steering_analysis: true`

### 训练速度慢

1. 启用混合精度：`fp16: true`
2. 增大批大小（如显存允许）
3. 降低日志频率：增大 `logging_steps` 和 `eval_steps`
4. 加速数据加载：`dataloader_num_workers: 4`

## 🔗 相关文档

- 模型配置参考：[`steer_moe/models.py`](../steer_moe/models.py)
- 训练脚本：[`scripts/README.md`](../scripts/README.md) | [`scripts/README_CN.md`](../scripts/README_CN.md)
- 数据整理器：[`steer_moe/utils.py`](../steer_moe/utils.py)
