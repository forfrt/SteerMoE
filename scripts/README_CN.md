# 训练与评估脚本

本文件夹包含 SteerMoE 模型训练、评估和分析的主要脚本。

## 📋 概览

脚本按编码器类型和用途分类：

| 脚本 | 编码器 | 用途 | 适用场景 |
|------|--------|------|----------|
| `train_layer_wise.py` | Whisper | **主训练脚本** | 英文 ASR、音频问答 |
| `train_layer_wise_prompt.py` | Whisper | 带文本提示的训练 | 需要文本提示的任务 |
| `train_layer_wise_conformer.py` | Conformer | **主训练脚本** | 中文 ASR、流式场景 |
| `train_layer_wise_conformer_clothoaqa.py` | Conformer | ClothoAQA 训练 | 音频问答任务 |
| `train_layer_wise_linear_whisper.py` | Whisper | 消融实验基线 | 仅线性投影对比 |
| `train_conformer_linear.py` | Conformer | 消融实验基线 | 仅线性投影对比 |
| `train_lora.py` | Whisper | LoRA 消融实验 | LoRA 基线对比 |
| `train.py` | Whisper | 原始 SteerMoE 训练 | 非逐层变体 |
| `eval_lora.py` | Whisper | LoRA 评估 | 评估 LoRA 模型 |
| `cer.py` | - | 评估指标 | 字符错误率 (CER) |
| `wer.py` | - | 评估指标 | 词错误率 (WER) |

## 🎯 主要训练脚本

### `train_layer_wise.py`（推荐）

Whisper + SteerMoE 架构的主训练脚本。

#### 功能特性

- 逐层引导 (Layer-wise Steering) + MoE 路由
- 所有层共享单一高效路由器
- 可训练的引导向量和层缩放因子
- 线性投影适配器
- DeepSpeed 分布式训练
- 梯度裁剪保证训练稳定性
- 引导模式分析回调

#### 使用方法

**训练**：
```bash
# 单 GPU
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --mode train

# 多 GPU + DeepSpeed（推荐）
deepspeed --num_gpus=4 scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --deepspeed_config configs/stage2_simple.json \
  --mode train

# 从检查点恢复训练
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --mode train \
  --resume_from /path/to/checkpoint-5000
```

**评估**：
```bash
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_test.yaml \
  --mode eval \
  --model_path results/layer_wise_steermoe/final
```

**引导模式分析**：
```bash
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_test.yaml \
  --mode analyze \
  --model_path results/layer_wise_steermoe/final
```

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | `configs/layer_wise.yaml` | 配置 YAML 文件路径 |
| `--deepspeed_config` | str | `configs/stage2_simple.json` | DeepSpeed 配置文件 |
| `--eval_dataset` | str | `None` | 评估数据集名称 |
| `--resume_from` | str | `None` | 恢复训练的检查点路径 |
| `--mode` | str | `train` | 模式：`train`、`eval` 或 `analyze` |
| `--model_path` | str | `None` | 评估/分析时的模型路径 |

### `train_layer_wise_prompt.py`

在 `train_layer_wise.py` 基础上增加了文本提示支持。

- 使用 `DataCollatorSpeechSeqSeqWithPaddingPrompt` 数据整理器
- 支持数据集中的 `text_prompt` 列
- 其余训练基础设施与 `train_layer_wise.py` 相同

### `train_layer_wise_conformer.py`

基于 Conformer 编码器的逐层引导训练脚本。

- 使用 `EfficientLayerWiseSteeringConformerEncoder`
- 支持中文 ASR 和流式场景
- 包含引导分析和梯度裁剪的自定义回调
- 支持 `train`、`eval`、`analyze` 三种模式

### `train_layer_wise_conformer_clothoaqa.py`

基于 Conformer 编码器的 ClothoAQA 音频问答训练脚本。

- 专为音频问答任务设计
- 使用 Conformer 编码器 + 逐层引导

## 🔬 消融实验脚本

### `train_layer_wise_linear_whisper.py`

线性路由消融基线，使用 `SteerMoEEfficientLayerWiseModelLinear` 替代标准路由机制。

### `train_conformer_linear.py`

Conformer 编码器 + 线性路由消融基线。

### `train_lora.py`

LoRA 消融实验脚本。

- 训练 LoRA 适配的编码器，LLM 解码器冻结
- 使用 `LoRAModel` 和 `LoRAWhisperEncoder`
- 提供与 SteerMoE 对比的基线

### `eval_lora.py`

LoRA 模型评估脚本。

- 计算 CER、WER、token 准确率、原始准确率
- 参数：`--config`、`--model_path`、`--local_rank`

## 📊 评估指标脚本

### `cer.py` — 字符错误率 (CER)

HuggingFace Datasets 指标类，计算字符级别错误：

```
CER = (替换数 + 删除数 + 插入数) / 总字符数
```

支持 `concatenate_texts` 选项用于批量评估，使用 `jiwer` 库计算。

### `wer.py` — 词错误率 (WER)

HuggingFace Datasets 指标类，计算词级别错误：

```
WER = (替换数 + 删除数 + 插入数) / 总词数
```

支持 `concatenate_texts` 选项用于批量评估，使用 `jiwer` 库计算。

## 📝 其他文件

### `train.py`

原始 SteerMoE 模型（非逐层变体）的训练脚本。支持 `SteerMoEModel` 和 `SteerMoEHybridModel`，包含 DeepSpeed 集成、HuggingFace Trainer、Hub 集成等功能。

### `example_training.py`

示例脚本，展示如何使用 `train.py` 的功能，包括数据集加载、过滤和训练示例。

### Shell 脚本

| 脚本 | 说明 |
|------|------|
| `steermoe_train.sh` | 逐层 SteerMoE 的 DeepSpeed 训练脚本（GPU 1,2） |
| `lora_train.sh` | LoRA 模型的 DeepSpeed 训练脚本（GPU 1,2） |

### 提示文件

| 文件 | 说明 |
|------|------|
| `prompt` | DataCollatorSpeechSeqSeqWithPadding 源码文档 |
| `prompt_conformer` | Conformer 专用数据整理器源码文档 |

### 日志文件

| 文件 | 说明 |
|------|------|
| `lora_whisper_qwen7b_libri_train.log` | LoRA 模型训练日志 |
| `lora_whisper_qwen7b_libri_test.log` | LoRA 模型测试日志 |
| `layer_wise_whisper_qwen7b_libri_train_moe16.log` | 16 专家逐层 SteerMoE 训练日志 |
| `layer_wise_whisper_qwen7b_libri_test_moe16.log` | 16 专家逐层 SteerMoE 测试日志 |

## 🔗 相关文档

- 模型实现：[`steer_moe/models.py`](../steer_moe/models.py)
- 训练配置：[`configs/README.md`](../configs/README.md) | [`configs/README_CN.md`](../configs/README_CN.md)
- 数据预处理：[`pre_process/README.md`](../pre_process/README.md) | [`pre_process/README_CN.md`](../pre_process/README_CN.md)
