# SteerMoE: 基于混合专家引导模块的高效音频-语言对齐

[![Paper](https://img.shields.io/badge/Interspeech-2026-blue)](papers/Interspeech_26_SteerMOE/camera_ready.pdf)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README_EN.md) | 中文

**SteerMoE 实现了强大的音频-语言模型，能够理解语音和文本，同时保留大语言模型的完整推理能力。**

与传统方法为了音频处理而牺牲语言理解不同，我们的方法保持 LLM 完全冻结，确保您的音频-语言模型保持复杂的文本推理、推理和生成能力——同时在音频理解任务上实现强大的性能。

## 🎯 我们的成果

### 音频 + 语言理解与完整 LLM 推理

我们的模型可以：
- ✅ **转录语音**，具有竞争力的准确率（LibriSpeech 上 Conformer 达到 2.42% WER）
- ✅ **回答关于音频的问题**（Clotho-AQA 上 52.35% 准确率）
- ✅ **使用 LLM 强大的推理能力对音频内容进行推理**
- ✅ **保持完整的文本能力**（冻结的 LLM 保留所有语言理解）
- ✅ **支持多种语言**（英语、中文等）

### 关键创新：冻结架构

**问题**：传统音频-语言模型微调 LLM，这会降低其复杂的语言推理能力。

**我们的解决方案**：保持音频编码器和语言解码器完全冻结。仅训练一个轻量级对齐模块（约 180 万参数）来桥接两种模态。

**结果**：两全其美——强大的音频理解 + 保留的 LLM 推理。

## 📊 性能亮点

### 英文 ASR（LibriSpeech test-clean）

| 模型 | WER ↓ | 文本推理 | 可训练参数 |
|------|-------|----------|-----------|
| Whisper-large-v3（冻结） | 2.7% | ❌ 无 LLM | 1550M |
| Encoder-LoRA（调优编码器） | 2.51% | ✅ 保留 | 15.5M |
| SteerMoE (W7B) | 5.69% | ✅ **完全保留** | **1.8M** |
| SteerMoE (C3B) | 3.26% | ✅ **完全保留** | **1.8M** |
| **SteerMoE (C7B)** | **2.42%** | ✅ **完全保留** | **1.8M** |

### 中文 ASR（AISHELL-2）

| 模型 | 测试 CER ↓ | 可训练参数 |
|------|-----------|-----------|
| Whisper-large-v3（冻结） | 4.96% | 1550M |
| SteerMoE (W7B) | 5.96% | **1.8M** |
| SteerMoE (C3B) | 3.44% | **1.8M** |
| **SteerMoE (C7B)** | **2.50%** | **1.8M** |

### 音频问答（Clotho-AQA）

| 模型 | 准确率 ↑ | 总参数 | 可训练参数 |
|------|---------|--------|-----------|
| Kimi-Audio | 71.24% | 9.77B | 未公开（LLM 微调） |
| Step-Audio-Chat | 45.84% | 130B | 未公开（LLM 微调） |
| **SteerMoE (W7B)** | **52.35%** | 7B+1.5B | **1.8M** |
| SteerMoE (C3B) | 46.24% | 3B+1.5B | **1.8M** |
| SteerMoE (C7B) | 49.06% | 7B+1.5B | **1.8M** |

**关键见解**：我们以**完全保留的 LLM 推理**和仅 **180 万可训练参数**（约占完整模型大小的 0.02%）实现了**有竞争力的音频性能**。SteerMoE 在仅训练 180 万参数且解码器冻结的情况下，比 1300 亿参数的 Step-Audio-Chat 模型高出 6 个百分点以上。

## 💡 为什么这很重要

### 保留的语言能力

您的音频-语言模型保持 LLM 的所有能力：

```python
# 在音频任务上训练后，LLM 仍然擅长纯文本：

# 复杂推理（保留）
prompt = "如果 Alice 的苹果数量是 Bob 的两倍，Bob 有 3 个苹果，
          考虑 15% 的税，Alice 以每个 2 美元购买她的苹果需要支付多少？"
model.generate(prompt)  # ✅ 完美运行 - LLM 推理完好

# 代码生成（保留）
prompt = "编写一个 Python 函数来实现二分查找"
model.generate(prompt)  # ✅ 仍然生成正确的代码

# 音频理解（新获得）
audio = load_audio("speech.wav")
prompt = "转录并总结要点："
model.generate(audio, prompt)  # ✅ 理解音频 + 对内容进行推理
```

**为什么这很重要**：
- 为音频和文本任务部署一个模型
- 不妥协语言理解质量
- LLM 的常识推理有助于音频理解
- 安全部署到生产环境（无意外行为变化）

## 🔬 SteerMoE 技术

### 如何实现：基于混合专家的逐层引导

为了在不微调任何一方的情况下桥接冻结的音频编码器和冻结的 LLM，我们引入了 **SteerMoE**——一个轻量级、可训练的对齐模块，动态地将音频特征"引导"到 LLM 的表示空间。

#### 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│  音频输入（例如，"你好世界"语音）                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │   冻结音频编码器               │  ← Whisper/Conformer
         │   (15 亿参数，冻结)            │     无训练
         └───────────────┬───────────────┘
                         │ 音频特征
                         ▼
         ┌───────────────────────────────┐
         │      SteerMoE 对齐器          │  ← 我们的创新
         │   逐层引导 + MoE              │     约 180 万参数
         │   (仅可训练部分)               │     动态适配
         └───────────────┬───────────────┘
                         │ 对齐特征
                         ▼
         ┌───────────────────────────────┐
         │    线性投影                    │  ← 简单适配器
         │   (1280 → 896 维度)           │     约 100 万参数
         └───────────────┬───────────────┘
                         │ LLM 兼容嵌入
                         ▼
         ┌───────────────────────────────┐
         │   冻结语言解码器               │  ← Qwen/LLaMA
         │   (70 亿参数，冻结)            │     无训练
         │   推理保留 ✓                  │     所有能力完好
         └───────────────┬───────────────┘
                         │
                         ▼
         文本输出："你好世界"（+ 推理/问答等）
```

#### 核心思想：逐层动态引导

SteerMoE 不是学习单一的静态转换，而是根据输入内容在**每个编码器层应用自适应调整**：

```python
# 对于每个音频编码器层 l：
for layer_idx in range(num_layers):
    # 1. 通过冻结的编码器层处理
    h_l = frozen_encoder_layer[layer_idx](h_l_minus_1)

    # 2. MoE 路由器决定使用哪些专家（取决于音频内容）
    expert_weights = Router(h_l)  # 对语音/音乐/噪音等不同

    # 3. 应用动态引导调整
    steering = Σ expert_weights[k] * steering_vectors[layer_idx, k]

    # 4. 调整特征
    h_l = h_l + layer_scale[layer_idx] * steering
```

**为什么有效**：
- 🎯 **内容自适应**：路由器学习为不同音频类型选择不同专家
- 🔀 **层特定**：早期层关注声学特征，后期层关注语义对齐
- 📊 **高效**：所有层使用单一路由器（比朴素 MoE 少 32 倍参数）
- 🎚️ **可控**：层缩放允许每层细粒度调整强度

#### 什么使其成为"混合专家"？

每层有**多个专家引导向量**（通常为 8 个）：

```
层 0:  [专家_0: 声学模式] [专家_1: 噪音处理] [专家_2: 音乐] ...
层 1:  [专家_0: 音素特征] [专家_1: 音高变化] ...
...
层 31: [专家_0: 语义概念] [专家_1: 上下文对齐] ...
```

**路由器网络**学习：
- 为清晰语音选择**专家_0**
- 为嘈杂音频选择**专家_1**
- 为背景音乐选择**专家_2**
- 为复杂音频场景混合专家

这种**动态专业化**是 SteerMoE 优于静态适配器的原因。

### 技术优势

#### 1. 参数效率（1000 倍减少）

传统微调：
```
可训练：15 亿（音频编码器）+ 70 亿（LLM）= 85 亿参数
训练时间：约 500 GPU 小时
GPU 显存：8× A100 80GB
```

SteerMoE：
```
可训练：180 万参数（仅引导 + 投影）
训练时间：约 10 GPU 小时
GPU 显存：1× A100 40GB
风险：最小（LLM 行为不变）
```

**180 万参数的分解**：
- 引导向量：`32 层 × 8 专家 × 1280 维` = 32.7 万参数
- 路由器网络：`1280 维 → (8×32)` = 32.7 万参数
- 层缩放：`32` = 32 参数
- 线性投影：`1280 → 896` = 110 万参数
- **总计：约 180 万参数（完整模型的 0.02%）**

#### 2. 保留的泛化能力

因为音频编码器保持冻结：
- ✅ 保持 Whisper 对口音、噪音等的鲁棒性
- ✅ 不会过拟合到您的特定数据集
- ✅ 在域外音频上工作而不降级

因为 LLM 保持冻结：
- ✅ 所有文本推理能力保留
- ✅ 无灾难性遗忘
- ✅ 安全用于生产部署

#### 3. 快速迭代和灵活性

- 🔄 尝试不同的音频编码器（Whisper、Conformer 等）
- 🔄 交换 LLM 主干（Qwen、LLaMA、Mistral 等）
- 🔄 在几小时而非几周内训练新语言
- 🔄 轻松适应新任务（ASR → 问答 → 字幕）

## 🏗️ 架构变体

我们提供多种模型配置：

| 编码器 | 最适合 | 语言 | 训练时间 | 可训练参数 |
|--------|--------|------|----------|-----------|
| **Whisper-large-v3** | 通用 ASR、英语 | 90+ 种语言 | 约 10 小时 | 180 万 |
| **Conformer** | 中文/亚洲语言、流式 | 中文、日语、韩语 | 约 12 小时 | 180 万 |

### 模型架构细节

- **SteerMoE (Whisper/Conformer)**：使用带 MoE 路由的逐层引导。可训练参数：**180 万**（引导向量 + 路由器 + 投影）。
- **Encoder-LoRA（基线）**：在 Whisper 编码器层上使用 LoRA 适配器的基线方法。可训练参数：**1550 万**（注意力和 FFN 模块上的 LoRA 适配器）。保留 LLM 推理但使用约 8.6 倍于 SteerMoE 的参数。

两种 SteerMoE 变体使用**相同的技术**，只是使用不同的音频编码器。

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/SteerMoE.git
cd SteerMoE

# 创建环境
conda create -n steermoe python=3.10
conda activate steermoe
pip install -r requirements.txt

# 下载预训练模型
# Whisper: openai/whisper-large-v3
# LLM: Qwen/Qwen2.5-7B-Instruct
```

### 1. 预处理数据集

```bash
# 英文（LibriSpeech）
python pre_process/pre_process_librispeech.py \
  --audio_dir /path/to/LibriSpeech/train-clean-100 \
  --output_dir /path/to/processed_librispeech \
  --whisper_model /path/to/whisper-large-v3 \
  --llm_tokenizer /path/to/Qwen2.5-7B-Instruct

# 中文（AISHELL-2）
python pre_process/pre_process_aishell.py \
  --audio_dir /path/to/aishell2/wav \
  --trans_file /path/to/aishell2/trans.txt \
  --output_dir /path/to/processed_aishell2
```

查看 [`pre_process/README_CN.md`](pre_process/README_CN.md) 了解其他数据集。

### 2. 配置训练

编辑 `configs/layer_wise_whisper_qwen7b_libri_train.yaml`：

```yaml
# 音频编码器（冻结）
whisper_encoder:
  model_path: "/path/to/whisper-large-v3"

# 语言解码器（冻结）
llm_decoder:
  model_name: "/path/to/Qwen2.5-7B-Instruct"

# SteerMoE 设置（可训练）
steering:
  num_experts: 8
  steering_scale: 0.1
  steering_learning_rate: 1e-2  # 引导的更高学习率

# 数据集
parquet_dirs:
  - "/path/to/processed_librispeech/train.clean.100/"

# 任务提示
textual_prompt: "please transcribe the audio content into text: "
```

### 3. 训练 SteerMoE

```bash
# 单 GPU
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --mode train

# 多 GPU（推荐）
deepspeed --num_gpus=4 scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --deepspeed_config configs/stage2_simple.json \
  --mode train
```

在 LibriSpeech-100h 上训练在 4× A100 GPU 上需要约 10 小时。

### 4. 评估

```bash
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_test.yaml \
  --mode eval \
  --model_path results/steermoe_checkpoint/final
```

### 5. 用于推理

```python
from transformers import AutoTokenizer
from steer_moe.models import SteerMoEEfficientLayerWiseModel
import torch

# 加载模型
model = SteerMoEEfficientLayerWiseModel.load(
    checkpoint_path="results/steermoe_checkpoint/final"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 加载和预处理音频
audio_features = preprocess_audio("speech.wav")  # (1, 128, T)

# 转录
prompt = tokenizer("转录：", return_tensors="pt").input_ids
output_ids = model.generate(
    input_features=audio_features,
    decoder_input_ids=prompt,
    max_new_tokens=256
)
transcription = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(transcription)

# 问答（同一模型！）
prompt = tokenizer("音频中表达了什么情感？", return_tensors="pt").input_ids
output_ids = model.generate(input_features=audio_features, decoder_input_ids=prompt)
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(answer)  # ✅ 使用 LLM 推理分析情感
```

## 📖 文档

每个组件的综合指南：

- **[`configs/README_CN.md`](configs/README_CN.md)** - 配置文件和超参数
- **[`pre_process/README_CN.md`](pre_process/README_CN.md)** - ASR、问答等的数据集预处理
- **[`scripts/README_CN.md`](scripts/README_CN.md)** - 训练、评估和分析脚本
- **[`steer_moe/README_CN.md`](steer_moe/README_CN.md)** - 核心模型实现细节

## 🔬 消融研究

我们通过全面的消融验证 SteerMoE 的设计：

### SteerMoE 消融：专家数量

| 专家数 | WER ↓ | 可训练参数 |
|--------|-------|-----------|
| 16 | 2.43% | 250 万 |
| **8（默认）** | **2.42%** | **180 万** |
| 4 | 3.10% | 150 万 |
| 2 | 6.22% | 130 万 |
| 静态适配器（无 MoE） | >100% | 110 万 |

**结论**：从 SteerMoE（2 个专家）到静态适配器（无 MoE）的显著性能崩溃证实了动态引导是必不可少的——没有内容自适应路由，静态投影无法弥合音频-文本表示差距，产生基本上无法理解的输出。SteerMoE 表明 8 个专家提供了最佳权衡：虽然 16 个专家仅提供 0.01% WER 的边际改进，但可训练参数增加 39% 使得 8 个专家成为最高效的配置。

## 📁 项目结构

```
SteerMoE/
├── configs/              # 训练配置
│   ├── layer_wise_whisper_qwen7b_libri_train.yaml
│   ├── layer_wise_conformer_qwen7b_aishell_train.yaml
│   └── README.md / README_CN.md
├── pre_process/          # 数据集预处理
│   ├── pre_process_librispeech.py
│   ├── pre_process_aishell.py
│   ├── pre_process_clothoaqa.py
│   └── README.md / README_CN.md
├── scripts/              # 训练和评估
│   ├── train_layer_wise.py              # 主训练（Whisper）
│   ├── train_layer_wise_conformer.py    # 主训练（Conformer）
│   ├── train_layer_wise_linear_whisper.py  # 消融基线
│   ├── cer.py, wer.py                    # 评估指标
│   └── README.md / README_CN.md
├── steer_moe/            # 核心实现
│   ├── models.py                         # SteerMoE 模型类
│   ├── efficient_layer_wise_whisper.py  # Whisper + 引导
│   ├── efficient_layer_wise_conformer.py # Conformer + 引导
│   ├── utils.py                          # 数据整理器
│   └── README.md / README_CN.md
├── papers/               # 研究论文
│   ├── Interspeech_26_SteerMOE/         # Interspeech 2026 投稿
│   └── ICASSP_SteerMoE/                 # ICASSP 2025 投稿
├── results/              # 训练输出
└── README.md / README_CN.md  # 本文件
```

## 🎓 研究背景

### 传统音频-LLM 方法的问题

大多数音频-语言模型使用以下方法之一：

**方法 1：微调整个 LLM**
```
音频 → 编码器 → [微调的 LLM] → 输出
                   ⚠️ 70 亿参数训练
                   ⚠️ 语言推理降级
                   ⚠️ 昂贵且训练缓慢
```

**方法 2：基于适配器（简单投影）**
```
音频 → 编码器 → [线性] → [冻结 LLM] → 输出
                   ✅ LLM 保留
                   ⚠️ 有限的音频理解
                   ⚠️ 静态转换
```

**我们的方法：SteerMoE**
```
音频 → 编码器 → [SteerMoE: 动态引导] → [冻结 LLM] → 输出
                   ✅ LLM 完全保留
                   ✅ 强大的音频理解
                   ✅ 内容自适应转换
                   ✅ 仅训练 180 万参数
```

### 我们研究的关键见解

1. **冻结优于微调**：冻结的 LLM 保留推理，冻结的编码器保留鲁棒性
2. **动态优于静态**：MoE 路由比固定投影更好地适应不同音频类型
3. **逐层至关重要**：不同的编码器层需要不同的对齐策略
4. **效率是可实现的**：单路由器比朴素多路由器 MoE 减少 32 倍参数

查看我们的论文（[`papers/Interspeech_26_SteerMOE/camera_ready.pdf`](papers/Interspeech_26_SteerMOE/camera_ready.pdf)）了解详细分析和更多结果。

## 💻 硬件要求

### 训练

| 配置 | GPU | 批大小 | 训练时间（LibriSpeech-100h） |
|------|-----|--------|----------------------------|
| 最小 | 1× A100 40GB | 1-2 | 约 40 小时 |
| 推荐 | 4× A100 40GB | 每 GPU 4 | 约 10 小时 |
| 大规模 | 8× A100 80GB | 每 GPU 8 | 约 5 小时 |

### 推理

| 模型大小 | GPU 显存 | Token/秒 |
|---------|---------|----------|
| Qwen-7B + Whisper | 16GB (FP16) | 约 50 |
| Qwen-3B + Whisper | 8GB (FP16) | 约 100 |

## 📧 联系与支持

**作者**：
- 冯瑞涛 - [GitHub: @forfrt](https://github.com/forfrt) - ruitaofeng@outlook.com
- 张碧玺 - [GitHub: @zbxforward](https://github.com/zbxforward) - bixizhang@hku.hk
- 梁晟 - shengliang@outlook.com
- 袁铮（通讯作者）- zheng.yuan@univ-amu.fr

**获取帮助**：
- 🐛 **错误报告**：[GitHub Issues](https://github.com/forfrt/SteerMoE/issues)
- 💬 **问题**：[GitHub Discussions](https://github.com/forfrt/SteerMoE/discussions)

**论文**：查看 [`papers/Interspeech_26_SteerMOE/camera_ready.pdf`](papers/Interspeech_26_SteerMOE/camera_ready.pdf) 了解完整的 Interspeech 2026 投稿，包含详细方法和额外实验。

## 📄 许可证

本项目根据 MIT 许可证授权 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本工作建立在优秀的开源项目之上：

- **[Whisper](https://github.com/openai/whisper)**（OpenAI）- 鲁棒的语音识别
- **[Qwen](https://github.com/QwenLM/Qwen)**（阿里巴巴）- 强大的多语言 LLM
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)**（微软）- 高效的分布式训练
- **[Transformers](https://github.com/huggingface/transformers)**（HuggingFace）- 模型实现和训练工具

我们还感谢研究社区提供的数据集：
- LibriSpeech、AISHELL-2、Clotho-AQA 和其他基准数据集
- 开源音频处理库（librosa、soundfile、torchaudio）

## 🌟 Star 历史

如果您觉得 SteerMoE 有用，请考虑给它一个 star！⭐

---

**由 SteerMoE 团队用 ❤️ 构建。有问题？提出 issue 或讨论！**
