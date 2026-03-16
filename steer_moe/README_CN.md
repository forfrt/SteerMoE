# SteerMoE 核心实现

本文件夹包含 SteerMoE 架构的核心实现，包括模型类、逐层引导机制和工具函数。

## 概览

SteerMoE 是一种参数高效的语音转文本模型架构：
- **冻结**预训练的 LLM 解码器（如 Qwen、LLaMA）
- **仅训练**语音编码器的引导向量，由混合专家 (MoE) 路由器管理
- 使用**逐层引导**使冻结的编码器输出适配冻结的 LLM 解码器

### 架构流程

```
音频输入 → 冻结语音编码器 → SteerMoE（可训练）→ 投影层 → 冻结 LLM 解码器 → 文本输出
```

**关键组件**：
1. **冻结语音编码器**：预训练的 Whisper 或 Conformer 编码器（权重冻结）
2. **SteerMoE 模块**：可训练的引导向量 + MoE 路由器（逐层应用）
3. **投影层**：线性层，将编码器维度映射到 LLM 维度（可训练）
4. **冻结 LLM 解码器**：预训练的因果语言模型（权重冻结）

**参数效率**：仅约 180 万可训练参数（占总 85 亿参数的 0.02%）

## 📁 文件组织

```
steer_moe/
├── models.py                           # 主 SteerMoE 模型类
├── efficient_layer_wise_whisper.py    # Whisper 编码器 + 逐层引导
├── efficient_layer_wise_conformer.py  # Conformer 编码器 + 逐层引导
├── lora_model.py                       # LoRA 消融基线
├── utils.py                            # 数据整理器和工具函数
├── fireredasr_aed.py                  # 音频事件检测工具
├── conformer_module/                   # Conformer 实现
│   ├── adapter.py                     # 编码器输出适配器
│   ├── asr_feat.py                    # FBANK 特征提取
│   ├── conformer_encoder.py           # Conformer 编码器架构
│   ├── conformer_encoder_bak.py       # 备份版本
│   └── transformer_decoder.py         # Transformer 解码器
└── tokenizer/                          # 音频分词器
    └── glm4/                          # CosyVoice TTS 和语音分词模块
```

## 🏗️ 核心模型类

### `models.py`

包含所有 SteerMoE 模型架构。

#### 主要模型

| 类 | 说明 | 适用场景 |
|---|------|----------|
| `SteerMoEEfficientLayerWiseModel` | **主模型**，逐层引导 | 生产训练 |
| `SteerMoEEfficientLayerWiseModelLinear` | 仅线性基线（无引导） | 消融实验 |
| `SteerMoEEfficientLayerWiseModelForConformer` | Conformer 变体 | 中文 ASR、流式场景 |
| `SteerMoEHybridModel` | 混合连续提示方法 | 研究变体 |
| `SteerMoEModel` | 基础模型（编码器 → 对齐器 → 解码器） | 早期版本 |

#### 架构概览

```python
class SteerMoEEfficientLayerWiseModel(nn.Module):
    """
    音频 → 冻结编码器 → SteerMoE → 投影 → 冻结 LLM

    组件：
    - whisper_encoder: EfficientLayerWiseSteeringWhisperEncoder
    - llm_decoder: 冻结 LLM（Qwen、LLaMA 等）
    - prompt_proj: Linear(encoder_dim, llm_dim)
    """

    def __init__(
        self,
        whisper_encoder,              # 预初始化的引导编码器
        llm_decoder,                  # 冻结 LLM
        num_experts=8,                # MoE 专家数量
        max_prompt_tokens=2048,       # 最大音频序列长度
        use_adapter=True              # 使用线性投影
    ):
        # 可训练：引导向量、路由器、投影
        # 冻结：whisper_encoder.original_encoder、llm_decoder
```

#### 前向传播

```python
def forward(
    self,
    audio_waveform=None,      # 原始波形（推理）
    input_features=None,      # 预处理特征（训练）
    decoder_input_ids=None,   # 文本 token ID
    labels=None,              # 训练标签
    prompt_tokens_only=False, # 仅返回音频嵌入
    return_gating=False       # 返回 MoE 门控分数
):
    """
    训练：
        input_features: (batch, 128, 3000) mel 频谱图
        decoder_input_ids: (batch, seq_len) 文本 token
        labels: (batch, seq_len) 目标 token

    返回：
        output: ModelOutput，包含 loss 和 logits
    """
    # 1. 带引导的音频编码
    h_audio = self.whisper_encoder._forward_with_steering(input_features)
    # h_audio: (batch, audio_seq_len, 1280)

    # 2. 投影到 LLM 维度
    audio_prompts = self.prompt_proj(h_audio)
    # audio_prompts: (batch, audio_seq_len, 896)

    # 3. 获取文本嵌入
    text_embeds = self.llm_decoder.model.embed_tokens(decoder_input_ids)
    # text_embeds: (batch, text_seq_len, 896)

    # 4. 拼接
    inputs_embeds = torch.cat([audio_prompts, text_embeds], dim=1)

    # 5. 创建标签（用 -100 掩码音频 token）
    full_labels = torch.cat([
        labels.new_full((batch, audio_seq_len), -100),
        labels
    ], dim=1)

    # 6. LLM 前向传播
    output = self.llm_decoder(
        inputs_embeds=inputs_embeds,
        labels=full_labels
    )

    return output  # 包含 loss、logits
```

## 🔧 逐层引导实现

### `efficient_layer_wise_whisper.py`

为 Whisper 编码器实现高效的逐层引导。

#### 关键类：`EfficientLayerWiseSteeringWhisperEncoder`

```python
class EfficientLayerWiseSteeringWhisperEncoder(nn.Module):
    """
    用逐层引导包装 Whisper 编码器。
    所有层使用单一路由器（参数高效）。
    """

    def __init__(
        self,
        whisper_encoder_path,     # Whisper 模型路径
        num_experts=8,             # 每层专家数量
        steering_scale=0.1,        # 初始引导强度
        pooling_kernel_size=4,     # 可选下采样
        pooling_type=None,         # "avg" 或 "max"
        pooling_position=32        # 应用池化的层
    ):
        # 加载原始 Whisper 编码器（冻结）
        self.original_encoder = WhisperEncoder(whisper_encoder_path)

        # 可训练引导参数
        self.steering_vectors = nn.Parameter(
            torch.randn(num_layers, num_experts, feature_dim) * 0.01
        )
        self.router = nn.Linear(feature_dim, num_experts * num_layers)
        self.layer_scales = nn.Parameter(
            torch.ones(num_layers) * steering_scale
        )

        # 冻结原始编码器
        for param in self.original_encoder.parameters():
            param.requires_grad = False
```

#### 引导算法

```python
def _forward_with_steering(self, mel_features, return_gating=False):
    """
    前向传播时应用逐层引导。

    对于每层 l in 0..31：
        1. 通过原始冻结层
        2. 计算 MoE 门控分数
        3. 应用引导调整
        4. 继续到下一层
    """

    # 初始处理（卷积、位置嵌入）
    x = self.original_encoder.conv1(mel_features)
    x = self.original_encoder.conv2(x)
    x = x.permute(0, 2, 1)
    x = x + self.original_encoder.embed_positions.weight[:x.size(1)]
    x = x * self.original_encoder.embed_scale

    # 带引导的逐层处理
    for layer_idx, layer in enumerate(self.original_encoder.layers):
        # 1. 原始层前向
        layer_output = layer(x)[0]  # (batch, seq, feature_dim)

        # 2. 计算门控分数
        router_output = self.router(layer_output)  # (batch, seq, num_experts*num_layers)

        # 提取该层的专家
        start = layer_idx * self.num_experts
        end = (layer_idx + 1) * self.num_experts
        gating_logits = router_output[:, :, start:end]  # (batch, seq, num_experts)

        # 3. Softmax 计算专家权重
        gating_scores = F.softmax(gating_logits, dim=-1)

        # 4. 引导向量的加权组合
        steering_vectors = self.steering_vectors[layer_idx]  # (num_experts, feature_dim)
        steering_adjustment = torch.einsum(
            'bte,ef->btf',
            gating_scores,         # (batch, seq, num_experts)
            steering_vectors       # (num_experts, feature_dim)
        )  # (batch, seq, feature_dim)

        # 5. 应用层特定缩放的引导
        layer_scale = self.layer_scales[layer_idx]
        x = layer_output + layer_scale * steering_adjustment

        # 可选：在特定层应用池化
        if layer_idx + 1 == self.pooling_position and self.pooling_layer:
            x = x.permute(0, 2, 1)  # (batch, feature_dim, seq)
            x = self.pooling_layer(x)
            x = x.permute(0, 2, 1)  # (batch, seq', feature_dim)

    # 最终层归一化
    x = self.original_encoder.layer_norm(x)

    return x
```

#### 为什么这样设计？

**效率**：
- 单路由器将参数从 `num_layers × (feature_dim × num_experts)` 减少到 `feature_dim × (num_experts × num_layers)`
- 示例：32 层，8 专家，1280 维
  - 多路由器：32 × (1280 × 8) = 327,680 × 32 = 1050 万参数
  - 单路由器：1280 × (8 × 32) = 1280 × 256 = 32.7 万参数
  - **32 倍参数减少！**

**灵活性**：
- 每层仍有自己的专家集
- 路由器学习为每层不同地路由
- 层缩放允许逐层强度调整

### `efficient_layer_wise_conformer.py`

Conformer 编码器的类似实现。

**关键差异**：
1. 使用 Conformer 架构而非 Whisper
2. 不同的特征提取（FBANK + CMVN）
3. 流式友好设计
4. 更适合亚洲语言

## 🛠️ 工具函数

### `utils.py`

包含数据整理器和辅助函数。

#### `DataCollatorSpeechSeqSeqWithPadding`

SteerMoE 训练的主数据整理器。

```python
@dataclass
class DataCollatorSpeechSeqSeqWithPadding:
    """
    整理预处理的音频-文本对用于训练。

    处理：
    - 音频特征填充
    - 带文本提示的文本分词
    - 音频 token 的标签掩码
    """

    feature_extractor: Any        # Whisper/Conformer 特征提取器
    tokenizer: Any                # LLM 分词器
    textual_prompt: str = None    # 例如 "Transcribe: "
    max_length: int = 448         # 最大文本序列长度
    audio_column: str = "input_features"
    text_column: str = "labels"
```

#### `DataCollatorSpeechSeqSeqWithPaddingForConformer`

Conformer 编码器的变体，处理不同的特征。

#### 辅助函数

```python
def load_balancing_loss(gating_scores):
    """
    计算辅助损失以鼓励均匀的专家使用。

    参数：
        gating_scores: (batch, seq_len, num_experts)

    返回：
        loss: 标量张量（与均匀分布的 KL 散度）
    """
    expert_usage = gating_scores.mean(dim=(0, 1))  # 每个专家的平均使用率
    num_experts = gating_scores.size(-1)
    target = torch.full_like(expert_usage, 1.0 / num_experts)  # 均匀目标
    loss = F.kl_div(expert_usage.log(), target, reduction="batchmean")
    return loss
```

## 🔬 消融研究

### LoRA 基线（`lora_model.py`）

为与 SteerMoE 对比，提供了基于 LoRA 的基线：
- 冻结 LLM 解码器（与 SteerMoE 相同）
- 对编码器层应用 LoRA 适配器而非引导向量
- 使用类似接口以便公平比较

**关键差异**：
- **SteerMoE**：带 MoE 路由的加性引导向量
- **LoRA**：应用于注意力和 FFN 层的低秩适配矩阵

## 📊 关键设计决策

### 为什么单路由器？

1. **参数效率**：相比多路由器减少 32 倍（32.7 万 vs 1050 万参数）
2. **泛化能力**：共享路由器学习通用路由模式
3. **灵活性**：仍允许通过切片进行层特定的专家选择

### 为什么逐层引导？

1. **细粒度控制**：每层可独立适配
2. **保留编码器结构**：原始编码器保持冻结
3. **可解释性**：可分析哪些专家在哪些层激活

### 为什么冻结其他部分？

1. **参数效率**：仅训练最小适配组件
2. **知识保留**：利用预训练表示
3. **训练稳定性**：冻结组件提供稳定梯度
4. **迁移学习**：可用最少数据适配新任务

## 🔗 相关文档

- 训练脚本：[`scripts/README.md`](../scripts/README.md) | [`scripts/README_CN.md`](../scripts/README_CN.md)
- 配置文件：[`configs/README.md`](../configs/README.md) | [`configs/README_CN.md`](../configs/README_CN.md)
- 预处理：[`pre_process/README.md`](../pre_process/README.md) | [`pre_process/README_CN.md`](../pre_process/README_CN.md)
