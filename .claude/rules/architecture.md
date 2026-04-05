# SteerMoE 系统架构

> SteerMoE: Efficient Audio-Language Alignment with a Mixture-of-Experts Steering Module
> 发表于 Interspeech 2026

## 核心思想

通过在冻结的 audio encoder（Whisper/Conformer）的每一层 attention 输出上施加内容自适应的残差偏移（steering），使音频表征对齐到冻结的 LLM decoder（Qwen），仅训练少量参数（Conformer+Qwen7B 约 1.8M，Whisper+Qwen7B 约 5.2M）。

## 端到端数据流

```
原始音频 (.flac/.wav)
    │
    ▼
离线预处理 (pre_process/)
    │  Whisper路径: WhisperFeatureExtractor → Mel (128, T) @50Hz
    │  Conformer路径: 保存 int16 raw audio → 运行时 ASRFeatExtractor → FBANK (T, 80) @100Hz + CMVN
    ▼
HuggingFace Dataset (Parquet)
{input_features, labels, input_length, attention_mask, text [, text_prompt, sample_rate]}
    │
    ▼
DataCollator (steer_moe/utils.py)
    │  Whisper: feature_extractor.pad() → (B, 128, T_max)
    │  Conformer: ASRFeatExtractor(raw_int16) → (B, T_max, 80) + input_lengths
    │  构造 decoder_input_ids = [prompt_tokens | label_tokens]
    │  构造 labels = [-100 × prompt_len | actual_token_ids]
    ▼
STAGE 1: 带 Layer-Wise Steering 的音频编码 (冻结 encoder + 可训练 steering)
    │
    │  For layer_idx = 0..N-1:
    │    layer_output = FrozenLayer[layer_idx](x)
    │    gating_scores = softmax(SingleRouter(layer_output)[:, :, slice])  → (B, T, num_experts)
    │    steering_adj = einsum('bte,ef->btf', gating_scores, steering_vectors[layer_idx])
    │    x = layer_output + layer_scales[layer_idx] * steering_adj
    │
    │  Pooling: MaxPool1d/AvgPool1d → 时间维度下采样 (Whisper kernel=4, Conformer kernel=2)
    │  LayerNorm (仅 Whisper；Conformer 无此步骤)
    ▼
h_audio: (B, T_audio, encoder_dim)   [Whisper: 1280, Conformer: 512]
    │
    ▼
STAGE 2: 维度投影 (可训练 Linear)
    │  prompt_proj: encoder_dim → llm_hidden_dim
    │  Whisper→Qwen7B: 1280→3584, Conformer→Qwen3B: 512→2048, Conformer→Qwen7B: 512→3584
    ▼
audio_prompts: (B, T_audio, LLM_dim)
    │
    ▼
STAGE 3: 多模态序列拼接
    │  text_embeds = frozen_llm.embed_tokens(decoder_input_ids)
    │  inputs_embeds = cat([audio_prompts, text_embeds], dim=1)
    │  full_labels = cat([-100 × T_audio, labels], dim=1)
    ▼
STAGE 4: 冻结 LLM Decoder 前向
    │  output = frozen_llm(inputs_embeds, attention_mask, labels=full_labels)
    ▼
STAGE 5: Loss (Cross-Entropy, 仅文本位置)
    │  loss = output.loss   # -100 位置自动忽略
    ▼
反向传播仅流经:
    ✅ steering_vectors    (N_layers × num_experts × feature_dim)
    ✅ router              (Linear: feature_dim → num_experts × N_layers)
    ✅ layer_scales        (N_layers 个标量)
    ✅ prompt_proj         (Linear: encoder_dim → llm_hidden_dim)
    ❌ 所有冻结的 encoder 权重
    ❌ 所有冻结的 LLM decoder 权重
```

## 可训练参数统计

参数量随 LLM 规模和编码器类型变化：

### Whisper + Qwen2.5-7B (32 layers, feature_dim=1280, hidden_size=3584, 8 experts)

| 组件 | 形状 | 参数量 |
|------|------|--------|
| `steering_vectors` | (32, 8, 1280) | 327,680 |
| `router` (Linear) | 1280 → 256 | 327,936 |
| `layer_scales` | (32,) | 32 |
| `prompt_proj` (Linear) | 1280 → 3584 | 4,591,104 |
| **总计** | | **~5.2M** |

### Conformer (FireRedASR-AED-L) + Qwen2.5-7B (feature_dim=512, 8 experts)

| 组件 | 形状 | 参数量 |
|------|------|--------|
| `steering_vectors` | (N, 8, 512) | N × 4,096 |
| `router` (Linear) | 512 → 8N | 512 × 8N + 8N |
| `layer_scales` | (N,) | N |
| `prompt_proj` (Linear) | 512 → 3584 | 1,838,592 |
| **总计** | | **~1.8M** (prompt_proj 占绝大部分) |

注意: N = Conformer 编码器层数，由 FireRedASR-AED-L 模型结构决定。README 中的 "1.8M" 主要由 prompt_proj 贡献。

冻结参数: ~1.55B (Whisper) + ~7B (Qwen) = ~8.55B

## 项目目录结构

```
SteerMoE/
├── steer_moe/                       # 核心模型实现
│   ├── models.py                    # 所有模型类定义
│   ├── efficient_layer_wise_whisper.py   # Whisper + SteerMoE 编码器
│   ├── efficient_layer_wise_conformer.py # Conformer + SteerMoE 编码器 + 完整模型
│   ├── fireredasr_aed.py            # FireRed ASR AED 模型工具
│   ├── utils.py                     # DataCollator、load_balancing_loss
│   ├── conformer_module/
│   │   └── asr_feat.py              # FBANK + CMVN 特征提取
│   └── tokenizer/whisper_Lv3/
│       └── whisper.py               # WhisperEncoder 封装
├── scripts/                         # 训练与评估脚本
│   ├── train_layer_wise.py          # Whisper + SteerMoE 训练入口
│   ├── train_layer_wise_conformer.py # Conformer + SteerMoE 训练入口
│   ├── train_layer_wise_conformer_clothoaqa.py # Conformer + ClothoAQA 训练
│   ├── train_layer_wise_prompt.py   # Whisper + 逐样本 prompt (QA 任务)
│   ├── train_layer_wise_linear_whisper.py  # 消融实验：仅 Linear 无 MoE (Whisper)
│   ├── train_conformer_linear.py    # 消融实验：仅 Linear 无 MoE (Conformer)
│   ├── train_lora.py / eval_lora.py # LoRA 基线实验
│   ├── steermoe_train.sh            # DeepSpeed 多GPU启动脚本
│   ├── cer.py                       # 字符错误率 (CER) 指标
│   └── wer.py                       # 词错误率 (WER) 指标
├── pre_process/                     # 数据预处理
│   ├── preprocess_utils.py          # 核心工具: prepare_dataset / prepare_dataset_for_conformer
│   ├── modules.py                   # DataCollator 和 ASREvaluator
│   ├── pre_process_librispeech.py   # LibriSpeech (Whisper)
│   ├── pre_process_librispeech_for_conformer.py  # LibriSpeech (Conformer)
│   ├── pre_process_aishell.py       # AISHELL-1 训练集
│   ├── pre_process_aishell_test.py  # AISHELL-1 测试集
│   ├── pre_process_clothoaqa*.py    # ClothoAQA 音频问答
│   ├── pre_process_mmau.py          # MMAU 多任务音频理解
│   └── pre_process_openaudiobench.py # OpenAudioBench 评估
├── configs/                         # YAML 配置文件 + DeepSpeed JSON
├── cross_modal_steer/               # 研究探索（路由器架构对比）
├── docs/                            # 架构分析文档
├── papers/                          # 论文 LaTeX 源码
│   ├── Interspeech_26_SteerMOE/
│   └── ICASSP_SteerMoE/
├── results/                         # 训练输出（检查点）
└── train.sh                         # 顶层训练入口
```

## 两条编码器路径

| 特性 | Whisper 路径 | Conformer 路径 |
|------|-------------|---------------|
| 编码器 | openai/whisper-large-v3 | FireRedASR-AED-L |
| 特征维度 | 1280 | 512 |
| 层数 | 32 | 视模型配置 |
| 特征提取 | WhisperFeatureExtractor → Mel (128, T) | ASRFeatExtractor → FBANK (T, 80) + CMVN |
| 预处理输出 | float32 Mel spectrogram | int16 raw audio (运行时提取 FBANK) |
| Pooling 默认 | MaxPool1d(kernel=4), 代码中硬编码 | 由参数控制，默认 MaxPool1d(kernel=2) |
| 核心类 | `EfficientLayerWiseSteeringWhisperEncoder` | `EfficientLayerWiseSteeringConformerEncoder` |
| 训练脚本 | `train_layer_wise.py` | `train_layer_wise_conformer.py` |
