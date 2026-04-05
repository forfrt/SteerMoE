# SteerMoE 核心模块

## 模型模块 (`steer_moe/`)

### `models.py` — 模型类定义

| 类名 | 状态 | 说明 |
|------|------|------|
| `SteerMoEEfficientLayerWiseModel` | **主力 (Whisper)** | 单 router + Whisper 编码器 + 冻结 LLM |
| `SteerMoEEfficientLayerWiseModelForConformer` | **主力 (Conformer)** | 定义在 `efficient_layer_wise_conformer.py` |
| `SteerMoEEfficientLayerWiseModelLinear` | 消融 (Whisper) | 仅 Linear 投影无 steering，消融实验基线 |
| `SteerMoEModel` | 遗留 | 简单版: Whisper → aligner → LLM |
| `SteerMoEHybridModel` | 遗留 | 混合版: 可选投影层 |
| `SteerMoELayerWiseModel` | 遗留 | 多 router 版 (每层一个 router) |

### `efficient_layer_wise_whisper.py` — Whisper 编码器 + Steering

**`EfficientLayerWiseSteeringWhisperEncoder`**:
- 加载冻结的 WhisperEncoder
- 创建可训练参数: `steering_vectors`, `router`, `layer_scales`
- `_forward_with_steering()`: 在 Whisper 的 32 层中逐层施加 steering 残差
- 单个 router 同时输出所有层的 gating logits (`feature_dim → num_experts × num_layers`)
- 可选 pooling: MaxPool1d/AvgPool1d，在指定层后执行

**核心公式**:
```
x[l] = FrozenLayer[l](x[l-1]) + layer_scale[l] * Σ_k(gating_score[l,k] * steering_vector[l,k])
```

### `efficient_layer_wise_conformer.py` — Conformer 编码器 + Steering + 完整模型

**`LoadConformerEncoder`**: 从 `.pth.tar` 加载 FireRedASR-AED-L Conformer 权重。

**`EfficientLayerWiseSteeringConformerEncoder`**: 同 Whisper 版本逻辑，但:
- 处理 Conformer 的 `input_preprocessor` (卷积下采样 + 线性)
- 处理 `positional_encoding` → `pos_emb`
- 每层 forward 传入 `(enc_output, pos_emb, slf_attn_mask, pad_mask)`

**`SteerMoEEfficientLayerWiseModelForConformer`**: 完整模型类
- `forward()`: steering 编码 → `prompt_proj` 投影 → 拼接音频+文本 → 冻结 LLM
- `generate()`: 自回归推理，先编码音频，再与文本 prompt embedding 拼接后送入 LLM 生成

### `fireredasr_aed.py` — FireRed ASR AED 辅助

加载和管理 FireRedASR AED 模型的工具函数。

### `utils.py` — 数据处理工具

**数据整理器 (DataCollator)**:

| 类 | 编码器 | 功能 |
|----|--------|------|
| `DataCollatorSpeechSeqSeqWithPadding` | Whisper | pad mel → 构造 labels (前缀 -100) → 追加 EOS |
| `DataCollatorSpeechSeqSeqWithPaddingPrompt` | Whisper (QA) | 同上，但从每个样本的 `text_prompt` 字段读取提示，而非全局 textual_prompt |
| `DataCollatorSpeechSeqSeqWithPaddingForConformer` | Conformer | raw int16 → ASRFeatExtractor → FBANK → 同上 |
| `DataCollatorSpeechSeqSeqWithPaddingLegacy` | Whisper (遗留) | 兼容旧版 Whisper 训练代码的简单 collator |

**标签构造逻辑** (两者相同):
1. 从 dataset 读取 `labels` (tokenizer output IDs)
2. 去除 padding 和特殊 token
3. 追加 `eos_token_id`
4. 前缀拼接 tokenized `textual_prompt` → `decoder_input_ids`
5. Labels: `[-100 × prompt_len, actual_token_ids]` (prompt 部分不计算 loss)

**其他**:
- `load_balancing_loss(gating_scores)`: 专家负载均衡损失 (KL from uniform，当前未在训练循环中使用)

### `conformer_module/asr_feat.py` — FBANK 特征提取

- **`ASRFeatExtractor`**: 使用 `kaldi_native_fbank` 提取 80 维 FBANK → CMVN 归一化 → 填充到 `UNIVERSAL_MAX_LEN=3000` 帧
- **`CMVN`**: 从 Kaldi `.ark` 文件加载均值/方差统计量

## 训练脚本 (`scripts/`)

### `train_layer_wise_conformer.py` — Conformer 训练入口

**`train_layer_wise_steermoe_for_conformer()`**:
1. 加载 YAML 配置
2. 加载冻结 LLM decoder (`AutoModelForCausalLM`)
3. 加载 Conformer 特征提取器 (`ASRFeatExtractor` with CMVN)
4. 构建 `SteerMoEEfficientLayerWiseModelForConformer` 模型
5. 加载 parquet 数据集 → train/val split
6. 创建 DataCollator
7. 注册 callbacks: `SteeringAnalysisCallback` + `GradientClippingCallback`
8. HuggingFace `Trainer` 训练
9. 保存模型 + `pytorch_model.bin`

**`evaluate_layer_wise_model()`**:
1. 加载模型 + checkpoint
2. 手动 DataLoader 循环
3. `model.generate()` → decode → 去除 prompt 前缀
4. 计算 CER/WER

### `train_layer_wise.py` — Whisper 训练入口

同 Conformer 版本逻辑，但使用 WhisperFeatureExtractor 和 Whisper 编码器。

### 消融实验脚本

- `train_layer_wise_linear_whisper.py`: 仅 Linear 投影 (无 MoE routing) — Whisper
- `train_conformer_linear.py`: 仅 Linear 投影 — Conformer

### 其他脚本

- `train_layer_wise_conformer_clothoaqa.py`: Conformer + ClothoAQA 音频问答训练
- `train_layer_wise_prompt.py`: 带逐样本 prompt 的 Whisper 训练 (用于 QA 任务)
- `train_lora.py` / `eval_lora.py`: LoRA 基线实验
- `train.py` / `example_training.py`: 早期/示例训练脚本

### CLI 参数

两个主力训练脚本共享相同的 argparse 参数:

| 参数 | 用途 | 适用模式 |
|------|------|---------|
| `--config` | YAML 配置文件路径 | train/eval |
| `--mode` | `train`、`eval`、`analyze` | 全部 |
| `--deepspeed_config` | DeepSpeed JSON 路径 | train |
| `--resume_from` | 训练恢复的 checkpoint 路径 | train |
| `--model_path` | 评估/分析的模型路径 | eval/analyze |
| `--eval_dataset` | 评估数据集名称 | eval |
| `--local_rank` | 分布式训练 rank | train/eval |

**注意**: eval 模式使用 `--model_path` 而非 `--resume_from`。`--resume_from` 仅用于训练恢复。

### Callbacks

- **`SteeringAnalysisCallback`**: 定期记录 steering 参数统计 (范数、gating 分布等)
- **`GradientClippingCallback`**: 对 steering 参数单独做梯度裁剪

## 预处理模块 (`pre_process/`)

### `preprocess_utils.py` — 核心预处理函数

- **`prepare_dataset()`**: Whisper 路径，调用 `WhisperFeatureExtractor` 提取 Mel
- **`prepare_dataset_for_conformer()`**: Conformer 路径，将 float 音频转为 int16 (×32768)

### `modules.py` — 遗留数据工具

- **`DataCollatorSpeechSeq2SeqWithPadding`**: 早期版本的 collator
- **`ASREvaluator`**: 使用本地 `wer.py` 计算 WER 指标

### 预处理脚本共同模式

所有脚本遵循相同流程:
1. 读取原始数据 (parquet/wav/CSV)
2. 构建 HuggingFace `Dataset`
3. Cast audio 列为 `Audio(sampling_rate=16000)`
4. `.map(prepare_dataset, num_proc=200)` 并行处理
5. 保存到 `processed_datasets/` 目录

注意: 预处理脚本中可能包含旧的硬编码路径，需要按 `datasets.md` 中的映射表更新。
