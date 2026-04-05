# SteerMoE

SteerMoE: Efficient Audio-Language Alignment with a Mixture-of-Experts Steering Module (Interspeech 2026)

冻结 audio encoder (Whisper/Conformer) 和 LLM decoder (Qwen)，仅训练 steering 对齐模块 (参数量随 LLM 规模变化，Conformer+Qwen7B 约 1.8M，Whisper+Qwen7B 约 5.2M)，实现音频到语言模型的跨模态对齐。

## 项目结构

```
steer_moe/          核心模型: steering 编码器、模型类、DataCollator、特征提取
scripts/            训练与评估入口脚本、WER/CER 指标、DeepSpeed 启动
pre_process/        各数据集预处理脚本 (LibriSpeech, AISHELL, ClothoAQA, MMAU, OpenAudioBench)
configs/            YAML 训练配置 + DeepSpeed JSON
results/            训练输出 (checkpoints)
papers/             论文 LaTeX 源码 (Interspeech 2026 / ICASSP 2025)
cross_modal_steer/  路由器架构探索实验
docs/               架构分析文档
```

## 关键路径

- 数据集: `~/data/`，模型权重: `~/models/`
- 预处理后数据: `~/data/processed_datasets/`
- 训练输出: `results/` 或配置中指定的 `output_dir`
- 详见 @.claude/rules/datasets.md

## 常用命令

```bash
# 训练 (Conformer + Qwen7B, LibriSpeech)
python scripts/train_layer_wise_conformer.py \
  --config configs/layer_wise_conformer_qwen7b_libri_train_aed.yaml --mode train

# 训练 (Whisper + Qwen7B, LibriSpeech)
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml --mode train

# DeepSpeed 多 GPU
deepspeed --include localhost:0,1 scripts/train_layer_wise_conformer.py \
  --config configs/layer_wise_conformer_qwen7b_libri_train_aed.yaml \
  --deepspeed configs/stage2.json --mode train

# 评估
python scripts/train_layer_wise_conformer.py \
  --config configs/layer_wise_conformer_qwen7b_libri_eval_aed.yaml \
  --mode eval --eval_dataset librispeech_test_clean \
  --model_path results/<experiment>/checkpoint-<step>

# 数据预处理
python pre_process/pre_process_librispeech.py              # Whisper
python pre_process/pre_process_librispeech_for_conformer.py # Conformer
```

## 开发规范

### 路径
- 配置文件和代码中涉及数据路径一律使用 `~/data/` 前缀
- 模型权重路径一律使用 `~/models/` 前缀
- 旧路径 (`/mnt/`, `/root/autodl-*`, `/home/fengruitao/rt_nas/`) 遇到即更新
- 完整路径映射见 @.claude/rules/datasets.md

### 模型修改
- 主力模型类: `SteerMoEEfficientLayerWiseModel` (Whisper) 和 `SteerMoEEfficientLayerWiseModelForConformer` (Conformer)
- 遗留模型类 (`SteerMoEModel`, `SteerMoEHybridModel`, `SteerMoELayerWiseModel`) 不再维护
- 修改 steering 逻辑时，Whisper 和 Conformer 两条路径需同步更新
- encoder 和 LLM decoder 始终冻结，仅 steering_vectors / router / layer_scales / prompt_proj 可训练
- 模块详情见 @.claude/rules/modules.md

### 配置
- 新建 YAML 配置遵循命名规范: `layer_wise_{encoder}_{llm}_{dataset}_{mode}[_suffix].yaml`
- 三组独立学习率: `learning_rate` (prompt_proj), `steering_learning_rate`, `router_learning_rate`
- 配置详情见 @.claude/rules/config.md

### 预处理
- 所有预处理脚本使用 `num_proc=200` 并行
- Whisper 路径输出 float32 Mel spectrogram；Conformer 路径输出 int16 raw audio
- 新增数据集预处理时复用 `preprocess_utils.py` 中的 `prepare_dataset()` / `prepare_dataset_for_conformer()`

### 评估
- 英文 ASR 用 WER (`scripts/wer.py`)，中文 ASR 用 CER (`scripts/cer.py`)
- 评估时 textual_prompt 必须与训练时一致
- 详情见 @.claude/rules/testing.md

## 架构概览

详见 @.claude/rules/architecture.md

核心公式 (每一层):
```
x[l] = FrozenLayer[l](x[l-1]) + layer_scale[l] * Σ_k(gating_score[l,k] * steering_vector[l,k])
```

单个 router 同时输出所有层的 gating logits，每层切片使用，参数量减少 N 倍。
Pooling (MaxPool1d/AvgPool1d) 在指定层后执行时间维度下采样（Whisper: kernel=4 硬编码，Conformer: kernel=2 默认）。
最后通过 `prompt_proj` (Linear) 将 encoder_dim 投影到 LLM hidden_dim，与文本 embedding 拼接送入冻结 LLM。
