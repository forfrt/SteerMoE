# SteerMoE 配置参考

## YAML 配置文件

所有训练/评估配置位于 `configs/` 目录，命名规范：
```
{method}_{encoder}_{llm}_{dataset}_{mode}[_suffix].yaml
```
- `method`: `layer_wise` (SteerMoE)、`linear` (消融: 仅 Linear)、`lora` (LoRA 基线)
- `encoder`: `whisper` 或 `conformer`
- `llm`: `qwen3b` 或 `qwen7b`
- `dataset`: `libri`、`clothoaqa`、`aishell`、`openaudiobench`
- `mode`: `train`、`eval`、`test`
- `suffix`: 可选，如 `aed`（AED 模型变体）

### 配置结构

```yaml
# 编码器配置
whisper_encoder:              # 或 conformer_encoder
  model_path: "~/models/..."  # 编码器权重路径
  feature_dim: 1280           # 仅 Whisper 需要
  num_layers: 32              # 仅 Whisper 需要

# LLM 解码器配置
llm_decoder:
  model_name: "~/models/Qwen2.5-7B-Instruct/"
  max_length: 512
  use_cache: false

# Steering 配置
steering:
  num_experts: 8              # 专家数量 (2/4/8/16)
  steering_scale: 0.1         # 初始缩放因子
  use_layer_scales: true
  use_gradient_clipping: true
  steering_gradient_clip: 1.0

# 训练配置
training:
  output_dir: "..."
  batch_size: 3-4             # Whisper: 4, Conformer: 3
  epochs: 10
  learning_rate: 1e-3         # prompt_proj 学习率
  steering_learning_rate: 1e-2  # steering_vectors 学习率
  router_learning_rate: 1e-3  # router 学习率
  warmup_steps: 1000
  fp16: true
  use_deepspeed: true

# 数据集配置
parquet_dirs:
  - "~/data/processed_datasets/..."
textual_prompt: "please transcribe the audio content into text: "
max_audio_length: 30.0
max_text_length: 448

# Pooling 配置
pooling_kernel_size: 4        # 时间维度下采样倍率
pooling_position: 32          # 在第几层后池化 (Whisper 默认最后一层)
pooling_type: "avg"           # "avg" 或 "max"
```

### 关键超参数说明

| 参数 | Whisper 路径 | Conformer 路径 | 说明 |
|------|-------------|---------------|------|
| `learning_rate` | 1e-4 | 1e-3 | prompt_proj 学习率 |
| `steering_learning_rate` | 1e-2 | 1e-2 | steering_vectors 学习率，需要较高以快速收敛 |
| `router_learning_rate` | 1e-3 | 1e-3 | router 学习率 |
| `batch_size` | 4 | 3 | Conformer 内存占用更大 |
| `num_experts` | 8 或 16 | 8 | 8 experts 为默认 |

## DeepSpeed 配置

位于 `configs/stage2.json` 和 `configs/stage2_simple.json`，使用 ZeRO Stage 2。

## 启动命令

```bash
# 单 GPU 训练
python scripts/train_layer_wise_conformer.py \
  --config configs/layer_wise_conformer_qwen7b_libri_train_aed.yaml \
  --mode train

# 多 GPU DeepSpeed 训练
deepspeed --include localhost:0,1 scripts/train_layer_wise_conformer.py \
  --config configs/layer_wise_conformer_qwen7b_libri_train_aed.yaml \
  --deepspeed configs/stage2.json \
  --mode train

# 评估
python scripts/train_layer_wise_conformer.py \
  --config configs/layer_wise_conformer_qwen7b_libri_eval_aed.yaml \
  --mode eval \
  --eval_dataset librispeech_test_clean \
  --model_path results/xxx/checkpoint-xxx

# 便捷脚本
bash scripts/steermoe_train.sh   # 预设的 DeepSpeed 启动
bash train.sh                     # 顶层入口
```

## 文本提示 (Textual Prompt)

不同任务使用不同的文本提示前缀，追加在音频 embedding 之后、LLM 生成之前：

| 任务 | 提示 |
|------|------|
| 英文 ASR | `"please transcribe the audio content into text: "` |
| 中文 ASR | `"请转写以下音频内容为文字："` |
| 音频 QA | `"please answer the following question. The question is : {question}"` |
| 多选 QA | `"please answer the following question by choosing the correct answer from the choices; ..."` |

## 环境依赖

- Python 3.10+
- PyTorch 2.x + CUDA
- transformers, datasets, accelerate, deepspeed
- kaldi_native_fbank (Conformer 特征提取)
- evaluate (WER/CER 指标)
