# Configuration Files

This folder contains YAML configuration files that control all aspects of training and evaluation for SteerMoE models.

## 📁 File Organization

Configuration files follow the naming convention:
```
{method}_{encoder}_{decoder}_{dataset}_{mode}_{variant}.yaml
```

Where:
- **method**: `layer_wise` (SteerMoE) or `linear` (ablation baseline)
- **encoder**: `whisper` or `conformer`
- **decoder**: `qwen3b`, `qwen7b`, `llama7b`, etc.
- **dataset**: `libri`, `aishell`, `clothoaqa`, etc.
- **mode**: `train` or `eval`/`test`
- **variant**: `aed` (audio event detection), `linear` (linear-only), etc.

## 🎯 Main Configuration Files

### Training Configurations

#### English ASR (LibriSpeech)

| File | Encoder | Decoder | Description |
|------|---------|---------|-------------|
| `layer_wise_whisper_qwen7b_libri_train.yaml` | Whisper-large-v3 | Qwen2.5-7B | **Recommended for English** |
| `layer_wise_whisper_qwen3b_libri_train.yaml` | Whisper-large-v3 | Qwen2.5-3B | Faster training, lower memory |
| `layer_wise_conformer_qwen7b_libri_train_aed.yaml` | Conformer | Qwen2.5-7B | Alternative encoder |

#### Chinese ASR (AISHELL)

| File | Encoder | Decoder | Description |
|------|---------|---------|-------------|
| `layer_wise_whisper_qwen7b_aishell_train.yaml` | Whisper-large-v3 | Qwen2.5-7B | Standard Chinese ASR |
| `layer_wise_conformer_qwen7b_aishell_train.yaml` | Conformer | Qwen2.5-7B | **Better for Chinese** |

#### Audio Question Answering (ClothoAQA)

| File | Encoder | Decoder | Description |
|------|---------|---------|-------------|
| `layer_wise_whisper_qwen7b_clothoaqa_train.yaml` | Whisper-large-v3 | Qwen2.5-7B | Audio QA task |

#### Ablation Study (Linear-only Baseline)

| File | Description |
|------|-------------|
| `linear_whisper_qwen7b_libri_train.yaml` | Linear projection only (no SteerMoE) |
| `linear_conformer_qwen3b_libri_train_aed.yaml` | Conformer + Linear baseline |

### Evaluation Configurations

Evaluation configs use the same naming but with `_test` or `_eval` suffix:
- `layer_wise_whisper_qwen7b_libri_test.yaml`
- `layer_wise_whisper_qwen7b_aishell_test.yaml`
- `layer_wise_whisper_qwen7b_clothoaqa_test.yaml`

## 📝 Configuration Structure

### Example Configuration

```yaml
# configs/layer_wise_whisper_qwen7b_libri_train.yaml

# ============= Audio Encoder =============
whisper_encoder:
  model_path: "/mnt/models/whisper-large-v3/"
  feature_dim: 1280              # Whisper-large output dimension
  num_layers: 32                 # Number of encoder layers

# ============= Language Decoder =============
llm_decoder:
  model_name: "/mnt/models/Qwen2.5-7B-Instruct/"
  max_length: 512
  use_cache: false               # Disable KV cache during training

# ============= Steering Configuration =============
steering:
  num_experts: 8                 # Number of MoE experts
  steering_scale: 0.1            # Initial steering strength
  use_layer_scales: true         # Enable per-layer scaling
  use_gradient_clipping: true    # Clip steering gradients
  steering_gradient_clip: 1.0    # Max gradient norm

# ============= Training Configuration =============
training:
  output_dir: "/mnt/results/layer_wise_steermoe_qwen7b_whisper_libri"
  logging_dir: "/mnt/logs/layer_wise_steermoe_qwen7b_whisper_libri"
  batch_size: 4                  # Per-device batch size
  epochs: 10
  learning_rate: 1e-4            # Base learning rate
  steering_learning_rate: 1e-2   # Higher LR for steering vectors
  router_learning_rate: 1e-3     # LR for MoE router
  weight_decay: 0.01
  warmup_steps: 1000
  gradient_clipping: 1.0
  fp16: true                     # Mixed precision training
  use_deepspeed: true            # Enable DeepSpeed

# ============= Dataset Configuration =============
parquet_dirs:
  - "/mnt/processed_datasets/librispeech_asr/train.clean.100/"
  - "/mnt/processed_datasets/librispeech_asr/train.clean.360/"

audio_column: "audio"            # Column name for audio data
text_column: "text"              # Column name for text
sample_rate: 16000
max_audio_length: 30.0           # Max audio duration (seconds)
max_text_length: 448             # Max text tokens
filter_dataset: true             # Filter by length

# ============= Textual Prompt =============
textual_prompt: "please transcribe the audio content into text: "

# ============= Model Architecture =============
max_prompt_tokens: 2048          # Max audio tokens as prompt
use_adapter: true                # Use linear projection layer
save_total_limit: 2              # Keep only 2 recent checkpoints

# ============= Logging & Evaluation =============
logging_steps: 100
eval_steps: 1000
save_steps: 1000

# ============= Callbacks Configuration =============
log_steering_analysis: true      # Log steering patterns
steering_log_interval: 100
clip_steering_gradients: true
use_early_stopping: true
early_stopping_patience: 3

# ============= Advanced Features =============
# Pooling configuration (optional, for downsampling)
pooling_kernel_size: 4           # Downsample by 4×
pooling_position: 32             # Apply at layer 32
pooling_type: "avg"              # "avg" or "max"
```

## 🔧 Key Configuration Parameters

### Audio Encoder Settings

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `model_path` | Path to pre-trained encoder | `/path/to/whisper-large-v3/` |
| `feature_dim` | Encoder output dimension | 1280 (Whisper-large), 512 (Conformer) |
| `num_layers` | Number of encoder layers | 32 (Whisper-large), 12 (Conformer) |

### Steering (SteerMoE) Settings

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `num_experts` | Number of MoE experts | 8 (balance performance/memory) |
| `steering_scale` | Initial steering strength | 0.1 (stable training) |
| `steering_learning_rate` | LR for steering vectors | 1e-2 to 1e-3 (10-100× base LR) |
| `router_learning_rate` | LR for MoE router | 1e-3 to 1e-4 (10× base LR) |
| `steering_gradient_clip` | Max gradient norm | 1.0 (prevent explosion) |

### Training Settings

| Parameter | Description | Recommendations |
|-----------|-------------|----------------|
| `batch_size` | Per-device batch size | 2-4 (depends on GPU memory) |
| `epochs` | Training epochs | 10 (usually converges by epoch 7) |
| `learning_rate` | Base learning rate | 1e-4 (stable baseline) |
| `warmup_steps` | LR warmup steps | 1000 (smooth start) |
| `fp16` | Mixed precision | `true` (2× memory savings) |

### Dataset Settings

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `max_audio_length` | Max audio duration | 30.0 seconds (balance length/memory) |
| `max_text_length` | Max text tokens | 448 tokens (typical ASR utterance) |
| `textual_prompt` | Task instruction | Language-specific, task-specific |

### Important Notes on Prompts

The `textual_prompt` is crucial for guiding the LLM:

- **English ASR**: `"please transcribe the audio content into text: "`
- **Chinese ASR**: `"请转写以下音频内容为文字："` or `"请逐字复述音频内容为文字: "`
- **Audio QA**: `"Please answer the question about the audio: "`
- **Audio Captioning**: `"Describe the audio content: "`

## 🎛️ Creating Custom Configurations

### Step 1: Choose a Template

Start with an existing config that matches your use case:
```bash
cp configs/layer_wise_whisper_qwen7b_libri_train.yaml \
   configs/my_custom_config.yaml
```

### Step 2: Update Paths

```yaml
whisper_encoder:
  model_path: "/your/path/to/whisper-large-v3/"

llm_decoder:
  model_name: "/your/path/to/Qwen2.5-7B-Instruct/"

parquet_dirs:
  - "/your/path/to/processed_data/train/"
```

### Step 3: Adjust Hyperparameters

For **low-resource scenarios** (small GPUs):
```yaml
training:
  batch_size: 2              # Reduce batch size
  epochs: 15                 # Train longer

max_prompt_tokens: 1024      # Limit audio token length
max_text_length: 256         # Shorter text sequences
fp16: true                   # Essential for memory
```

For **high-performance scenarios** (large GPUs):
```yaml
training:
  batch_size: 8              # Increase batch size
  epochs: 10

steering:
  num_experts: 16            # More experts
  steering_learning_rate: 5e-3  # Higher LR

max_prompt_tokens: 4096      # Longer audio
```

### Step 4: Adjust for Your Task

For **Question Answering**:
```yaml
textual_prompt: "Please answer the question about the audio: "
max_text_length: 512         # QA answers can be longer
```

For **Multi-lingual**:
```yaml
textual_prompt: "Transcribe the audio in [TARGET_LANGUAGE]: "
llm_decoder:
  model_name: "/path/to/multilingual-llm/"
```

## 📊 DeepSpeed Configuration

SteerMoE supports DeepSpeed for distributed training. Common DeepSpeed configs:

### `stage2_simple.json` (Recommended)

```json
{
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

### `stage3.json` (Maximum Memory Efficiency)

```json
{
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    },
    "offload_param": {
      "device": "cpu"
    }
  }
}
```

**Usage**:
```bash
deepspeed --num_gpus=4 scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --deepspeed_config configs/stage2_simple.json
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

1. **Reduce batch size**: `batch_size: 1` or `2`
2. **Enable gradient checkpointing**: `gradient_checkpointing: true`
3. **Limit sequence lengths**: Lower `max_prompt_tokens` and `max_text_length`
4. **Use DeepSpeed ZeRO-3**: Switch to `stage3.json`

### Poor Convergence

If loss doesn't decrease:

1. **Check learning rates**: Ensure `steering_learning_rate >> learning_rate`
2. **Increase warmup**: `warmup_steps: 2000`
3. **Verify data**: Check `textual_prompt` matches your language/task
4. **Monitor steering**: Enable `log_steering_analysis: true`

### Slow Training

If training is too slow:

1. **Enable mixed precision**: `fp16: true`
2. **Increase batch size**: If memory allows
3. **Reduce logging frequency**: Increase `logging_steps` and `eval_steps`
4. **Use faster data loading**: `dataloader_num_workers: 4`

## 📚 Configuration Reference

For a complete list of all configuration options, see:
- Main model config: [`steer_moe/models.py`](../steer_moe/models.py)
- Training script: [`scripts/train_layer_wise.py`](../scripts/train_layer_wise.py)
- Data collator: [`steer_moe/utils.py`](../steer_moe/utils.py)

---

**Need help?** Open an issue on GitHub or refer to the main [README.md](../README.md).

