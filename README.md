# SteerMoE

SteerMoE is an alignment architecture for audio-language modeling, built on the principles of parameter-efficiency, dynamic specialization, and representational steering.

## Architecture

SteerMoE consists of three main components:

1. **Frozen Audio Encoder**: A pre-trained Whisper encoder (reused from Kimi-Audio), which converts input audio into continuous acoustic features (downsampled to 12.5Hz).
2. **Frozen LLM Decoder**: A pre-trained large language model (e.g., Llama, GPT), kept entirely frozen.
3. **SteerMoE Aligner (Trainable)**: A lightweight bridge that aligns the audio encoder's output to the LLM's input space. It consists of:
   - A bank of learnable steering vectors ("experts")
   - A lightweight MoE router that dynamically combines experts for each token

## Model Variants

### 1. Original SteerMoE Model
- Direct injection of aligned audio features into LLM decoder
- Audio features are fed directly as hidden states to the LLM
- Suitable for specialized ASR tasks

### 2. SteerMoE Hybrid Model
- Uses SteerMoE aligner output as trainable continuous prompts
- Concatenates audio prompts with text embeddings
- Supports configurable prompt token length and optional projection
- Better for general audio-language tasks with text instructions

## Aligner Mechanics

For each audio token representation $h_t$:
- The router computes gating scores $g_t = \text{Softmax}(\text{Router}(h_t))$
- An adjustment vector $\Delta h_t = \sum_i g_{t,i} s_i$ is computed from the steering vectors $S = \{s_1, ..., s_N\}$
- The aligned representation is $h_{\text{aligned}, t} = h_t + \Delta h_t$
- The sequence $H_{\text{aligned}}$ is fed as a continuous prompt to the frozen LLM

## Dataset Loading

The training script supports loading datasets from parquet directories, similar to `main_word_correct_clips.py`:

```python
# Load datasets from multiple parquet directories
parquet_dirs = [
    "/path/to/dataset1",
    "/path/to/dataset2",
    "/path/to/dataset3"
]

# Load and concatenate datasets
dataset = load_parquet_datasets_for_steermoe(parquet_dirs)

# Filter by audio and text length
filtered_dataset = filter_dataset_by_length(
    dataset,
    max_audio_length=30.0,  # seconds
    max_text_length=448     # tokens
)
```

## Training

### Basic Training
```python
from scripts.train import train_with_deepspeed

# Train original SteerMoE model
train_with_deepspeed(
    config_path="configs/default.yaml",
    use_hybrid=False
)

# Train hybrid model
train_with_deepspeed(
    config_path="configs/default.yaml",
    use_hybrid=True
)
```

### Advanced Training with Evaluation
```python
# Train with evaluation on standard datasets
train_with_deepspeed(
    config_path="configs/default.yaml",
    eval_dataset_name="librispeech_test_clean",
    custom_test_set_path="/path/to/custom/test",
    resume_from_checkpoint="/path/to/checkpoint",
    use_hybrid=True
)
```

### Distributed Training with DeepSpeed
```bash
# Single GPU
python scripts/train.py

# Multi-GPU with DeepSpeed
deepspeed --num_gpus=4 scripts/train.py \
    --config_path configs/default.yaml \
    --deepspeed_config_path configs/deepspeed_config.json \
    --use_hybrid True
```

## Configuration

Update `configs/default.yaml` to configure your experiment:

```yaml
# Model configuration
whisper_encoder:
  model_path: /path/to/whisper/encoder
  freeze: true

llm_decoder:
  model_name: gpt2
  freeze: true

aligner:
  num_experts: 8
  feature_dim: 1024

# Hybrid model parameters
max_prompt_tokens: 512
use_adapter: true

# Dataset configuration
parquet_dirs: []
audio_column: "audio"
text_column: "text"
sample_rate: 16000

# Filtering parameters
filter_dataset: true
max_audio_length: 30.0
max_text_length: 448

# Training parameters
training:
  batch_size: 8
  lr: 1e-3
  epochs: 10
  load_balance_loss_weight: 0.01
```

## Evaluation Metrics

The training script includes evaluation metrics:
- **CER (Character Error Rate)**: Character-level accuracy
- **WER (Word Error Rate)**: Word-level accuracy
- **Load Balancing Loss**: Ensures all experts are utilized

## Model Saving and Loading

### Save Model
```python
from scripts.train import save_trainer_model

# Save model in HuggingFace format
save_trainer_model(trainer, "./results/best_model_hf")
```

### Load Model
```python
from scripts.train import load_trainer_model
from steer_moe.models import SteerMoEModel, SteerMoEHybridModel

# Load original model
model, tokenizer = load_trainer_model(SteerMoEModel, "./results/best_model_hf")

# Load hybrid model
model, tokenizer = load_trainer_model(SteerMoEHybridModel, "./results/best_model_hf")
```

### Push to HuggingFace Hub
```python
from scripts.train import push_to_hub

push_to_hub(
    output_dir="./results/best_model_hf",
    repo_name="your-username/steermoe-demo",
    token="YOUR_HF_TOKEN"
)
```

## Code Reuse
- The Whisper encoder and feature extraction logic are reused from the Kimi-Audio project
- Dataset loading functionality is adapted from `main_word_correct_clips.py`

## Project Structure
```
SteerMoE/
  steer_moe/
    aligner.py          # SteerMoE aligner implementation
    models.py           # SteerMoE and hybrid model classes
    utils.py            # Utility functions
    __init__.py
  scripts/
    train.py            # Main training script
    example_training.py # Example usage
  configs/
    default.yaml        # Default configuration
    deepspeed_config.json # DeepSpeed configuration
  requirements.txt
  README.md
  tests/
    test_aligner.py
```

## Usage Examples

See `scripts/example_training.py` for complete usage examples:

```python
# Example 1: Train original SteerMoE
example_steermoe_training()

# Example 2: Train hybrid model
example_hybrid_training()

# Example 3: Custom configuration
example_with_custom_config()
```

## Key Features

- ✅ **Parameter Efficient**: Only aligner parameters are trainable
- ✅ **Dynamic Specialization**: MoE router adapts to different audio patterns
- ✅ **Representational Steering**: Learnable steering vectors guide audio-text alignment
- ✅ **Hybrid Support**: Both direct injection and continuous prompt approaches
- ✅ **Distributed Training**: DeepSpeed integration for multi-GPU training
- ✅ **Standard Evaluation**: CER/WER metrics on standard ASR datasets
- ✅ **Flexible Dataset Loading**: Support for parquet datasets with filtering
- ✅ **HuggingFace Integration**: Easy model saving/loading and Hub publishing

---
For more details, see the code and comments in each module.
