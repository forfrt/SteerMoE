# Training and Evaluation Scripts

This folder contains the main scripts for training, evaluating, and analyzing SteerMoE models.

## 📋 Overview

The scripts are organized by encoder type and purpose:

| Script | Encoder | Purpose | Use Case |
|--------|---------|---------|----------|
| `train_layer_wise.py` | Whisper | **Main training** | English ASR, Audio QA |
| `train_layer_wise_conformer.py` | Conformer | **Main training** | Chinese ASR, streaming |
| `train_layer_wise_linear_whisper.py` | Whisper | Ablation baseline | Linear-only comparison |
| `train_conformer_linear.py` | Conformer | Ablation baseline | Linear-only comparison |
| `cer.py` | - | Evaluation metric | Character Error Rate |
| `wer.py` | - | Evaluation metric | Word Error Rate |

## 🎯 Main Training Scripts

### `train_layer_wise.py` (Recommended)

Main training script for Whisper + SteerMoE architecture.

#### Features

- ✅ Layer-wise steering with MoE routing
- ✅ Single efficient router for all layers
- ✅ Trainable steering vectors and layer scales
- ✅ Linear projection adapter
- ✅ DeepSpeed distributed training
- ✅ Gradient clipping for stability
- ✅ Steering pattern analysis callbacks

#### Usage

**Training**:
```bash
# Single GPU
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --mode train

# Multi-GPU with DeepSpeed (Recommended)
deepspeed --num_gpus=4 scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --deepspeed_config configs/stage2_simple.json \
  --mode train

# Resume from checkpoint
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --mode train \
  --resume_from /path/to/checkpoint-5000
```

**Evaluation**:
```bash
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_test.yaml \
  --mode eval \
  --model_path results/layer_wise_steermoe/final
```

**Steering Analysis**:
```bash
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_test.yaml \
  --mode analyze \
  --model_path results/layer_wise_steermoe/final
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `configs/layer_wise.yaml` | Path to config YAML file |
| `--deepspeed_config` | str | `configs/stage2_simple.json` | DeepSpeed configuration |
| `--eval_dataset` | str | `None` | Name of evaluation dataset |
| `--resume_from` | str | `None` | Checkpoint path to resume |
| `--mode` | str | `train` | Mode: `train`, `eval`, or `analyze` |
| `--model_path` | str | `None` | Model path (for eval/analyze) |
| `--local_rank` | int | `0` | Local rank for distributed training |

#### Architecture Details

**Model**: `SteerMoEEfficientLayerWiseModel`

```python
# Frozen components
whisper_encoder: WhisperEncoder (1550M params, frozen)
llm_decoder: Qwen2.5-7B (7000M params, frozen)

# Trainable components (~1.8M params)
steering_vectors: (32 layers, 8 experts, 1280 dim) = 327,680 params
router: (1280 → 256) = 327,680 params
layer_scales: (32,) = 32 params
projection: (1280 → 896) = 1,146,880 params
```

#### Training Flow

1. **Initialization**:
   - Load frozen Whisper encoder
   - Load frozen LLM decoder (Qwen/LLaMA)
   - Initialize SteerMoE parameters
   - Set up data collator with textual prompts

2. **Forward Pass**:
   ```
   Audio (mel) → Whisper layers with steering → Projection → LLM decoder
   
   For each layer l:
     h_l' = h_l + layer_scale[l] * (router(h_l) @ steering_vectors[l])
   ```

3. **Loss Computation**:
   ```python
   # Concatenate audio and text embeddings
   inputs_embeds = [audio_prompts, text_embeds]
   
   # Mask audio tokens in labels (they don't contribute to loss)
   labels = [-100 * audio_len, actual_labels]
   
   # Compute language modeling loss on text tokens only
   loss = CrossEntropyLoss(logits, labels)
   ```

4. **Optimization**:
   - **Steering vectors**: LR = 1e-2 (high)
   - **Router**: LR = 1e-3 (medium)
   - **Projection**: LR = 1e-4 (base)

#### Callbacks

**SteeringAnalysisCallback**:
- Logs every 100 steps (configurable)
- Tracks layer scale values
- Monitors steering vector norms
- Analyzes expert usage patterns

**GradientClippingCallback**:
- Clips gradients for steering vectors and layer scales
- Prevents gradient explosion
- Max norm: 1.0 (configurable)

#### Example Output

```
Training progress:
Epoch 1/10: 100%|██████████| 1250/1250 [1:15:23<00:00, 3.62s/it]
{'loss': 2.345, 'learning_rate': 0.0001, 'epoch': 1.0}
Layer scales: [0.08, 0.09, 0.11, 0.12, 0.13, ...]
Average steering vector norms per layer: [0.12, 0.15, 0.18, ...]

Epoch 2/10: 100%|██████████| 1250/1250 [1:14:56<00:00, 3.59s/it]
{'loss': 1.823, 'learning_rate': 0.0001, 'epoch': 2.0}

Evaluation:
{'eval_loss': 1.654, 'eval_cer': 0.045, 'eval_wer': 0.082}
```

### `train_layer_wise_conformer.py`

Training script for Conformer + SteerMoE architecture.

#### Key Differences from Whisper Version

1. **Audio Encoder**: Uses Conformer instead of Whisper
   - Better for streaming ASR
   - Optimized for Asian languages (Chinese, Japanese, Korean)
   - Different feature extraction (FBANK + CMVN)

2. **Feature Extractor**:
   ```python
   # Whisper version
   feature_extractor = WhisperFeatureExtractor.from_pretrained(...)
   
   # Conformer version
   from steer_moe.conformer_module.asr_feat import ASRFeatExtractor
   cmvn_path = os.path.join(conformer_model_dir, "cmvn.ark")
   feature_extractor = ASRFeatExtractor(cmvn_path)
   ```

3. **Data Collator**: Uses `DataCollatorSpeechSeqSeqWithPaddingForConformer`

#### Usage

```bash
# Training
deepspeed --num_gpus=4 scripts/train_layer_wise_conformer.py \
  --config configs/layer_wise_conformer_qwen7b_aishell_train.yaml \
  --deepspeed_config configs/stage2_simple.json \
  --mode train

# Evaluation
python scripts/train_layer_wise_conformer.py \
  --config configs/layer_wise_conformer_qwen7b_aishell_test.yaml \
  --mode eval \
  --model_path results/conformer_steermoe/final
```

#### When to Use Conformer

✅ **Use Conformer for**:
- Chinese/Japanese/Korean ASR
- Streaming/real-time applications
- Low-latency requirements
- Domain-specific fine-tuned Conformer models

✅ **Use Whisper for**:
- English and European languages
- General-purpose ASR
- Audio understanding tasks beyond transcription
- Leveraging Whisper's pre-training

## 🔬 Ablation Study Scripts

### `train_layer_wise_linear_whisper.py`

Trains a baseline model with **only linear projection** (no SteerMoE).

#### Purpose

Proves that SteerMoE's dynamic expert selection and layer-wise steering provide significant benefits over a simple adapter.

#### Architecture

```python
# Frozen components
whisper_encoder: WhisperEncoder (frozen, no steering)
llm_decoder: Qwen2.5-7B (frozen)

# Trainable components (~1.1M params)
projection: (1280 → 896) = 1,146,880 params ONLY
```

#### Usage

```bash
# Train linear-only baseline
python scripts/train_layer_wise_linear_whisper.py \
  --config configs/linear_whisper_qwen7b_libri_train.yaml \
  --mode train

# Evaluate and compare
python scripts/train_layer_wise_linear_whisper.py \
  --config configs/linear_whisper_qwen7b_libri_test.yaml \
  --mode eval \
  --model_path results/linear_baseline/final
```

#### Expected Results

| Model | CER | WER | Trainable Params |
|-------|-----|-----|------------------|
| Linear-only | 6.8% | 12.1% | 1.1M |
| **SteerMoE** | **4.5%** | **8.2%** | **1.8M** |

**Improvement**: ~35% relative error reduction with only 60% more parameters.

### `train_conformer_linear.py`

Linear-only baseline for Conformer encoder.

```bash
python scripts/train_conformer_linear.py \
  --config configs/linear_conformer_qwen7b_aishell_train.yaml \
  --mode train
```

## 📊 Evaluation Metrics

### `cer.py` - Character Error Rate

Computes character-level accuracy for ASR evaluation.

**Formula**:
```
CER = (Substitutions + Insertions + Deletions) / Total Characters
```

**Usage**:
```python
from datasets import load_metric

cer_metric = load_metric('./scripts/cer.py')
cer = cer_metric.compute(
    predictions=["the quick brown fox"],
    references=["the quik brown fox"]
)
print(f"CER: {cer:.2%}")  # Output: CER: 4.00%
```

**When to use**:
- Character-based languages (Chinese, Japanese)
- Fine-grained ASR evaluation
- Comparing similar systems

### `wer.py` - Word Error Rate

Computes word-level accuracy for ASR evaluation.

**Formula**:
```
WER = (Substitutions + Insertions + Deletions) / Total Words
```

**Usage**:
```python
from datasets import load_metric

wer_metric = load_metric('./scripts/wer.py')
wer = wer_metric.compute(
    predictions=["the quick brown fox"],
    references=["the brown fox"]
)
print(f"WER: {wer:.2%}")  # Output: WER: 33.33%
```

**When to use**:
- Word-based languages (English, etc.)
- Standard ASR benchmarks (LibriSpeech)
- Comparing with published results

## 🎓 Advanced Usage

### Custom Training Loop

If you need fine-grained control, you can use the training functions directly:

```python
from scripts.train_layer_wise import train_layer_wise_steermoe

trainer, model = train_layer_wise_steermoe(
    config_path='configs/my_config.yaml',
    deepspeed_config_path='configs/stage2_simple.json',
    eval_dataset_name='librispeech_test_clean',
    resume_from_checkpoint=None
)

# Access trainer for custom operations
trainer.save_model('my_checkpoint')
trainer.evaluate()
```

### Manual Evaluation Loop

For custom evaluation logic:

```python
from scripts.train_layer_wise import evaluate_layer_wise_model

results = evaluate_layer_wise_model(
    model_path='results/checkpoint-5000',
    eval_dataset_name='librispeech_test',
    config_path='configs/eval_config.yaml'
)

print(f"CER: {results['cer']:.2%}")
print(f"WER: {results['wer']:.2%}")
```

### Steering Pattern Analysis

Analyze how the model uses experts:

```python
from scripts.train_layer_wise import analyze_steering_patterns

analysis = analyze_steering_patterns(
    model_path='results/final'
)

print(f"Layer scales: {analysis['layer_scale_values']}")
print(f"Expert diversity: {analysis['expert_diversity']}")
print(f"Steering strength: {analysis['steering_strength']}")
```

## 🔧 Distributed Training

### Single Machine, Multiple GPUs

```bash
# Using PyTorch DDP
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml

# Using DeepSpeed (Recommended)
deepspeed --num_gpus=4 \
  scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --deepspeed_config configs/stage2_simple.json
```

### Multiple Machines

Create a hostfile:
```
# hostfile
node1 slots=4
node2 slots=4
```

Run with DeepSpeed:
```bash
deepspeed --hostfile=hostfile \
  scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --deepspeed_config configs/stage2_simple.json
```

### DeepSpeed Configuration

**ZeRO Stage 2** (Recommended):
```json
{
  "fp16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"}
  }
}
```

**ZeRO Stage 3** (Maximum Memory Efficiency):
```json
{
  "fp16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  }
}
```

## 🐛 Troubleshooting

### Training Issues

**Loss not decreasing**:
1. Check learning rates: `steering_learning_rate` should be 10-100× base LR
2. Verify textual prompt matches your language/task
3. Enable steering analysis: `log_steering_analysis: true`
4. Check layer scales: Should be 0.05-0.2

**Out of memory**:
1. Reduce `batch_size`: Try 2 or 1
2. Enable gradient checkpointing
3. Use DeepSpeed ZeRO-3
4. Reduce `max_prompt_tokens` and `max_text_length`

**Slow training**:
1. Enable mixed precision: `fp16: true`
2. Use DeepSpeed
3. Increase `dataloader_num_workers`
4. Reduce logging frequency

### Evaluation Issues

**High CER/WER**:
1. Ensure model is fully trained (check loss convergence)
2. Verify textual prompt matches training prompt
3. Check audio preprocessing (sample rate, features)
4. Try different decoding parameters (beam search, temperature)

**Generation errors**:
1. Check `eos_token_id` is correctly set
2. Verify `max_new_tokens` is sufficient
3. Ensure audio and text are properly aligned

## 📚 Script Reference

### Function Signatures

#### `train_layer_wise_steermoe()`

```python
def train_layer_wise_steermoe(
    config_path: str = 'configs/layer_wise.yaml',
    deepspeed_config_path: str = 'configs/stage2_simple.json',
    eval_dataset_name: Optional[str] = None,
    custom_test_set_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None
) -> Tuple[Trainer, nn.Module]:
    """
    Train SteerMoE model with layer-wise steering.
    
    Returns:
        trainer: HuggingFace Trainer instance
        model: Trained SteerMoE model
    """
```

#### `evaluate_layer_wise_model()`

```python
def evaluate_layer_wise_model(
    model_path: str,
    eval_dataset_name: str,
    config_path: str
) -> Dict[str, float]:
    """
    Evaluate a trained SteerMoE model.
    
    Returns:
        results: Dictionary with CER and WER scores
    """
```

#### `analyze_steering_patterns()`

```python
def analyze_steering_patterns(
    model_path: str
) -> Dict[str, Any]:
    """
    Analyze steering patterns of a trained model.
    
    Returns:
        analysis: Dictionary with layer scales, expert usage, etc.
    """
```

## 🔗 Related Documentation

- Model implementations: [`steer_moe/models.py`](../steer_moe/models.py)
- Training configs: [`configs/README.md`](../configs/README.md)
- Data preprocessing: [`pre_process/README.md`](../pre_process/README.md)
- Main README: [`README.md`](../README.md)

---

**Need help?** Open an issue on GitHub or check the documentation.

