# SteerMoE Training Procedure Review and Updates

## Overview of Changes

This document reviews the updated SteerMoE training procedure with enhanced loss computation and data handling capabilities. The key improvements focus on proper loss masking for audio prompt tokens and enhanced data collation with textual prompt support.

## Major Updates

### 1. Enhanced Forward Function (`SteerMoEEfficientLayerWiseModel.forward()`)

#### Key Improvements:
- **Proper Loss Computation**: Now correctly computes loss while ignoring prepended audio prompt tokens
- **Flexible Input Handling**: Supports both `audio_waveform` and `input_features` parameter names
- **Attention Mask Support**: Creates proper attention masks for the combined audio-text sequence
- **Label Alignment**: Ensures labels correspond only to the text portion, with audio prompts masked as -100

#### Critical Changes:
```python
# Before: Simple concatenation without proper masking
inputs_embeds = torch.cat([prompts, input_embeds], dim=1)
labels = torch.cat([labels.new_full((labels.size(0), prompt_len), -100), labels], dim=1)

# After: Proper alignment and masking
if labels.size(1) != text_embeds.size(1):
    # Handle label-text length mismatch
    if labels.size(1) > text_embeds.size(1):
        labels = labels[:, :text_embeds.size(1)]
    else:
        pad_length = text_embeds.size(1) - labels.size(1)
        labels = torch.cat([labels, labels.new_full((labels.size(0), pad_length), -100)], dim=1)

# Prepend -100 tokens for audio prompts (ignored in loss)
full_labels = torch.cat([labels.new_full((labels.size(0), audio_prompt_len), -100), labels], dim=1)
```

### 2. Enhanced Data Collator (`DataCollatorSpeechSeqSeqWithPadding`)

#### New Features:
- **Textual Prompt Support**: Automatically prepends textual prompts to decoder input
- **Flexible Audio Handling**: Supports variable-length audio with proper padding/truncation
- **Proper Label Masking**: Handles labels correctly for prompt + text sequences
- **Robust Tokenization**: Handles edge cases like empty sequences and padding

#### Configuration Parameters:
```python
@dataclass
class DataCollatorSpeechSeqSeqWithPadding:
    tokenizer: Any
    textual_prompt: Optional[str] = None          # e.g., "请转写以下音频内容为文字："
    max_length: int = 512                         # Maximum sequence length
    audio_column: str = "audio_features"          # Audio feature column name
    text_column: str = "input_ids"                # Text token column name
    return_attention_mask: bool = True            # Whether to return attention masks
```

### 3. Improved Training Script (`train_layer_wise.py`)

#### Enhanced Dataset Handling:
- **Robust Directory Expansion**: Automatically handles nested parquet directories
- **Proper Dataset Preparation**: Maps and processes datasets correctly
- **Enhanced Metrics Computation**: Properly decodes predictions and labels for evaluation

#### Training Configuration:
```python
# Enhanced training arguments
training_args = TrainingArguments(
    # ... existing args ...
    dataloader_drop_last=True,      # Ensure consistent batch sizes
    remove_unused_columns=False,    # Keep all columns for custom data collator
)
```

## Training Procedure Breakdown

### Phase 1: Model Initialization
1. **Load Pre-trained Components**:
   - Whisper encoder (frozen)
   - LLM decoder (frozen) 
   - Tokenizer with proper padding token setup

2. **Create SteerMoE Architecture**:
   - Wrap Whisper encoder with efficient layer-wise steering
   - Initialize steering vectors and router
   - Set up projection layers if needed

### Phase 2: Dataset Preparation
1. **Load and Filter Data**:
   ```python
   # Load from multiple parquet directories
   dataset = load_parquet_datasets_for_steermoe(parquet_dirs)
   
   # Filter by audio/text length
   dataset = filter_dataset_by_length(dataset, max_audio_length=30.0, max_text_length=448)
   ```

2. **Process Features**:
   ```python
   # Map processing function to extract audio features and tokenize text
   processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
   ```

3. **Create Data Collator**:
   ```python
   data_collator = DataCollatorSpeechSeqSeqWithPadding(
       tokenizer=tokenizer,
       textual_prompt="请转写以下音频内容为文字：",
       max_length=448,
       audio_column="audio_features",
       text_column="input_ids"
   )
   ```

### Phase 3: Training Loop
1. **Forward Pass**:
   - Audio waveform → Whisper encoder with layer-wise steering
   - Audio features → Projection → Audio prompts
   - Text tokens → Text embeddings
   - Concatenate: [Audio prompts, Text embeddings]

2. **Loss Computation**:
   - Create full labels: [-100 for audio prompts, actual labels for text]
   - Compute cross-entropy loss only on text portion
   - Audio prompts are ignored (masked with -100)

3. **Optimization**:
   - Different learning rates for different components:
     - Steering vectors: 1e-3 (higher)
     - Router: 1e-4 (medium)
     - Other parameters: 1e-4 (standard)

### Phase 4: Evaluation and Analysis
1. **Metrics Computation**:
   - Character Error Rate (CER)
   - Word Error Rate (WER)
   - Steering analysis (expert usage, layer scales)

2. **Steering Analysis**:
   - Monitor layer scale values
   - Track expert utilization patterns
   - Analyze steering strength across layers

## Key Training Features

### 1. Steering Analysis Callback
- **Purpose**: Monitor steering patterns during training
- **Frequency**: Every 100 steps (configurable)
- **Metrics**: Layer scales, steering vector norms, expert usage

### 2. Gradient Clipping Callback
- **Purpose**: Prevent steering vectors from growing too large
- **Target**: Steering vectors and layer scales only
- **Max Norm**: 1.0 (configurable)

### 3. Early Stopping
- **Metric**: Character Error Rate (CER)
- **Patience**: 3 epochs
- **Goal**: Minimize overfitting

## Configuration Guidelines

### Essential Parameters
```yaml
# Steering Configuration
steering:
  num_experts: 8                    # Number of steering experts
  steering_scale: 0.1               # Initial steering strength
  use_layer_scales: true            # Enable layer-specific scaling
  steering_gradient_clip: 1.0       # Gradient clipping for steering

# Training Configuration  
training:
  batch_size: 4                     # Adjust based on GPU memory
  learning_rate: 1e-4               # Base learning rate
  steering_learning_rate: 1e-3      # Higher LR for steering vectors
  router_learning_rate: 1e-4        # Medium LR for router

# Dataset Configuration
textual_prompt: "请转写以下音频内容为文字："  # Task instruction
max_audio_length: 30.0              # Seconds
max_text_length: 448                # Tokens
```

### Memory Optimization
- **Batch Size**: Start with 4, adjust based on GPU memory
- **Max Prompt Tokens**: Limit to 512 to control memory usage
- **FP16**: Enable for reduced memory footprint
- **Gradient Checkpointing**: Consider for very large models

## Expected Training Behavior

### Convergence Patterns
1. **Initial Phase** (Epochs 1-3):
   - Rapid decrease in training loss
   - High steering vector activity
   - Unstable expert usage patterns

2. **Stabilization Phase** (Epochs 4-7):
   - Steady improvement in CER/WER
   - Stabilizing layer scales
   - More balanced expert utilization

3. **Fine-tuning Phase** (Epochs 8-10):
   - Marginal improvements
   - Stable steering patterns
   - Risk of overfitting

### Monitoring Checkpoints
- **Layer Scales**: Should stabilize between 0.05-0.2
- **Expert Usage**: Should show some specialization but not extreme concentration
- **Validation Metrics**: CER should improve consistently

## Troubleshooting Common Issues

### 1. Loss Not Decreasing
- **Check**: Label masking is correct (-100 for audio prompts)
- **Verify**: Text embeddings alignment with labels
- **Adjust**: Learning rates or batch size

### 2. Memory Issues
- **Reduce**: Batch size or max sequence lengths
- **Enable**: Gradient checkpointing
- **Use**: DeepSpeed for large models

### 3. Expert Imbalance
- **Monitor**: Expert usage patterns
- **Adjust**: Load balancing loss weight
- **Tune**: Steering scale initialization

### 4. Steering Vector Explosion
- **Enable**: Gradient clipping for steering vectors
- **Reduce**: Steering learning rate
- **Monitor**: Layer scale values

## Performance Optimization

### Training Speed
- **DataLoader**: Set `num_workers=4` for faster data loading
- **Compilation**: Use `torch.compile()` for PyTorch 2.0+
- **Mixed Precision**: Enable FP16 or BF16

### Model Quality
- **Data Quality**: Ensure clean audio-text pairs
- **Prompt Engineering**: Optimize textual prompts for target domain
- **Regularization**: Use appropriate weight decay and dropout

## Conclusion

The updated SteerMoE training procedure provides robust handling of audio-text sequences with proper loss masking and enhanced data collation. The key improvements ensure that:

1. **Loss computation** correctly ignores audio prompt tokens
2. **Data collation** handles variable-length sequences robustly
3. **Training stability** is maintained through proper gradient clipping and learning rate scheduling
4. **Model analysis** is supported through comprehensive steering callbacks

This enhanced system is ready for production training and should provide better convergence and stability compared to the previous implementation. 