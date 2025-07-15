# SteerMoE Current Implementation Summary

## Overview
The current SteerMoE implementation uses a **single router with layer-wise steering vectors** approach. The audio encoder (Whisper) output is directly fed to the LLM decoder, with steering vectors applied at each layer of the Whisper encoder.

## Architecture Components

### 1. EfficientLayerWiseSteeringWhisperEncoder
**Location**: `steer_moe/efficient_layer_wise_whisper.py`

**Purpose**: Wraps the original Whisper encoder with layer-wise steering capabilities.

**Key Components**:
- `original_encoder`: Frozen Whisper encoder
- `steering_vectors`: Learnable vectors for each layer `(num_layers, num_experts, feature_dim)`
- `router`: Single router outputting weights for all layers `(feature_dim → num_experts * num_layers)`
- `layer_scales`: Learnable scaling factors for each layer

**Input/Output**:
- **Input**: Mel spectrogram `(batch, seq_len, mel_dim)`
- **Output**: Steered features `(batch, seq_len, feature_dim)`

**Workflow**:
```python
# For each Whisper layer:
1. Apply original layer: layer_output = layer(x)
2. Get router output: router_output = self.router(layer_output)
3. Extract layer-specific weights: layer_gating_logits = router_output[:, :, start_idx:end_idx]
4. Apply softmax: gating_scores = F.softmax(layer_gating_logits, dim=-1)
5. Apply steering: steering_adjustment = torch.einsum('bte,ef->btf', gating_scores, steering_vectors)
6. Add to output: steered_output = layer_output + layer_scale * steering_adjustment
```

### 2. SteerMoEEfficientLayerWiseModel
**Location**: `steer_moe/models.py`

**Purpose**: Main model combining steered Whisper encoder with LLM decoder.

**Key Components**:
- `whisper_encoder`: EfficientLayerWiseSteeringWhisperEncoder
- `llm_decoder`: Frozen LLM decoder
- `prompt_proj`: Optional projection layer

**Input/Output**:
- **Input**: Audio waveform + optional text tokens
- **Output**: Model logits or loss

**Workflow**:
```python
1. Extract features: h_audio = self.whisper_encoder.tokenize_waveform(audio_waveform)
2. Project if needed: prompts = self.prompt_proj(h_audio) if self.prompt_proj else h_audio
3. Handle text embeddings: input_embeds = self.llm_decoder.get_input_embeddings()(decoder_input_ids)
4. Concatenate: inputs_embeds = torch.cat([prompts, input_embeds], dim=1)
5. Pass to decoder: output = self.llm_decoder(inputs_embeds=inputs_embeds, labels=labels)
```

## Training Process

### Updated Training Script
**Location**: `scripts/train_layer_wise.py`

**Key Features**:
1. **Layer-wise Training**: Supports training specific components
2. **Steering Analysis**: Real-time analysis of steering patterns
3. **Gradient Clipping**: Prevents steering vectors from growing too large
4. **Learning Rate Scheduling**: Different learning rates for different components
5. **Validation Metrics**: Steering-specific validation metrics

**Training Flow**:
```python
1. Load models (Whisper encoder, LLM decoder)
2. Create layer-wise steering Whisper encoder
3. Create main SteerMoE model
4. Load and prepare dataset
5. Create training callbacks (steering analysis, gradient clipping)
6. Train with HuggingFace Trainer
7. Save final model
```

### Configuration
**Location**: `configs/layer_wise.yaml`

**Key Settings**:
- `steering.num_experts`: Number of steering experts (default: 8)
- `steering.steering_scale`: Initial steering scale (default: 0.1)
- `training.steering_learning_rate`: Learning rate for steering vectors (default: 1e-3)
- `training.router_learning_rate`: Learning rate for router (default: 1e-4)

## Key Advantages

1. **Parameter Efficiency**: Single router vs multiple routers
2. **Layer-wise Control**: Fine-grained steering per Whisper layer
3. **Frozen Components**: Leverages pre-trained Whisper and LLM
4. **Direct Integration**: Audio features directly feed into LLM
5. **Analysis Capability**: Can analyze steering patterns across layers

## Training Features

### 1. Steering Analysis Callback
- Logs layer scales and steering vector norms
- Monitors steering patterns during training
- Helps understand how steering evolves

### 2. Gradient Clipping Callback
- Clips gradients for steering vectors and layer scales
- Prevents steering vectors from growing too large
- Maintains training stability

### 3. Layer-wise Optimizer
- Different learning rates for different components
- Steering vectors: Higher learning rate (1e-3)
- Router: Lower learning rate (1e-4)
- Other components: Standard learning rate (1e-4)

### 4. Early Stopping
- Monitors validation metrics
- Stops training when no improvement
- Saves best model automatically

## Usage Examples

### Training
```bash
python scripts/train_layer_wise.py \
    --config configs/layer_wise.yaml \
    --mode train
```

### Evaluation
```bash
python scripts/train_layer_wise.py \
    --config configs/layer_wise.yaml \
    --mode eval \
    --model_path ./results/layer_wise_steermoe/final \
    --eval_dataset librispeech_test_clean
```

### Analysis
```bash
python scripts/train_layer_wise.py \
    --config configs/layer_wise.yaml \
    --mode analyze \
    --model_path ./results/layer_wise_steermoe/final
```

## Monitoring and Analysis

### Steering Analysis
The training script includes real-time steering analysis:
- Layer scale values
- Steering vector norms per layer
- Expert utilization patterns
- Steering strength metrics

### Validation Metrics
- CER (Character Error Rate)
- WER (Word Error Rate)
- Steering-specific metrics

## Limitations and Considerations

1. **Feature Space Gap**: No explicit bridging between audio and text feature spaces
2. **Training Complexity**: Requires careful tuning of steering scales
3. **Memory Usage**: Significant due to layer-wise steering vectors
4. **Computational Cost**: Higher than simple post-encoder alignment

## Future Improvements

1. **Cross-modal Alignment**: Better bridging between audio and text spaces
2. **Progressive Training**: Train different layers at different stages
3. **Adaptive Steering**: Dynamic adjustment of steering scales
4. **Multi-modal Tasks**: Extend to other modalities beyond audio
5. **Efficiency Optimizations**: Reduce memory and computational requirements 