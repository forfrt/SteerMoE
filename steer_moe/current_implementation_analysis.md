# Current SteerMoE Implementation Analysis

## Overview
The current implementation uses a **single router with layer-wise steering vectors** approach, where the output of the audio encoder (Whisper) is directly fed to the LLM decoder. This is implemented in `efficient_layer_wise_whisper.py` and `models.py`.

## Workflow Summary

```
Audio Waveform → Whisper Encoder (frozen) → Layer-wise Steering → LLM Decoder (frozen) → Output
```

### Detailed Workflow:

1. **Audio Input**: Raw audio waveform
2. **Whisper Encoder**: Frozen Whisper encoder processes audio to continuous features
3. **Layer-wise Steering**: Each layer of Whisper gets steering vectors applied
4. **LLM Decoder**: Frozen LLM decoder receives the steered features as input embeddings
5. **Output**: Generated text or loss for training

## Component Analysis

### 1. EfficientLayerWiseSteeringWhisperEncoder (`efficient_layer_wise_whisper.py`)

**Purpose**: Wraps the original Whisper encoder with layer-wise steering capabilities.

**Key Components**:
- `original_encoder`: Frozen Whisper encoder
- `steering_vectors`: Learnable steering vectors for each layer `(num_layers, num_experts, feature_dim)`
- `router`: Single router that outputs weights for all layers `(feature_dim → num_experts * num_layers)`
- `layer_scales`: Learnable scaling factors for each layer

**Input/Output**:
- **Input**: Mel spectrogram `(batch, seq_len, mel_dim)`
- **Output**: Steered features `(batch, seq_len, feature_dim)`

**Function Flow**:
```python
def forward(self, mel_spec, return_gating=False):
    # For each layer:
    # 1. Apply original Whisper layer
    layer_output = layer(x)
    
    # 2. Get router output for this layer
    router_output = self.router(layer_output)  # (batch, seq_len, num_experts * num_layers)
    layer_gating_logits = router_output[:, :, start_idx:end_idx]
    
    # 3. Apply softmax to get gating scores
    gating_scores = F.softmax(layer_gating_logits, dim=-1)
    
    # 4. Apply steering vectors
    steering_adjustment = torch.einsum('bte,ef->btf', gating_scores, steering_vectors)
    steered_output = layer_output + layer_scale * steering_adjustment
    
    # 5. Pass to next layer
    x = steered_output
```

### 2. SteerMoEEfficientLayerWiseModel (`models.py`)

**Purpose**: Main model that combines the steered Whisper encoder with the LLM decoder.

**Key Components**:
- `whisper_encoder`: EfficientLayerWiseSteeringWhisperEncoder
- `llm_decoder`: Frozen LLM decoder (e.g., LLaMA, GPT-2)
- `prompt_proj`: Optional projection layer if dimensions don't match

**Input/Output**:
- **Input**: Audio waveform + optional text tokens
- **Output**: Model logits or loss

**Function Flow**:
```python
def forward(self, audio_waveform, decoder_input_ids=None, labels=None, 
            prompt_tokens_only=False, return_gating=False):
    # 1. Extract features with layer-wise steering
    h_audio = self.whisper_encoder.tokenize_waveform(audio_waveform)
    
    # 2. Project to decoder dimension if needed
    prompts = self.prompt_proj(h_audio) if self.prompt_proj else h_audio
    
    # 3. Handle text embeddings and concatenation
    if decoder_input_ids is not None:
        input_embeds = self.llm_decoder.get_input_embeddings()(decoder_input_ids)
        inputs_embeds = torch.cat([prompts, input_embeds], dim=1)
        
        # Handle labels for training
        if labels is not None:
            prompt_len = prompts.size(1)
            labels = torch.cat([
                labels.new_full((labels.size(0), prompt_len), -100),
                labels
            ], dim=1)
        
        # Pass to decoder
        output = self.llm_decoder(inputs_embeds=inputs_embeds, labels=labels)
    else:
        # Pure audio generation
        output = prompts
    
    return output
```

## Key Features

### 1. Single Router Efficiency
- **Parameter Count**: `feature_dim * num_experts * num_layers` (much smaller than multiple routers)
- **Computation**: Single forward pass through router, then extract layer-specific weights
- **Memory**: Efficient memory usage compared to multiple routers

### 2. Layer-wise Steering
- **Granular Control**: Each Whisper layer gets its own steering vectors
- **Learnable Scaling**: Each layer has a learnable scaling factor
- **Analysis Capability**: Can return gating scores for analysis

### 3. Frozen Components
- **Whisper Encoder**: Completely frozen, only steering vectors are trainable
- **LLM Decoder**: Completely frozen, receives steered features as input embeddings
- **Training Focus**: Only the steering mechanism is trained

## Training Process

### Current Training Script (`scripts/train.py`)

**Key Functions**:
1. `train_with_deepspeed()`: Main training function
2. `load_parquet_datasets_for_steermoe()`: Load training data
3. `filter_dataset_by_length()`: Filter by audio/text length
4. `prepare_asr_dataset()`: Prepare dataset for training

**Training Flow**:
```python
# 1. Load models
whisper_encoder = WhisperEncoder(config['whisper_encoder']['model_path'])
llm_decoder = AutoModelForCausalLM.from_pretrained(config['llm_decoder']['model_name'])
aligner = SteerMoEAligner(feature_dim, num_experts)

# 2. Create model
model = SteerMoEEfficientLayerWiseModel(
    whisper_encoder_path=whisper_encoder,
    llm_decoder=llm_decoder,
    num_experts=8
)

# 3. Prepare dataset
dataset = load_parquet_datasets_for_steermoe(parquet_dirs)
processed_dataset = dataset.map(prepare_dataset)

# 4. Train with HuggingFace Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    eval_dataset=processed_val,
    compute_metrics=compute_metrics_trainer
)
trainer.train()
```

## Advantages of Current Implementation

1. **Parameter Efficiency**: Single router vs multiple routers
2. **Layer-wise Control**: Fine-grained steering per Whisper layer
3. **Frozen Components**: Leverages pre-trained Whisper and LLM
4. **Direct Integration**: Audio features directly feed into LLM
5. **Analysis Capability**: Can analyze steering patterns across layers

## Limitations

1. **Feature Space Gap**: No explicit bridging between audio and text feature spaces
2. **Limited Cross-modal Understanding**: Relies on LLM's ability to understand audio features
3. **Training Complexity**: Requires careful tuning of steering scales
4. **Memory Usage**: Still significant due to layer-wise steering vectors

## Recommendations for Training Updates

1. **Add Layer-wise Training**: Support for training specific layers
2. **Steering Analysis**: Real-time analysis of steering patterns
3. **Gradient Clipping**: Prevent steering vectors from growing too large
4. **Learning Rate Scheduling**: Different learning rates for different components
5. **Validation Metrics**: Add steering-specific validation metrics 