# SteerMoE Core Implementation

This folder contains the core implementation of the SteerMoE architecture, including model classes, layer-wise steering mechanisms, and utility functions.

## 📁 File Organization

```
steer_moe/
├── models.py                           # Main SteerMoE model classes
├── efficient_layer_wise_whisper.py    # Whisper encoder with layer-wise steering
├── efficient_layer_wise_conformer.py  # Conformer encoder with layer-wise steering
├── utils.py                            # Data collators and utility functions
├── fireredasr_aed.py                  # Audio event detection utilities
├── conformer_module/                   # Conformer implementation
│   └── asr_feat.py                    # FBANK feature extraction
└── tokenizer/                          # Audio tokenizers
    └── whisper_Lv3/
        └── whisper.py                 # Whisper encoder wrapper
```

## 🏗️ Core Model Classes

### `models.py`

Contains all SteerMoE model architectures.

#### Main Models

| Class | Description | Use Case |
|-------|-------------|----------|
| `SteerMoEEfficientLayerWiseModel` | **Main model** with layer-wise steering | Production training |
| `SteerMoEEfficientLayerWiseModelLinear` | Linear-only baseline (no steering) | Ablation studies |
| `SteerMoELayerWiseModel` | Original multi-router version (legacy) | Reference implementation |
| `SteerMoEHybridModel` | Hybrid continuous prompt approach | Research variant |

#### Architecture Overview

```python
class SteerMoEEfficientLayerWiseModel(nn.Module):
    """
    Audio → Frozen Encoder → SteerMoE → Projection → Frozen LLM
    
    Components:
    - whisper_encoder: EfficientLayerWiseSteeringWhisperEncoder
    - llm_decoder: Frozen LLM (Qwen, LLaMA, etc.)
    - prompt_proj: Linear(encoder_dim, llm_dim)
    """
    
    def __init__(
        self,
        whisper_encoder,              # Pre-initialized steering encoder
        llm_decoder,                  # Frozen LLM
        num_experts=8,                # Number of MoE experts
        max_prompt_tokens=2048,       # Max audio sequence length
        use_adapter=True              # Use linear projection
    ):
        # Trainable: steering vectors, router, projection
        # Frozen: whisper_encoder.original_encoder, llm_decoder
```

#### Forward Pass

```python
def forward(
    self,
    audio_waveform=None,      # Raw waveform (inference)
    input_features=None,      # Preprocessed features (training)
    decoder_input_ids=None,   # Text token IDs
    labels=None,              # Training labels
    prompt_tokens_only=False, # Return audio embeddings only
    return_gating=False       # Return MoE gating scores
):
    """
    Training:
        input_features: (batch, 128, 3000) mel-spectrogram
        decoder_input_ids: (batch, seq_len) text tokens
        labels: (batch, seq_len) target tokens
    
    Returns:
        output: ModelOutput with loss and logits
    """
    # 1. Audio encoding with steering
    h_audio = self.whisper_encoder._forward_with_steering(input_features)
    # h_audio: (batch, audio_seq_len, 1280)
    
    # 2. Project to LLM dimension
    audio_prompts = self.prompt_proj(h_audio)
    # audio_prompts: (batch, audio_seq_len, 896)
    
    # 3. Get text embeddings
    text_embeds = self.llm_decoder.model.embed_tokens(decoder_input_ids)
    # text_embeds: (batch, text_seq_len, 896)
    
    # 4. Concatenate
    inputs_embeds = torch.cat([audio_prompts, text_embeds], dim=1)
    
    # 5. Create labels (mask audio tokens with -100)
    full_labels = torch.cat([
        labels.new_full((batch, audio_seq_len), -100),
        labels
    ], dim=1)
    
    # 6. LLM forward pass
    output = self.llm_decoder(
        inputs_embeds=inputs_embeds,
        labels=full_labels
    )
    
    return output  # Contains loss, logits
```

#### Generation (Inference)

```python
def generate(
    self,
    input_features,           # Preprocessed audio features
    decoder_input_ids=None,   # Initial text prompt
    max_new_tokens=512,       # Max tokens to generate
    **kwargs                  # Other generation params
):
    """
    Autoregressive generation for inference.
    
    Example:
        audio = preprocess_audio("audio.wav")
        prompt = tokenizer("Transcribe: ")
        
        output_ids = model.generate(
            input_features=audio,
            decoder_input_ids=prompt,
            max_new_tokens=512,
            num_beams=5,
            do_sample=False
        )
        
        text = tokenizer.decode(output_ids[0])
    """
```

#### Model Properties

```python
# Trainable parameters
self.steering_vectors: (num_layers, num_experts, feature_dim)
self.router: Linear(feature_dim, num_experts * num_layers)
self.layer_scales: (num_layers,)
self.prompt_proj: Linear(encoder_dim, llm_dim)

# Frozen parameters
self.whisper_encoder.original_encoder: WhisperEncoder (frozen)
self.llm_decoder: AutoModelForCausalLM (frozen)
```

## 🔧 Layer-Wise Steering Implementation

### `efficient_layer_wise_whisper.py`

Implements efficient layer-wise steering for Whisper encoder.

#### Key Class: `EfficientLayerWiseSteeringWhisperEncoder`

```python
class EfficientLayerWiseSteeringWhisperEncoder(nn.Module):
    """
    Wraps Whisper encoder with layer-wise steering.
    Uses a SINGLE router for all layers (parameter efficient).
    """
    
    def __init__(
        self,
        whisper_encoder_path,     # Path to Whisper model
        num_experts=8,             # Number of experts per layer
        steering_scale=0.1,        # Initial steering strength
        pooling_kernel_size=4,     # Optional downsampling
        pooling_type=None,         # "avg" or "max"
        pooling_position=32        # Layer to apply pooling
    ):
        # Load original Whisper encoder (frozen)
        self.original_encoder = WhisperEncoder(whisper_encoder_path)
        
        # Trainable steering parameters
        self.steering_vectors = nn.Parameter(
            torch.randn(num_layers, num_experts, feature_dim) * 0.01
        )
        self.router = nn.Linear(feature_dim, num_experts * num_layers)
        self.layer_scales = nn.Parameter(
            torch.ones(num_layers) * steering_scale
        )
        
        # Freeze original encoder
        for param in self.original_encoder.parameters():
            param.requires_grad = False
```

#### Steering Algorithm

```python
def _forward_with_steering(self, mel_features, return_gating=False):
    """
    Apply layer-wise steering during forward pass.
    
    For each layer l in 0..31:
        1. Pass through original frozen layer
        2. Compute MoE gating scores
        3. Apply steering adjustment
        4. Continue to next layer
    """
    
    # Initial processing (conv, positional embedding)
    x = self.original_encoder.conv1(mel_features)
    x = self.original_encoder.conv2(x)
    x = x.permute(0, 2, 1)
    x = x + self.original_encoder.embed_positions.weight[:x.size(1)]
    x = x * self.original_encoder.embed_scale
    
    # Process through layers with steering
    for layer_idx, layer in enumerate(self.original_encoder.layers):
        # 1. Original layer forward
        layer_output = layer(x)[0]  # (batch, seq, feature_dim)
        
        # 2. Compute gating scores
        router_output = self.router(layer_output)  # (batch, seq, num_experts*num_layers)
        
        # Extract this layer's experts
        start = layer_idx * self.num_experts
        end = (layer_idx + 1) * self.num_experts
        gating_logits = router_output[:, :, start:end]  # (batch, seq, num_experts)
        
        # 3. Softmax for expert weights
        gating_scores = F.softmax(gating_logits, dim=-1)
        
        # 4. Weighted combination of steering vectors
        steering_vectors = self.steering_vectors[layer_idx]  # (num_experts, feature_dim)
        steering_adjustment = torch.einsum(
            'bte,ef->btf',
            gating_scores,         # (batch, seq, num_experts)
            steering_vectors       # (num_experts, feature_dim)
        )  # (batch, seq, feature_dim)
        
        # 5. Apply steering with layer-specific scaling
        layer_scale = self.layer_scales[layer_idx]
        x = layer_output + layer_scale * steering_adjustment
        
        # Optional: Apply pooling at specific layer
        if layer_idx + 1 == self.pooling_position and self.pooling_layer:
            x = x.permute(0, 2, 1)  # (batch, feature_dim, seq)
            x = self.pooling_layer(x)
            x = x.permute(0, 2, 1)  # (batch, seq', feature_dim)
    
    # Final layer norm
    x = self.original_encoder.layer_norm(x)
    
    return x
```

#### Why This Design?

**Efficiency**:
- Single router reduces parameters from `num_layers × (feature_dim × num_experts)` to `feature_dim × (num_experts × num_layers)`
- Example: 32 layers, 8 experts, 1280 dim
  - Multi-router: 32 × (1280 × 8) = 327,680 × 32 = 10.5M params
  - Single router: 1280 × (8 × 32) = 1280 × 256 = 327,680 params
  - **32× parameter reduction!**

**Flexibility**:
- Each layer still gets its own set of experts
- Router learns to route differently for each layer
- Layer scales allow per-layer strength adjustment

### `efficient_layer_wise_conformer.py`

Similar implementation for Conformer encoder.

**Key Differences**:
1. Uses Conformer architecture instead of Whisper
2. Different feature extraction (FBANK + CMVN)
3. Streaming-friendly design
4. Better for Asian languages

```python
class EfficientLayerWiseSteeringConformerEncoder(nn.Module):
    """
    Conformer encoder with layer-wise steering.
    Architecture similar to Whisper version.
    """
    
    def __init__(
        self,
        conformer_encoder_path,
        num_experts=8,
        steering_scale=0.1,
        pooling_kernel_size=2,
        pooling_type="max"
    ):
        # Load Conformer encoder
        self.original_encoder = LoadConformerEncoder(conformer_encoder_path)
        # ... same steering setup as Whisper version
```

## 🛠️ Utility Functions

### `utils.py`

Contains data collators and helper functions.

#### `DataCollatorSpeechSeqSeqWithPadding`

Main data collator for SteerMoE training.

```python
@dataclass
class DataCollatorSpeechSeqSeqWithPadding:
    """
    Collates preprocessed audio-text pairs for training.
    
    Handles:
    - Audio feature padding
    - Text tokenization with textual prompts
    - Label masking for audio tokens
    """
    
    feature_extractor: Any        # Whisper/Conformer feature extractor
    tokenizer: Any                # LLM tokenizer
    textual_prompt: str = None    # e.g., "Transcribe: "
    max_length: int = 448         # Max text sequence length
    audio_column: str = "input_features"
    text_column: str = "labels"
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Input features (each sample):
        {
            'input_features': np.ndarray,  # Audio features
            'labels': List[int],           # Token IDs
            'text': str                    # Original text
        }
        
        Output batch:
        {
            'input_features': (batch, 128, max_seq),  # Padded audio
            'decoder_input_ids': (batch, max_len),    # [prompt + labels]
            'labels': (batch, max_len)                # [-100 for prompt, actual labels]
        }
        """
```

**Key Features**:

1. **Audio Padding**:
   ```python
   # Pad audio features to same length
   audio_features = [{"input_features": f} for f in features]
   batch_audio = self.feature_extractor.pad(audio_features, return_tensors="pt")
   ```

2. **Textual Prompt Integration**:
   ```python
   # Prepend textual prompt to labels
   prompt_tokens = self.tokenizer.encode(textual_prompt, add_special_tokens=False)
   decoder_input = torch.cat([prompt_tokens, labels])
   ```

3. **Label Masking**:
   ```python
   # Mask prompt tokens with -100 (ignored in loss)
   empty_prompt = torch.full_like(prompt_tokens, fill_value=-100)
   full_labels = torch.cat([empty_prompt, labels])
   ```

#### `DataCollatorSpeechSeqSeqWithPaddingForConformer`

Variant for Conformer encoder with different feature handling.

```python
@dataclass
class DataCollatorSpeechSeqSeqWithPaddingForConformer:
    """
    Conformer-specific data collator.
    
    Differences:
    - Uses ASRFeatExtractor instead of WhisperFeatureExtractor
    - Handles raw audio waveforms (int16)
    - Includes input_lengths for variable-length sequences
    """
    
    def __call__(self, features):
        # Extract FBANK features from raw audio
        audio_features = [np.array(f['input_features'], dtype=np.int16) for f in features]
        batch_audio, lengths, durations = self.feature_extractor(
            audio_features,
            sample_rate=features[0]['sample_rate']
        )
        
        # ... rest similar to Whisper version
        
        return {
            'input_features': batch_audio,
            'decoder_input_ids': batch_input_ids,
            'labels': batch_labels,
            'input_lengths': lengths  # Important for Conformer
        }
```

#### Helper Functions

```python
def load_balancing_loss(gating_scores):
    """
    Compute auxiliary loss to encourage uniform expert usage.
    
    Args:
        gating_scores: (batch, seq_len, num_experts)
    
    Returns:
        loss: Scalar tensor (KL divergence from uniform distribution)
    """
    expert_usage = gating_scores.mean(dim=(0, 1))  # Average usage per expert
    num_experts = gating_scores.size(-1)
    target = torch.full_like(expert_usage, 1.0 / num_experts)  # Uniform target
    loss = F.kl_div(expert_usage.log(), target, reduction="batchmean")
    return loss


def prepare_dataset(batch, audio_column, text_column, feature_extractor, tokenizer, sample_rate):
    """
    Preprocess a single batch during dataset mapping.
    
    Extracts audio features and tokenizes text.
    """
    audio = batch[audio_column]
    text = batch[text_column]
    
    # Extract audio features
    input_features = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Tokenize text
    text_tokens = tokenizer(text)
    
    return {
        'input_features': input_features,
        'labels': text_tokens['input_ids'],
        'input_length': len(audio['array']) / audio['sampling_rate'],
        'attention_mask': text_tokens['attention_mask'],
        'text': text
    }
```

## 🎓 Advanced Topics

### Custom Steering Mechanisms

You can extend the steering mechanism for custom behaviors:

```python
class CustomSteeringWhisperEncoder(EfficientLayerWiseSteeringWhisperEncoder):
    """
    Custom steering with additional features.
    """
    
    def __init__(self, *args, use_attention_steering=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        if use_attention_steering:
            # Add attention-based routing
            self.attention_router = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=8
            )
    
    def _forward_with_steering(self, mel_features, return_gating=False):
        # Custom routing logic
        # ...
        pass
```

### Gradient Checkpointing

Enable gradient checkpointing for memory efficiency:

```python
model.gradient_checkpointing_enable()

# Training will use less memory but be slightly slower
trainer.train()

model.gradient_checkpointing_disable()
```

### Steering Analysis

Analyze steering patterns to understand model behavior:

```python
def analyze_steering_patterns(model, audio_input):
    """
    Analyze how the model uses experts.
    """
    with torch.no_grad():
        _, gating_scores = model(audio_input, return_gating=True)
    
    # Analyze per-layer expert usage
    for layer_idx, scores in enumerate(gating_scores):
        expert_usage = scores.mean(dim=(0, 1))  # (num_experts,)
        print(f"Layer {layer_idx}: {expert_usage}")
        
        # Check for expert specialization
        entropy = -(expert_usage * expert_usage.log()).sum()
        print(f"  Entropy: {entropy:.3f}")  # Low = specialized, High = uniform
```

## 🔍 Model Inspection

### Parameter Count

```python
def count_parameters(model):
    """
    Count trainable and frozen parameters.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"Trainable: {trainable:,} ({trainable/1e6:.1f}M)")
    print(f"Frozen: {frozen:,} ({frozen/1e6:.1f}M)")
    print(f"Total: {trainable + frozen:,} ({(trainable + frozen)/1e6:.1f}M)")

count_parameters(model)
# Output:
# Trainable: 1,801,792 (1.8M)
# Frozen: 8,550,000,000 (8550.0M)
# Total: 8,551,801,792 (8551.8M)
```

### Layer Scale Inspection

```python
# Check layer scales after training
layer_scales = model.whisper_encoder.layer_scales.detach().cpu().numpy()
print("Layer scales:", layer_scales)

# Typical range: 0.05 - 0.2
# Smaller scales = less steering influence
# Larger scales = more steering influence
```

## 🐛 Debugging

### Common Issues

**Gradient issues**:
```python
# Check for NaN gradients
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in {name}")
```

**Device mismatch**:
```python
# Ensure all components on same device
device = next(model.parameters()).device
print(f"Model device: {device}")

# Move data to correct device
input_features = input_features.to(device)
```

**Shape mismatches**:
```python
# Debug shapes
print(f"Audio features: {input_features.shape}")
print(f"Text tokens: {decoder_input_ids.shape}")
print(f"Labels: {labels.shape}")
```

## 📚 API Reference

### Model Classes

- `SteerMoEEfficientLayerWiseModel`: Main model class
- `EfficientLayerWiseSteeringWhisperEncoder`: Whisper + steering
- `EfficientLayerWiseSteeringConformerEncoder`: Conformer + steering

### Data Collators

- `DataCollatorSpeechSeqSeqWithPadding`: For Whisper
- `DataCollatorSpeechSeqSeqWithPaddingForConformer`: For Conformer

### Utility Functions

- `load_balancing_loss()`: MoE load balancing
- `prepare_dataset()`: Dataset preprocessing
- `prepare_dataset_for_conformer()`: Conformer preprocessing

## 🔗 Related Documentation

- Training scripts: [`scripts/README.md`](../scripts/README.md)
- Configuration files: [`configs/README.md`](../configs/README.md)
- Preprocessing: [`pre_process/README.md`](../pre_process/README.md)
- Main README: [`README.md`](../README.md)

---

**Need help?** Open an issue on GitHub or check the documentation.

