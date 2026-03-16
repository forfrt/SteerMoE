# Deprecated Directory

This folder contains earlier experimental implementations and baseline approaches that have been superseded by more efficient or effective versions in the main codebase.

## 📁 Purpose

These files document the evolution of the SteerMoE project and serve as:
- **Historical reference** for understanding design decisions
- **Educational examples** of different architectural approaches
- **Baseline comparisons** for ablation studies

⚠️ **Note**: These implementations are **not used in current experiments** but are preserved for reference and reproducibility.

## 📝 Files

### `aligner.py`

**Purpose**: Basic SteerMoE implementation with post-encoder steering.

**What it does**:
- Implements `SteerMoEAligner` - a simple module that applies steering vectors after the Whisper encoder
- Uses a single router (linear layer) to compute gating scores for each token
- Computes a weighted sum of steering vectors based on gating scores
- Adds the steering adjustment to the original audio representation

**Why deprecated**: This is a simplified, post-encoder-only approach. It lacks layer-wise control and was superseded by more sophisticated implementations that apply steering at each encoder layer for finer-grained control.

---

### `layer_wise_aligner.py`

**Purpose**: Layer-wise steering implementation with per-layer routers and steering vectors.

**What it does**:
- Implements `LayerWiseSteerMoEAligner` that applies steering at each of the 32 Whisper encoder layers
- Maintains separate routers for each layer (32 routers total)
- Maintains layer-specific steering vectors and scaling factors
- Processes audio through the encoder, applying steering adjustments at each layer
- Includes comparison analysis showing pros/cons of layer-wise vs post-encoder steering

**Why deprecated**: This approach has higher computational cost (32× more routing operations) and more parameters. The file itself contains analysis suggesting it may overfit and is harder to train. Replaced by more efficient alternatives like the shared router approach.

---

### `layer_wise_whisper.py`

**Purpose**: Modified Whisper encoder wrapper with layer-wise steering integrated directly.

**What it does**:
- Implements `LayerWiseSteeringWhisperEncoder` that wraps the original Whisper encoder
- Adds steering vectors and routers to each layer
- Freezes the original encoder parameters
- Applies steering adjustments layer-by-layer during forward pass
- Includes detailed analysis comparing layer-wise vs post-encoder steering approaches
- Provides recommendations on when to use each approach

**Why deprecated**: Similar to `layer_wise_aligner.py`, this represents an earlier attempt at layer-wise steering. The file's own analysis recommends starting with post-encoder steering and only moving to layer-wise if specific benefits are needed. Replaced by more efficient implementations.

---

### `shared_router_implementation.py`

**Purpose**: Exploration of different router architectures - comparing multiple routers vs single router vs shared router with layer-specific weights.

**What it does**:
- Implements `SharedRouterLayerWiseSteeringWhisperEncoder` - a hybrid approach using:
  - One shared router for all layers (reduces parameters)
  - Layer-specific weight matrices to transform the shared router output
  - Layer-specific scaling factors
- Includes detailed comparison of three router approaches:
  1. **Multiple routers** (one per layer) - most flexible but most parameters
  2. **Single router** - efficient but less flexible
  3. **Shared router with weights** - balanced approach
- Provides parameter count analysis and computational complexity comparison

**Why deprecated**: This file appears to be exploratory/analytical rather than a final implementation. It compares approaches and recommends the single router approach for efficiency. The shared router approach is more complex without clear benefits over simpler alternatives.

---

### `al_models.py`

**Purpose**: Audio-to-text model combining Whisper encoder with GPT-2 decoder through an adapter.

**What it does**:
- Implements `AudioToText` - an end-to-end ASR model
- Takes audio features from Whisper encoder and projects them to match GPT-2 embedding dimension
- Concatenates audio prompts with text token embeddings
- Passes combined embeddings through GPT-2 for text generation
- Includes inference example with greedy decoding
- Handles both training (with labels) and inference modes

**Why deprecated**: This is a basic audio-to-text architecture that doesn't incorporate the SteerMoE steering mechanism. It represents an earlier baseline approach before steering was integrated into the pipeline. Replaced by versions that incorporate steering vectors for better audio-to-text alignment.

---

## 🔍 Key Insights

These files document the evolution of the SteerMoE project:

1. **Early stage**: Simple post-encoder steering (`aligner.py`)
2. **Exploration phase**: Layer-wise steering attempts (`layer_wise_aligner.py`, `layer_wise_whisper.py`)
3. **Architecture analysis**: Router design comparisons (`shared_router_implementation.py`)
4. **Baseline model**: Basic audio-to-text without steering (`al_models.py`)

The deprecation suggests the project moved toward more efficient and effective implementations, likely consolidating insights from these experiments into a cleaner, production-ready version.

## 📊 Comparison with Current Implementation

| Aspect | Deprecated Versions | Current Implementation |
|--------|---------------------|------------------------|
| **Routing** | Multiple routers (32×) or shared router with weights | Single efficient router |
| **Parameters** | 10.5M (multi-router) or complex hybrid | 327K (32× reduction) |
| **Steering** | Post-encoder or per-layer with separate routers | Layer-wise with shared router |
| **Complexity** | Higher, more components | Simpler, more efficient |
| **Performance** | Baseline or experimental | Optimized and validated |

## 🔗 Related Documentation

- Current implementations: [`steer_moe/`](../steer_moe/)
- Training scripts: [`scripts/`](../scripts/)
- Architecture analysis: [`docs/SteerMoE_Architecture_Analysis.md`](../docs/SteerMoE_Architecture_Analysis.md)
