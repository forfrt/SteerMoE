# SteerMoE: Efficient Audio-Language Alignment with a Mixture-of-Experts Steering Module

[![Paper](https://img.shields.io/badge/Interspeech-2026-blue)](papers/Interspeech_26_SteerMOE/camera_ready.pdf)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

English | [中文](README.md)

**SteerMoE enables powerful audio-language models that understand both speech and text while preserving the full reasoning capabilities of large language models.**

Unlike traditional approaches that compromise language understanding for audio processing, our method keeps the LLM completely frozen, ensuring your audio-language model maintains sophisticated textual inference, reasoning, and generation abilities—while achieving strong performance on audio understanding tasks.

## 🎯 What We Achieve

### Audio + Language Understanding with Full LLM Reasoning

Our models can:
- ✅ **Transcribe speech** with competitive accuracy (2.42% WER on LibriSpeech with Conformer)
- ✅ **Answer questions about audio** (52.35% accuracy on Clotho-AQA)
- ✅ **Reason about audio content** using the LLM's powerful inference
- ✅ **Maintain full textual capabilities** (frozen LLM preserves all language understanding)
- ✅ **Work across multiple languages** (English, Chinese, etc.)

### Key Innovation: Frozen Architecture

**Problem**: Traditional audio-language models fine-tune the LLM, which degrades its sophisticated language reasoning abilities.

**Our Solution**: Keep both the audio encoder AND the language decoder completely frozen. Train only a lightweight alignment module (~1.8M parameters) that bridges the two modalities.

**Result**: Best of both worlds—strong audio understanding + preserved LLM reasoning.

## 📊 Performance Highlights

### English ASR (LibriSpeech test-clean)

| Model | WER ↓ | Textual Reasoning | Trainable Params |
|-------|-------|-------------------|------------------|
| Whisper-large-v3 (frozen) | 2.7% | ❌ No LLM | 1550M |
| Encoder-LoRA (tuned encoder) | 2.51% | ✅ Preserved | 15.5M |
| SteerMoE (W7B) | 5.69% | ✅ **Fully Preserved** | **1.8M** |
| SteerMoE (C3B) | 3.26% | ✅ **Fully Preserved** | **1.8M** |
| **SteerMoE (C7B)** | **2.42%** | ✅ **Fully Preserved** | **1.8M** |

### Chinese ASR (AISHELL-2)

| Model | Test CER ↓ | Trainable Params |
|-------|-----------|------------------|
| Whisper-large-v3 (frozen) | 4.96% | 1550M |
| SteerMoE (W7B) | 5.96% | **1.8M** |
| SteerMoE (C3B) | 3.44% | **1.8M** |
| **SteerMoE (C7B)** | **2.50%** | **1.8M** |

### Audio Question Answering (Clotho-AQA)

| Model | Accuracy ↑ | Total Params | Trainable Params |
|-------|-----------|--------------|------------------|
| Kimi-Audio | 71.24% | 9.77B | Undisclosed (LLM fine-tuned) |
| Step-Audio-Chat | 45.84% | 130B | Undisclosed (LLM fine-tuned) |
| **SteerMoE (W7B)** | **52.35%** | 7B+1.5B | **1.8M** |
| SteerMoE (C3B) | 46.24% | 3B+1.5B | **1.8M** |
| SteerMoE (C7B) | 49.06% | 7B+1.5B | **1.8M** |

**Key Insight**: We achieve **competitive audio performance** with **fully preserved LLM reasoning** and only **1.8M trainable parameters** (~0.02% of the full model size). SteerMoE outperforms the 130B Step-Audio-Chat model by over 6 points while training only 1.8M parameters with a frozen decoder.

## 💡 Why This Matters

### Preserved Language Capabilities

Your audio-language model maintains ALL the LLM's abilities:

```python
# After training on audio tasks, the LLM still excels at pure text:

# Complex reasoning (preserved)
prompt = "If Alice has twice as many apples as Bob, and Bob has 3 apples,
          considering a 15% tax, how much would Alice pay for her apples at $2 each?"
model.generate(prompt)  # ✅ Works perfectly - LLM reasoning intact

# Code generation (preserved)
prompt = "Write a Python function to implement binary search"
model.generate(prompt)  # ✅ Still generates correct code

# Audio understanding (newly acquired)
audio = load_audio("speech.wav")
prompt = "Transcribe and summarize the main points: "
model.generate(audio, prompt)  # ✅ Understands audio + reasons about content
```

**Why this is important**:
- Deploy ONE model for both audio and text tasks
- No compromise on language understanding quality
- LLM's common-sense reasoning helps with audio understanding
- Safe to deploy in production (no unexpected behavior changes)

## 🔬 The SteerMoE Technology

### How We Achieve This: Layer-Wise Steering with Mixture-of-Experts

To bridge frozen audio encoders and frozen LLMs without fine-tuning either, we introduce **SteerMoE**—a lightweight, trainable alignment module that dynamically "steers" audio features into the LLM's representation space.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Audio Input (e.g., "Hello world" speech)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │   Frozen Audio Encoder        │  ← Whisper/Conformer
         │   (1.5B params, frozen)       │     NO training
         └───────────────┬───────────────┘
                         │ Audio features
                         ▼
         ┌───────────────────────────────┐
         │      SteerMoE Aligner         │  ← Our innovation
         │   Layer-wise Steering + MoE   │     ~1.8M params
         │   (ONLY trainable part)       │     Dynamic adaptation
         └───────────────┬───────────────┘
                         │ Aligned features
                         ▼
         ┌───────────────────────────────┐
         │    Linear Projection          │  ← Simple adapter
         │   (1280 → 896 dimensions)     │     ~1M params
         └───────────────┬───────────────┘
                         │ LLM-compatible embeddings
                         ▼
         ┌───────────────────────────────┐
         │   Frozen Language Decoder     │  ← Qwen/LLaMA
         │   (7B params, frozen)         │     NO training
         │   Reasoning preserved ✓       │     All capabilities intact
         └───────────────┬───────────────┘
                         │
                         ▼
         Text output: "Hello world" (+ reasoning/QA/etc.)
```

#### The Core Idea: Layer-Wise Dynamic Steering

Instead of learning a single static transformation, SteerMoE applies **adaptive adjustments at each encoder layer** based on the input content:

```python
# For each audio encoder layer l:
for layer_idx in range(num_layers):
    # 1. Process through frozen encoder layer
    h_l = frozen_encoder_layer[layer_idx](h_l_minus_1)

    # 2. MoE router decides which experts to use (depends on audio content)
    expert_weights = Router(h_l)  # Different for speech/music/noise/etc.

    # 3. Apply dynamic steering adjustment
    steering = Σ expert_weights[k] * steering_vectors[layer_idx, k]

    # 4. Adjust the features
    h_l = h_l + layer_scale[layer_idx] * steering
```

**Why this works**:
- 🎯 **Content-adaptive**: Router learns to select different experts for different audio types
- 🔀 **Layer-specific**: Early layers focus on acoustic features, later layers on semantic alignment
- 📊 **Efficient**: Single router for all layers (32× fewer parameters than naive MoE)
- 🎚️ **Controllable**: Layer scales allow fine-grained adjustment strength per layer

#### What Makes It "Mixture-of-Experts"?

Each layer has **multiple expert steering vectors** (typically 8):

```
Layer 0:  [Expert_0: acoustic patterns] [Expert_1: noise handling] [Expert_2: music] ...
Layer 1:  [Expert_0: phonetic features] [Expert_1: pitch variation] ...
...
Layer 31: [Expert_0: semantic concepts] [Expert_1: context alignment] ...
```

The **router network** learns to:
- Select **Expert_0** for clean speech
- Select **Expert_1** for noisy audio
- Select **Expert_2** for background music
- Mix experts for complex audio scenes

This **dynamic specialization** is why SteerMoE outperforms static adapters.

### Technical Benefits

#### 1. Parameter Efficiency (1000× Reduction)

Traditional fine-tuning:
```
Trainable: 1.5B (audio encoder) + 7B (LLM) = 8.5B parameters
Training time: ~500 GPU hours
GPU memory: 8× A100 80GB
```

SteerMoE:
```
Trainable: 1.8M parameters (steering + projection only)
Training time: ~10 GPU hours
GPU memory: 1× A100 40GB
Risk: Minimal (LLM behavior unchanged)
```

**Breakdown of 1.8M parameters**:
- Steering vectors: `32 layers × 8 experts × 1280 dim` = 327K params
- Router network: `1280 dim → (8×32)` = 327K params
- Layer scales: `32` = 32 params
- Linear projection: `1280 → 896` = 1.1M params
- **Total: ~1.8M params (0.02% of full model)**

#### 2. Preserved Generalization

Because the audio encoder stays frozen:
- ✅ Keeps Whisper's robustness to accents, noise, etc.
- ✅ No overfitting to your specific dataset
- ✅ Works on out-of-domain audio without degradation

Because the LLM stays frozen:
- ✅ All textual reasoning capabilities preserved
- ✅ No catastrophic forgetting
- ✅ Safe for production deployment

#### 3. Fast Iteration & Flexibility

- 🔄 Experiment with different audio encoders (Whisper, Conformer, etc.)
- 🔄 Swap LLM backbones (Qwen, LLaMA, Mistral, etc.)
- 🔄 Train for new languages in hours, not weeks
- 🔄 Easily adapt to new tasks (ASR → QA → captioning)

## 🏗️ Architectural Variants

We provide multiple model configurations:

| Encoder | Best For | Languages | Training Time | Trainable Params |
|---------|----------|-----------|---------------|------------------|
| **Whisper-large-v3** | General ASR, English | 90+ languages | ~10 hours | 1.8M |
| **Conformer** | Chinese/Asian, Streaming | Chinese, Japanese, Korean | ~12 hours | 1.8M |

### Model Architecture Details

- **SteerMoE (Whisper/Conformer)**: Uses layer-wise steering with MoE routing. Trainable params: **1.8M** (steering vectors + router + projection).
- **Encoder-LoRA (baseline)**: Baseline approach using LoRA adapters on the Whisper encoder layers. Trainable params: **15.5M** (LoRA adapters on attention and FFN modules). Preserves LLM reasoning but uses ~8.6× more parameters than SteerMoE.

Both SteerMoE variants use the **same technology**, just with different audio encoders.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/SteerMoE.git
cd SteerMoE

# Create environment
conda create -n steermoe python=3.10
conda activate steermoe
pip install -r requirements.txt

# Download pre-trained models
# Whisper: openai/whisper-large-v3
# LLM: Qwen/Qwen2.5-7B-Instruct
```

### 1. Preprocess Your Dataset

```bash
# For English (LibriSpeech)
python pre_process/pre_process_librispeech.py \
  --audio_dir /path/to/LibriSpeech/train-clean-100 \
  --output_dir /path/to/processed_librispeech \
  --whisper_model /path/to/whisper-large-v3 \
  --llm_tokenizer /path/to/Qwen2.5-7B-Instruct

# For Chinese (AISHELL-2)
python pre_process/pre_process_aishell.py \
  --audio_dir /path/to/aishell2/wav \
  --trans_file /path/to/aishell2/trans.txt \
  --output_dir /path/to/processed_aishell2
```

See [`pre_process/README.md`](pre_process/README.md) for other datasets.

### 2. Configure Training

Edit `configs/layer_wise_whisper_qwen7b_libri_train.yaml`:

```yaml
# Audio encoder (frozen)
whisper_encoder:
  model_path: "/path/to/whisper-large-v3"

# Language decoder (frozen)
llm_decoder:
  model_name: "/path/to/Qwen2.5-7B-Instruct"

# SteerMoE settings (trainable)
steering:
  num_experts: 8
  steering_scale: 0.1
  steering_learning_rate: 1e-2  # Higher LR for steering

# Dataset
parquet_dirs:
  - "/path/to/processed_librispeech/train.clean.100/"

# Task prompt
textual_prompt: "please transcribe the audio content into text: "
```

### 3. Train SteerMoE

```bash
# Single GPU
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --mode train

# Multi-GPU (recommended)
deepspeed --num_gpus=4 scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_train.yaml \
  --deepspeed_config configs/stage2_simple.json \
  --mode train
```

Training on LibriSpeech-100h takes ~10 hours on 4× A100 GPUs.

### 4. Evaluate

```bash
python scripts/train_layer_wise.py \
  --config configs/layer_wise_whisper_qwen7b_libri_test.yaml \
  --mode eval \
  --model_path results/steermoe_checkpoint/final
```

### 5. Use for Inference

```python
from transformers import AutoTokenizer
from steer_moe.models import SteerMoEEfficientLayerWiseModel
import torch

# Load model
model = SteerMoEEfficientLayerWiseModel.load(
    checkpoint_path="results/steermoe_checkpoint/final"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Load and preprocess audio
audio_features = preprocess_audio("speech.wav")  # (1, 128, T)

# Transcribe
prompt = tokenizer("Transcribe: ", return_tensors="pt").input_ids
output_ids = model.generate(
    input_features=audio_features,
    decoder_input_ids=prompt,
    max_new_tokens=256
)
transcription = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(transcription)

# Question answering (same model!)
prompt = tokenizer("What emotion is expressed in the audio? ", return_tensors="pt").input_ids
output_ids = model.generate(input_features=audio_features, decoder_input_ids=prompt)
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(answer)  # ✅ Uses LLM reasoning to analyze emotion
```

## 📖 Documentation

Comprehensive guides for each component:

- **[`configs/README.md`](configs/README.md)** | **[`configs/README_CN.md`](configs/README_CN.md)** - Configuration files and hyperparameters
- **[`pre_process/README.md`](pre_process/README.md)** | **[`pre_process/README_CN.md`](pre_process/README_CN.md)** - Dataset preprocessing for ASR, QA, etc.
- **[`scripts/README.md`](scripts/README.md)** | **[`scripts/README_CN.md`](scripts/README_CN.md)** - Training, evaluation, and analysis scripts
- **[`steer_moe/README.md`](steer_moe/README.md)** | **[`steer_moe/README_CN.md`](steer_moe/README_CN.md)** - Core model implementation details

## 🔬 Ablation Studies

We validate SteerMoE's design through comprehensive ablations:

### SteerMoE Ablation: Number of Experts

| Num Experts | WER ↓ | Trainable Params |
|-------------|-------|------------------|
| 16 | 2.43% | 2.5M |
| **8 (Default)** | **2.42%** | **1.8M** |
| 4 | 3.10% | 1.5M |
| 2 | 6.22% | 1.3M |
| Static Adapter (No MoE) | >100% | 1.1M |

**Conclusion**: The dramatic performance collapse from SteerMoE (2 Experts) to Static Adapter (No MoE) confirms that dynamic steering is essential—without content-adaptive routing, the static projection fails to bridge the audio-text representational gap, producing largely unintelligible outputs. SteerMoE demonstrates that 8 experts provide the optimal trade-off: while 16 experts offer only a marginal 0.01% WER improvement, the 39% increase in trainable parameters makes 8 experts the most efficient configuration.

## 📁 Project Structure

```
SteerMoE/
├── configs/              # Training configurations
│   ├── layer_wise_whisper_qwen7b_libri_train.yaml
│   ├── layer_wise_conformer_qwen7b_aishell_train.yaml
│   └── README.md / README_CN.md
├── pre_process/          # Dataset preprocessing
│   ├── pre_process_librispeech.py
│   ├── pre_process_aishell.py
│   ├── pre_process_clothoaqa.py
│   └── README.md / README_CN.md
├── scripts/              # Training and evaluation
│   ├── train_layer_wise.py              # Main training (Whisper)
│   ├── train_layer_wise_conformer.py    # Main training (Conformer)
│   ├── train_layer_wise_linear_whisper.py  # Ablation baseline
│   ├── cer.py, wer.py                    # Evaluation metrics
│   └── README.md / README_CN.md
├── steer_moe/            # Core implementation
│   ├── models.py                         # SteerMoE model classes
│   ├── efficient_layer_wise_whisper.py  # Whisper + steering
│   ├── efficient_layer_wise_conformer.py # Conformer + steering
│   ├── utils.py                          # Data collators
│   └── README.md / README_CN.md
├── papers/               # Research papers
│   ├── Interspeech_26_SteerMOE/         # Interspeech 2026 submission
│   └── ICASSP_SteerMoE/                 # ICASSP 2025 submission
├── results/              # Training outputs
└── README.md / README_CN.md  # This file
```

## 🎓 Research Background

### The Problem with Traditional Audio-LLM Approaches

Most audio-language models use one of these approaches:

**Approach 1: Fine-tune the entire LLM**
```
Audio → Encoder → [Fine-tuned LLM] → Output
                   ⚠️ 7B params trained
                   ⚠️ Language reasoning degrades
                   ⚠️ Expensive & slow training
```

**Approach 2: Adapter-based (simple projection)**
```
Audio → Encoder → [Linear] → [Frozen LLM] → Output
                   ✅ LLM preserved
                   ⚠️ Limited audio understanding
                   ⚠️ Static transformation
```

**Our Approach: SteerMoE**
```
Audio → Encoder → [SteerMoE: Dynamic Steering] → [Frozen LLM] → Output
                   ✅ LLM fully preserved
                   ✅ Strong audio understanding
                   ✅ Content-adaptive transformation
                   ✅ Only 1.8M params trained
```

### Key Insights from Our Research

1. **Freezing is better than fine-tuning**: Frozen LLM retains reasoning, frozen encoder retains robustness
2. **Dynamic beats static**: MoE routing adapts to different audio types better than fixed projection
3. **Layer-wise is crucial**: Different encoder layers need different alignment strategies
4. **Efficiency is achievable**: Single router reduces parameters by 32× vs. naive multi-router MoE

See our paper ([`papers/Interspeech_26_SteerMOE/camera_ready.pdf`](papers/Interspeech_26_SteerMOE/camera_ready.pdf)) for detailed analysis and more results.

## 💻 Hardware Requirements

### Training

| Configuration | GPUs | Batch Size | Training Time (LibriSpeech-100h) |
|---------------|------|------------|----------------------------------|
| Minimum | 1× A100 40GB | 1-2 | ~40 hours |
| Recommended | 4× A100 40GB | 4 per GPU | ~10 hours |
| Large scale | 8× A100 80GB | 8 per GPU | ~5 hours |

### Inference

| Model Size | GPU Memory | Tokens/sec |
|------------|------------|------------|
| Qwen-7B + Whisper | 16GB (FP16) | ~50 |
| Qwen-3B + Whisper | 8GB (FP16) | ~100 |

## 📧 Contact & Support

**Authors**:
- Ruitao Feng - [GitHub: @forfrt](https://github.com/forfrt) - ruitaofeng@outlook.com
- Bixi Zhang - [GitHub: @zbxforward](https://github.com/zbxforward) - bixizhang@hku.hk
- Sheng Liang - shengliang@outlook.com
- Zheng Yuan (Corresponding Author) - zheng.yuan@univ-amu.fr

**Get Help**:
- 🐛 **Bug reports**: [GitHub Issues](https://github.com/forfrt/SteerMoE/issues)
- 💬 **Questions**: [GitHub Discussions](https://github.com/forfrt/SteerMoE/discussions)

**Paper**: See [`papers/Interspeech_26_SteerMOE/camera_ready.pdf`](papers/Interspeech_26_SteerMOE/camera_ready.pdf) for the full Interspeech 2026 submission with detailed methodology and additional experiments.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work builds upon excellent open-source projects:

- **[Whisper](https://github.com/openai/whisper)** (OpenAI) - Robust speech recognition
- **[Qwen](https://github.com/QwenLM/Qwen)** (Alibaba) - Powerful multilingual LLM
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)** (Microsoft) - Efficient distributed training
- **[Transformers](https://github.com/huggingface/transformers)** (HuggingFace) - Model implementations and training utilities

We also thank the research community for datasets:
- LibriSpeech, AISHELL-2, Clotho-AQA, and other benchmark datasets
- Open-source audio processing libraries (librosa, soundfile, torchaudio)

## 🌟 Star History

If you find SteerMoE useful, please consider giving it a star! ⭐

---

**Built with ❤️ by the SteerMoE team. Questions? Open an issue or discussion!**
