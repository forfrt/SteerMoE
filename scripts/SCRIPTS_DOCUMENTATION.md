# Scripts Documentation

This document describes the functionality of each Python script in the `scripts` directory.

## Overview

The scripts directory contains training, evaluation, and utility scripts for the SteerMoE project. These scripts support different model architectures (SteerMoE, LoRA, Conformer) and training configurations.

---

## Training Scripts

### `train.py`
**Purpose**: Main training script for original SteerMoE models (non-layer-wise variants).

**Key Features**:
- Supports both `SteerMoEModel` and `SteerMoEHybridModel`
- DeepSpeed integration for distributed training
- HuggingFace Trainer integration
- Evaluation on standard ASR datasets (LibriSpeech, AISHELL, etc.)
- Custom test set evaluation support
- Checkpoint resuming functionality
- HuggingFace Hub integration for model sharing

**Main Functions**:
- `train_with_deepspeed()`: Main training function with DeepSpeed support
- `load_parquet_datasets_for_steermoe()`: Load and concatenate datasets from parquet directories
- `filter_dataset_by_length()`: Filter datasets by audio and text length constraints
- `save_trainer_model()` / `load_trainer_model()`: Save/load models in HuggingFace format
- `push_to_hub()` / `load_from_hub()`: HuggingFace Hub integration

**Usage**: General purpose training script for SteerMoE baseline models.

---

### `train_layer_wise.py`
**Purpose**: Training script for layer-wise steering SteerMoE models using Whisper encoder.

**Key Features**:
- Uses `EfficientLayerWiseSteeringWhisperEncoder` for layer-wise steering
- Implements `SteerMoEEfficientLayerWiseModel`
- Custom callbacks for steering analysis and gradient clipping
- Supports preprocessed datasets with `input_features` and `labels` columns
- Uses `DataCollatorSpeechSeqSeqWithPadding` data collator
- Validation split creation if not present
- Custom compute metrics for CER/WER evaluation

**Main Functions**:
- `train_layer_wise_steermoe()`: Main training function
- `evaluate_layer_wise_model()`: Evaluation function with manual evaluation loop
- `analyze_steering_patterns()`: Analyze steering vector patterns
- `SteeringAnalysisCallback`: Log steering vector norms and layer scales
- `GradientClippingCallback`: Clip gradients for steering vectors

**Usage**: Training SteerMoE models with layer-wise steering mechanisms on Whisper encoder.

---

### `train_layer_wise_linear_whisper.py`
**Purpose**: Training script for layer-wise steering with linear routing variant.

**Key Features**:
- Similar to `train_layer_wise.py` but uses `SteerMoEEfficientLayerWiseModelLinear`
- Linear routing mechanism for expert selection
- Same training infrastructure as layer-wise variant

**Main Functions**:
- `train_layer_wise_steermoe()`: Training function for linear routing variant
- Similar callback and evaluation structure as `train_layer_wise.py`

**Usage**: Training SteerMoE with linear routing instead of standard routing mechanism.

---

### `train_layer_wise_prompt.py`
**Purpose**: Training script for layer-wise models with textual prompt support.

**Key Features**:
- Uses `DataCollatorSpeechSeqSeqWithPaddingPrompt` data collator
- Supports `text_prompt` column in datasets
- Similar architecture to `train_layer_wise.py` but with prompt integration
- Allows textual prompts to guide the model during training

**Main Functions**:
- `train_layer_wise_steermoe()`: Training function with prompt support
- Uses prompt-aware data collator for batch preparation

**Usage**: Training SteerMoE models with textual prompt conditioning.

---

### `train_layer_wise_conformer.py`
**Purpose**: Training script for layer-wise steering using Conformer encoder instead of Whisper.

**Key Features**:
- Uses Conformer encoder architecture
- `ASRFeatExtractor` for feature extraction instead of WhisperFeatureExtractor
- Uses `DataCollatorSpeechSeqSeqWithPaddingForConformer` data collator
- `SteerMoEEfficientLayerWiseModelForConformer` model class
- Supports `input_lengths` for variable-length sequences
- CMVN (Cepstral Mean and Variance Normalization) support

**Main Functions**:
- `train_layer_wise_steermoe_for_conformer()`: Main training function
- `evaluate_layer_wise_model()`: Evaluation with Conformer-specific handling
- Model class: `SteerMoEEfficientLayerWiseModelForConformer`

**Usage**: Training SteerMoE models with Conformer speech encoder instead of Whisper.

---

### `train_layer_wise_conformer_clothoaqa.py`
**Purpose**: Extended version of Conformer training script, likely with additional features for ClothoAQA dataset or specific configurations.

**Key Features**:
- Extended functionality for Conformer-based models
- Possibly includes additional dataset handling or evaluation metrics
- Similar structure to `train_layer_wise_conformer.py`

**Usage**: Specialized Conformer training with extended features.

---

### `train_conformer_linear.py`
**Purpose**: Training script for Conformer-based models with linear routing mechanism.

**Key Features**:
- Combines Conformer encoder with linear routing
- Similar to `train_layer_wise_linear_whisper.py` but for Conformer architecture

**Usage**: Training Conformer-based SteerMoE with linear routing.

---

### `train_lora.py`
**Purpose**: Training script for LoRA (Low-Rank Adaptation) ablation study baseline.

**Key Features**:
- Implements LoRA adapters on Whisper encoder layers
- Freezes LLM decoder completely
- Uses `LoRAWhisperEncoder` and `LoRAModel` classes
- Supports LoRA rank, alpha, and dropout configuration
- Custom callbacks for LoRA analysis and gradient clipping
- Uses `DataCollatorSpeechSeqSeqWithPaddingPrompt` data collator
- Supports pooling operations for downsampling

**Main Functions**:
- `train_lora_model()`: Main training function
- `LoRAAnalysisCallback`: Log LoRA adapter parameter norms
- `GradientClippingCallback`: Clip gradients for LoRA parameters
- `create_lora_optimizer()`: Create optimizer with different learning rates for LoRA adapters
- `save_custom()` / `load_custom()`: Save/load custom model state

**Usage**: Training LoRA baseline for ablation studies comparing against SteerMoE.

---

## Evaluation Scripts

### `eval_lora.py`
**Purpose**: Evaluation script for trained LoRA models.

**Key Features**:
- Loads trained LoRA models from checkpoints
- Manual evaluation loop with generation
- Computes CER, WER, token-level accuracy, and raw string accuracy
- Uses `DataCollatorSpeechSeqSeqWithPaddingPrompt` for batch preparation
- Handles prompt extraction and cleaning in predictions

**Main Functions**:
- `evaluate_lora_model()`: Main evaluation function
- Supports loading from checkpoint with proper model reconstruction

**Usage**: Evaluate trained LoRA models on test datasets.

---

## Utility Scripts

### `example_training.py`
**Purpose**: Example script demonstrating how to use training functions.

**Key Features**:
- Contains example functions for different training scenarios
- Demonstrates dataset loading
- Shows custom configuration creation
- Includes examples for both original and hybrid models

**Usage**: Reference/template for users to understand how to use training scripts.

---

### `cer.py`
**Purpose**: Character Error Rate (CER) metric implementation for HuggingFace datasets.

**Key Features**:
- Custom metric for computing CER
- Compatible with HuggingFace datasets `load_metric()` function
- Calculates edit distance at character level

**Usage**: Used by training/evaluation scripts to compute CER metrics.

---

### `wer.py`
**Purpose**: Word Error Rate (WER) metric implementation for HuggingFace datasets.

**Key Features**:
- Custom metric for computing WER
- Compatible with HuggingFace datasets `load_metric()` function
- Calculates edit distance at word level

**Usage**: Used by training/evaluation scripts to compute WER metrics.

---

## Shell Scripts

### `lora_train.sh`
**Purpose**: Shell script to launch LoRA training with DeepSpeed.

**Features**:
- Uses DeepSpeed for distributed training
- Specifies GPU devices (localhost:1,2)
- Configures config file path
- Sets resume checkpoint path
- Redirects output to log file

**Usage**: Launch LoRA training from command line with proper DeepSpeed configuration.

---

## Common Patterns Across Scripts

### Dataset Loading
Most scripts use a similar pattern:
1. Load configuration from YAML file
2. Expand parquet directories (handles subdirectories)
3. Load datasets using `load_from_disk()` and concatenate
4. Optionally filter by audio/text length

### Model Initialization
Common pattern:
1. Load frozen LLM decoder
2. Initialize tokenizer and feature extractor
3. Create encoder (Whisper/Conformer/LoRA variants)
4. Create main model combining encoder and decoder
5. Set training mode

### Training Arguments
Standard HuggingFace TrainingArguments with:
- DeepSpeed configuration support
- FP16 training
- Custom logging and saving strategies
- DataLoader configuration (workers, pin_memory, drop_last)

### Custom Callbacks
Most training scripts include:
- Analysis callbacks (steering/LoRA analysis)
- Gradient clipping callbacks
- Optional early stopping

### Evaluation
Evaluation scripts typically:
1. Load model from checkpoint
2. Create DataLoader for evaluation set
3. Run generation loop
4. Compute metrics (CER, WER, accuracy)

---

## Configuration Requirements

Scripts expect YAML configuration files with sections like:
- `whisper_encoder` / `conformer_encoder`: Encoder configuration
- `llm_decoder`: LLM decoder configuration
- `training`: Training hyperparameters (batch_size, epochs, etc.)
- `steering` / `lora`: Architecture-specific parameters
- `parquet_dirs`: Dataset paths
- `textual_prompt`: Optional prompt text

---

## Notes

- All scripts use logging with consistent format
- Most scripts support DeepSpeed for distributed training
- Datasets are expected to be preprocessed with `input_features` and `labels` columns
- Custom save/load functions handle model state dictionaries separately from HuggingFace format
- Evaluation scripts use manual loops rather than Trainer.evaluate() for more control

