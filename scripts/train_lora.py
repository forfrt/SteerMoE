"""
Training script for LoRA-based ablation study model.

This script trains a LoRA-adapted encoder with frozen LLM decoder,
providing a baseline for comparison with SteerMoE.

 * author Ruitao Feng
 * created on 16-07-2025
 * github: https://github.com/forfrt
"""

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback, WhisperFeatureExtractor
from datasets import Audio, load_dataset, concatenate_datasets, DatasetDict, load_metric, load_from_disk
import tqdm
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(process)d - %(levelname)s - %(filename)s>%(funcName)s>%(lineno)d - %(message)s')

# Import our models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from steer_moe.lora_model import LoRAModel, LoRAWhisperEncoder
from steer_moe.utils import load_parquet_datasets, prepare_dataset, DataCollatorSpeechSeqSeqWithPaddingPrompt


def load_config(path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_parquet_datasets_for_lora(parquet_dirs: List[str]) -> DatasetDict:
    """Load and concatenate datasets from parquet directories."""
    print('Loading datasets...')
    dataset = DatasetDict()
    train_datasets = []

    for folder in tqdm.tqdm(parquet_dirs):
        tmp = load_from_disk(folder)
        train_datasets.append(tmp)

    concat = train_datasets[0]
    for tmp_dataset in tqdm.tqdm(train_datasets[1:]):
        concat = concatenate_datasets([concat, tmp_dataset])

    dataset['train'] = concat
    print(f"Total samples: {len(concat)}")
    return dataset


def filter_dataset_by_length(dataset: DatasetDict, max_audio_length: float = 30.0,
                           max_text_length: int = 448) -> DatasetDict:
    """Filter dataset by audio length and text length."""
    max_input_length = max_audio_length * 16000

    def filter_inputs(input_length):
        """Filter inputs with zero input length or longer than max_audio_length"""
        return 0 < input_length < max_input_length

    def filter_labels(labels):
        """Filter label sequences longer than max length"""
        return len(labels) < max_text_length

    print(f"Dataset before filter: {dataset}")

    # Filter by audio length
    if "input_length" in dataset['train'].column_names:
        dataset = dataset.filter(filter_inputs, input_columns=["input_length"])

    # Filter by label length
    if "labels" in dataset['train'].column_names:
        dataset = dataset.filter(filter_labels, input_columns=["labels"])

    print(f"Dataset after filter: {dataset}")
    return dataset


class LoRAAnalysisCallback(TrainerCallback):
    """Callback for analyzing LoRA adapter patterns during training."""

    def __init__(self, model, log_interval: int = 100):
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            self._log_lora_analysis()

    def _log_lora_analysis(self):
        """Log LoRA adapter analysis metrics."""
        if hasattr(self.model, 'whisper_encoder') and hasattr(self.model.whisper_encoder, 'lora_adapters'):
            # Log LoRA adapter parameter norms
            total_lora_params = 0
            for layer_key, adapters in self.model.whisper_encoder.lora_adapters.items():
                for adapter_name, adapter in adapters.items():
                    if hasattr(adapter, 'lora_A') and hasattr(adapter, 'lora_B'):
                        lora_A_norm = torch.norm(adapter.lora_A).item()
                        lora_B_norm = torch.norm(adapter.lora_B).item()
                        total_lora_params += adapter.lora_A.numel() + adapter.lora_B.numel()
                        if self.step_count % (self.log_interval * 10) == 0:  # Less frequent detailed logging
                            print(f"Layer {layer_key}, Adapter {adapter_name}: A_norm={lora_A_norm:.4f}, B_norm={lora_B_norm:.4f}")
            
            if self.step_count % (self.log_interval * 10) == 0:
                print(f"Total LoRA parameters: {total_lora_params:,}")


class GradientClippingCallback(TrainerCallback):
    """Callback for gradient clipping on LoRA adapters."""

    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def on_step_end(self, args, state, control, **kwargs):
        # Get model from kwargs
        model = kwargs.get('model', None)
        if model is None:
            return
        # Clip gradients for LoRA adapters
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_(param, self.max_norm)


def create_lora_optimizer(model, learning_rate: float = 1e-4,
                         lora_lr: float = 1e-3):
    """Create optimizer with different learning rates for LoRA adapters."""
    # Separate parameter groups
    lora_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_A' in name or 'lora_B' in name:
                lora_params.append(param)
            else:
                other_params.append(param)

    # Create parameter groups with different learning rates
    param_groups = [
        {'params': lora_params, 'lr': lora_lr, 'name': 'lora'},
        {'params': other_params, 'lr': learning_rate, 'name': 'other'}
    ]

    return torch.optim.AdamW(param_groups)


def save_custom(model, path):
    """Save custom model state."""
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "pytorch_model.bin"))


def load_custom(path, whisper_encoder, llm_decoder, **kwargs):
    """Load custom model state."""
    model = LoRAModel(
        whisper_encoder=whisper_encoder,
        llm_decoder=llm_decoder,
        **kwargs
    )
    sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    return model


def train_lora_model(config_path: str = 'configs/lora_whisper_qwen7b_libri_train.yaml',
                     deepspeed_config_path: str = 'configs/stage2_simple.json',
                     eval_dataset_name: Optional[str] = None,
                     custom_test_set_path: Optional[str] = None,
                     resume_from_checkpoint: Optional[str] = None):
    """
    Train LoRA model with frozen LLM decoder.
    
    Args:
        config_path: Path to configuration file
        deepspeed_config_path: Path to DeepSpeed configuration
        eval_dataset_name: Name of evaluation dataset
        custom_test_set_path: Path to custom test set
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Load configuration
    config = load_config(config_path)

    print("Loading LLM decoder...")
    llm_decoder = AutoModelForCausalLM.from_pretrained(config['llm_decoder']['model_name'])
    llm_decoder.eval()
    for p in llm_decoder.parameters():
        p.requires_grad = False

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['llm_decoder']['model_name'])
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config['whisper_encoder']['model_path'])
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create LoRA-adapted Whisper encoder
    print("Creating LoRA-adapted Whisper encoder...")
    lora_whisper = LoRAWhisperEncoder(
        whisper_encoder_path=config['whisper_encoder']['model_path'],
        lora_rank=config['lora']['rank'],
        lora_alpha=config['lora'].get('alpha', 1.0),
        lora_dropout=config['lora'].get('dropout', 0.0),
        target_modules=config['lora'].get('target_modules', None),
        pooling_kernel_size=config.get('pooling_kernel_size', None),
        pooling_type=config.get('pooling_type', None),
        pooling_position=config.get('pooling_position', 32)
    )

    # Create main model
    print("Creating LoRA model...")
    model = LoRAModel(
        whisper_encoder=lora_whisper,
        llm_decoder=llm_decoder,
        max_prompt_tokens=config.get('max_prompt_tokens', 2048),
        use_adapter=config.get('use_adapter', True)
    )

    logging.info(f"model: {model}")

    # Set model to training mode
    model.train()

    # Load dataset
    print("Loading dataset...")
    parquet_dirs = config.get('parquet_dirs', [])
    batch_size = config['training']['batch_size']

    logging.info(f"batch_size: {batch_size}")

    # Expand parquet directories if they contain subdirectories
    expanded_dirs = []
    for parquet_dir in parquet_dirs:
        if os.path.isdir(parquet_dir):
            subdirs = [os.path.join(parquet_dir, d) for d in os.listdir(parquet_dir) 
                      if os.path.isdir(os.path.join(parquet_dir, d))]
            if subdirs:
                expanded_dirs.extend(subdirs)
            else:
                expanded_dirs.append(parquet_dir)
        else:
            expanded_dirs.append(parquet_dir)

    dataset = load_parquet_datasets_for_lora(expanded_dirs)
    print(f"Dataset: {type(dataset)} {dataset}")

    # Dataset is already preprocessed with input_features and labels
    processed_dataset = dataset['train']

    # Create data collator
    max_length = config.get('max_text_length', 512)

    data_collator = DataCollatorSpeechSeqSeqWithPaddingPrompt(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_length=max_length,
        audio_column="input_features",
        text_column="labels",
        prompt_column="text_prompt",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        per_device_train_batch_size=batch_size,
        num_train_epochs=config['training']['epochs'],
        save_strategy="steps",
        logging_dir=config['training']['logging_dir'],
        deepspeed=deepspeed_config_path if config.get('use_deepspeed', False) else None,
        fp16=config.get('fp16', True),
        report_to=["none"],
        load_best_model_at_end=False,
        save_total_limit=config.get('save_total_limit', 2),
        warmup_steps=config.get('warmup_steps', 0),
        weight_decay=config.get('weight_decay', 0.01),
        logging_steps=config.get('logging_steps', 10),
        save_steps=config.get('save_steps', 1000),
        dataloader_drop_last=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_safetensors=False,
    )

    # Create callbacks
    callbacks = []

    # Add LoRA analysis callback
    if config.get('log_lora_analysis', True):
        lora_callback = LoRAAnalysisCallback(model, log_interval=config.get('lora_log_interval', 100))
        callbacks.append(lora_callback)

    # Add gradient clipping callback
    if config.get('clip_lora_gradients', True):
        gradient_callback = GradientClippingCallback(max_norm=config.get('lora_gradient_clip', 1.0))
        callbacks.append(gradient_callback)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Start training
    print("Starting training...")
    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint if resume_from_checkpoint is not None else False,
    )

    # Save final model
    final_output_dir = os.path.join(config['training']['output_dir'], 'final')
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    save_custom(model, final_output_dir)

    print(f"Training completed. Model saved to {final_output_dir}")

    return trainer, model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train LoRA model')
    parser.add_argument('--config', type=str, default='configs/lora_whisper_qwen7b_libri_train.yaml',
                       help='Path to configuration file')
    parser.add_argument('--deepspeed_config', type=str, default='configs/stage2_simple.json',
                       help='Path to DeepSpeed configuration')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    train_lora_model(
        config_path=args.config,
        deepspeed_config_path=args.deepspeed_config,
        resume_from_checkpoint=args.resume_from
    )
