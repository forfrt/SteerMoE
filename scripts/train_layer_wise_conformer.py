"""
 * author B.X. Zhang
 * created on 09-09-2025
 * github: https://github.com/zbxforward
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


from steer_moe.efficient_layer_wise_conformer import EfficientLayerWiseSteeringConformerEncoder,SteerMoEEfficientLayerWiseModelForConformer
from steer_moe.utils import DataCollatorSpeechSeqSeqWithPaddingForConformer
from steer_moe.conformer_module.asr_feat import ASRFeatExtractor

class SteeringAnalysisCallback(TrainerCallback):
    """Callback for analyzing steering patterns during training."""

    def __init__(self, model, log_interval: int = 100):
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            self._log_steering_analysis()

    def _log_steering_analysis(self):
        """Log steering analysis metrics."""
        if hasattr(self.model, 'conformer_encoder') and hasattr(self.model.conformer_encoder, 'layer_scales'):
            layer_scales = self.model.conformer_encoder.layer_scales.detach().cpu().numpy()
            print(f"Layer scales: {layer_scales}")

            # Log steering vector norms
            steering_norms = torch.norm(self.model.conformer_encoder.steering_vectors, dim=-1)
            avg_norms = steering_norms.mean(dim=1).detach().cpu().numpy()
            print(f"Average steering vector norms per layer: {avg_norms}")


class GradientClippingCallback(TrainerCallback):
    """Callback for gradient clipping on steering vectors."""

    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def on_step_end(self, args, state, control, **kwargs):
        # Get model from kwargs
        model = kwargs.get('model', None)
        if model is None:
            return
        # Clip gradients for steering vectors
        for name, param in model.named_parameters():
            if 'steering_vectors' in name or 'layer_scales' in name:
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_(param, self.max_norm)


def load_config(path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_parquet_datasets_for_steermoe(parquet_dirs: List[str]) -> DatasetDict:
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


def save_custom(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "pytorch_model.bin"))




def train_layer_wise_steermoe_for_conformer(config_path: str = 'configs/layer_wise.yaml',
                             deepspeed_config_path: str = 'configs/stage2_simple.json',
                             eval_dataset_name: Optional[str] = None,
                             custom_test_set_path: Optional[str] = None,
                             resume_from_checkpoint: Optional[str] = None):

    """
    Train SteerMoE model with layer-wise steering.
    
    Args:
        config_path: Path to configuration file
        deepspeed_config_path: Path to DeepSpeed configuration
        eval_dataset_name: Name of evaluation dataset
        custom_test_set_path: Path to custom test set
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Load configuration
    config = load_config(config_path)


    logging.info("Loading LLM decoder...")
    llm_decoder = AutoModelForCausalLM.from_pretrained(config['llm_decoder']['model_name'])
    llm_decoder.eval()
    for p in llm_decoder.parameters():
        p.requires_grad = False

    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['llm_decoder']['model_name'])
    
    """
    -----------------------------------------------
    Change Feature Extractor To Conformer Version
    -----------------------------------------------
    """
    cmvn_path = os.path.join(config['conformer_encoder']['model_path'], "cmvn.ark")   ###TODO!!!! ADD conformer_model_dir TO YAML 
    feature_extractor = ASRFeatExtractor(cmvn_path)
    """
    -----------------------------------------------
    Change Feature Extractor To Conformer Version
    -----------------------------------------------
    """
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # # Create layer-wise steering Whisper encoder
    # print("Creating layer-wise steering Conformer encoder...")
    # layer_wise_conformer = EfficientLayerWiseSteeringConformerEncoder(
    #     conformer_encoder_path=config['conformer_encoder']['model_path']+"/asr_encoder.pth.tar",    
    #     num_experts=config['steering']['num_experts'],
    #     steering_scale=config['steering']['steering_scale']
    # )

    # Create main model
    logging.info("Creating SteerMoE model...")
    model = SteerMoEEfficientLayerWiseModelForConformer(
        conformer_encoder_path=config['conformer_encoder']['model_path']+"/asr_encoder.pth.tar",
        llm_decoder=llm_decoder,
        num_experts=config['steering']['num_experts'],
        steering_scale=config['steering']['steering_scale'],
        max_prompt_tokens=config.get('max_prompt_tokens', 2048),
        use_adapter=config.get('use_adapter', True)
    )

    logging.info(f"model: {model}")

    # Set model to training mode
    model.train()


    logging.info("Loading dataset...")
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

    dataset = load_parquet_datasets_for_steermoe(expanded_dirs)
    logging.info(f"Dataset: {type(dataset)} {dataset}")

    # Filter dataset if needed
    # if config.get('filter_dataset', True):
    #     dataset = filter_dataset_by_length(
    #         dataset,
    #         max_audio_length=config.get('max_audio_length', 30.0),
    #         max_text_length=config.get('max_text_length', 448)
    #     )

    # Dataset is already preprocessed with input_features and labels
    # No need for additional preparation
    processed_dataset = dataset['train']

    # Create validation split
    if 'validation' in dataset:
        processed_val = dataset['validation']
    else:
        # Split train into train/val
        split_dataset = processed_dataset.train_test_split(test_size=0.05, seed=42)
        processed_val = split_dataset['test']
        processed_dataset = split_dataset['train']


    # Create data collator for preprocessed data
    textual_prompt = config.get('textual_prompt', "请转写以下音频内容为文字：") 
    max_length = config.get('max_text_length', 512)


    # Create data collator with fixed max length for consistent evaluation
    data_collator = DataCollatorSpeechSeqSeqWithPaddingForConformer(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        textual_prompt=textual_prompt,
        max_length=max_length,  # Fixed max length
        audio_column="input_features",  # Use preprocessed audio features
        text_column="labels"  # Use preprocessed labels
    )


    # Training arguments
    training_args = TrainingArguments(
        learning_rate=5e-4,
        output_dir=config['training']['output_dir'],
        per_device_train_batch_size=batch_size,
        # per_device_eval_batch_size=batch_size,
        num_train_epochs=config['training']['epochs'],
        # evaluation_strategy="steps",
        save_strategy="steps",
        logging_dir=config['training']['logging_dir'],
        deepspeed=deepspeed_config_path if config.get('use_deepspeed', False) else None,
        fp16=config.get('fp16', True),
        report_to=["none"],
        load_best_model_at_end=False,
        # load_best_model_at_end=True,
        # metric_for_best_model="cer",
        # greater_is_better=False,
        save_total_limit=config.get('save_total_limit', 2),
        warmup_steps=config.get('warmup_steps', 0),
        weight_decay=config.get('weight_decay', 0.01),
        logging_steps=config.get('logging_steps', 10),
        # eval_steps=config.get('eval_steps', 500),
        eval_steps=config.get('eval_steps', 50),
        save_steps=config.get('save_steps', 1000),
        dataloader_drop_last=True,  # Ensure consistent batch sizes
        remove_unused_columns=False,  # Keep all columns for our custom data collator
        # gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        dataloader_num_workers=4,  # Add parallel data loading
        dataloader_pin_memory=True,  # Pin memory for faster GPU transfer
        eval_accumulation_steps=1,  # Process evaluation batches immediately to avoid memory issues
        save_safetensors=False
    )

    # Create callbacks
    callbacks = []

    # Add steering analysis callback
    if config.get('log_steering_analysis', True):
        steering_callback = SteeringAnalysisCallback(model, log_interval=config.get('steering_log_interval', 100))
        callbacks.append(steering_callback)

    # Add gradient clipping callback
    if config.get('clip_steering_gradients', True):
        gradient_callback = GradientClippingCallback(max_norm=config.get('steering_gradient_clip', 1.0))
        callbacks.append(gradient_callback)

    # Add early stopping
    # if config.get('use_early_stopping', True):
    #     early_stopping = EarlyStoppingCallback(early_stopping_patience=config.get('early_stopping_patience', 3))
    #     callbacks.append(early_stopping)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        # eval_dataset=processed_val,
        data_collator=data_collator,
        # compute_metrics=compute_metrics_trainer,
        callbacks=callbacks,
    )

    # Start training
    logging.info("Starting training...")
    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint if resume_from_checkpoint is not None else False,
    )

    # Save final model
    final_output_dir = os.path.join(config['training']['output_dir'], 'final')
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    save_custom(model, final_output_dir)

    logging.info(f"Training completed. Model saved to {final_output_dir}")

    return trainer, model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train SteerMoE with layer-wise steering')
    parser.add_argument('--config', type=str, default='configs/layer_wise.yaml',
                       help='Path to configuration file')
    parser.add_argument('--deepspeed_config', type=str, default='configs/stage2_simple.json',
                       help='Path to DeepSpeed configuration')
    parser.add_argument('--eval_dataset', type=str, default=None,
                       help='Evaluation dataset name')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'analyze'], default='train',
                       help='Mode: train, eval, or analyze')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model for evaluation/analysis')
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    if args.mode == 'train':
        train_layer_wise_steermoe_for_conformer(
            config_path=args.config,
            deepspeed_config_path=args.deepspeed_config,
            eval_dataset_name=args.eval_dataset,
            resume_from_checkpoint=args.resume_from
        )
    elif args.mode == 'eval':
        if args.model_path is None:
            raise ValueError("Model path required for evaluation")
        evaluate_layer_wise_model(args.model_path, args.eval_dataset, args.config)
    elif args.mode == 'analyze':
        if args.model_path is None:
            raise ValueError("Model path required for analysis")
        analyze_steering_patterns(args.model_path)
