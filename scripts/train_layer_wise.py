"""
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

from steer_moe.models import SteerMoEEfficientLayerWiseModel,SteerMoEEfficientLayerWiseModelForConformer
from steer_moe.efficient_layer_wise_whisper import EfficientLayerWiseSteeringWhisperEncoder
from steer_moe.utils import load_parquet_datasets, prepare_dataset, DataCollatorSpeechSeqSeqWithPadding
from steer_moe.tokenizer.whisper_Lv3.whisper import WhisperEncoder


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


def compute_metrics(pred):
    """Compute CER and WER for evaluation."""
    cer_metric = load_metric('./cer.py')
    wer_metric = load_metric('./wer.py')
    pred_str = pred.predictions
    label_str = pred.label_ids
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer, "wer": wer}


def prepare_asr_dataset(dataset, audio_column: str, text_column: str,
                       whisper_encoder, tokenizer, sample_rate: int = 16000):
    """Prepare ASR dataset for training."""
    def _prepare(batch):
        return prepare_dataset(batch, audio_column, text_column, whisper_encoder, tokenizer, sample_rate)

    # Cast audio column to datasets.Audio if needed
    if not isinstance(dataset.features[audio_column], Audio):
        dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sample_rate))

    processed = dataset.map(_prepare, remove_columns=dataset.column_names)
    return processed


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
        if hasattr(self.model, 'whisper_encoder') and hasattr(self.model.whisper_encoder, 'layer_scales'):
            layer_scales = self.model.whisper_encoder.layer_scales.detach().cpu().numpy()
            print(f"Layer scales: {layer_scales}")

            # Log steering vector norms
            steering_norms = torch.norm(self.model.whisper_encoder.steering_vectors, dim=-1)
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


def create_layer_wise_optimizer(model, learning_rate: float = 1e-4,
                               steering_lr: float = 1e-3, router_lr: float = 1e-4):
    """Create optimizer with different learning rates for different components."""
    # Separate parameter groups
    steering_params = []
    router_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'steering_vectors' in name or 'layer_scales' in name:
                steering_params.append(param)
            elif 'router' in name:
                router_params.append(param)
            else:
                other_params.append(param)

    # Create parameter groups with different learning rates
    param_groups = [
        {'params': steering_params, 'lr': steering_lr, 'name': 'steering'},
        {'params': router_params, 'lr': router_lr, 'name': 'router'},
        {'params': other_params, 'lr': learning_rate, 'name': 'other'}
    ]

    return torch.optim.AdamW(param_groups)

# in your models.py
def save_custom(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "pytorch_model.bin"))

@staticmethod
def load_custom(path, whisper_encoder, llm_decoder, **kwargs):
    model = SteerMoEEfficientLayerWiseModel(
        whisper_encoder=whisper_encoder,
        llm_decoder=llm_decoder,
        **kwargs
    )
    sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    return model


def train_layer_wise_steermoe(config_path: str = 'configs/layer_wise.yaml',
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

    # Create layer-wise steering Whisper encoder
    print("Creating layer-wise steering Whisper encoder...")
    layer_wise_whisper = EfficientLayerWiseSteeringWhisperEncoder(
        whisper_encoder_path=config['whisper_encoder']['model_path'],
        num_experts=config['steering']['num_experts'],
        steering_scale=config['steering']['steering_scale']
    )

    # Create main model
    print("Creating SteerMoE model...")
    model = SteerMoEEfficientLayerWiseModel(
        whisper_encoder=layer_wise_whisper,
        llm_decoder=llm_decoder,
        num_experts=config['steering']['num_experts'],
        max_prompt_tokens=config.get('max_prompt_tokens', 2048),
        use_adapter=config.get('use_adapter', True)
    )

    logging.info(f"model: {model}")

    # Set model to training mode
    model.train()

    # Load dataset #TODO eliminate 
    '''
    ----------------------------------------------------------
    '''
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

    dataset = load_parquet_datasets_for_steermoe(expanded_dirs)
    print(f"Dataset: {type(dataset)} {dataset}")

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
    '''
    ----------------------------------------------------------
    '''

    # Create data collator for preprocessed data
    textual_prompt = config.get('textual_prompt', "请转写以下音频内容为文字：")  # Default Chinese prompt
    max_length = config.get('max_text_length', 1024)


    # Create data collator with fixed max length for consistent evaluation
    data_collator = DataCollatorSpeechSeqSeqWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        textual_prompt=textual_prompt,
        max_length=max_length,  # Fixed max length
        audio_column="input_features",  # Use preprocessed audio features
        text_column="labels"  # Use preprocessed labels
    )

    # Create custom compute metrics function
    def compute_metrics_trainer(eval_pred):
        logging.info(f"start evaluation: eval_pred: {eval_pred}")

        try:
            logits, labels = eval_pred
            preds = logits.argmax(-1)
        
            # Decode predictions and labels
            pred_str = []
            label_str = []
        
            for pred_seq, label_seq in zip(preds, labels):
                # Remove padding and special tokens from predictions
                if hasattr(pred_seq, '__len__'):
                    pred_tokens = pred_seq[pred_seq != tokenizer.pad_token_id]
                    pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    pred_str.append(pred_text)

                # Remove -100 tokens from labels
                if hasattr(label_seq, '__len__'):
                    label_tokens = label_seq[label_seq != -100]
                    label_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
                    label_str.append(label_text)
        
            if len(pred_str) > 0 and len(label_str) > 0:
                try:
                    cer_metric = load_metric('./scripts/cer.py')
                    wer_metric = load_metric('./scripts/wer.py')
                    cer = cer_metric.compute(predictions=pred_str, references=label_str)
                    wer = wer_metric.compute(predictions=pred_str, references=label_str)
                    return {"cer": cer, "wer": wer}
                except Exception as e:
                    logging.warning(f"Metrics computation failed: {e}")
                    return {"cer": 1.0, "wer": 1.0}  # Return default values
            else:
                return {"cer": 1.0, "wer": 1.0}  # Return default values
                
        except Exception as e:
            logging.error(f"Evaluation error: {e}")
            return {"cer": 1.0, "wer": 1.0}  # Return default values
        # cer_metric = load_metric('cer')
        # wer_metric = load_metric('wer')
        # cer = cer_metric.compute(predictions=pred_str, references=label_str)
        # wer = wer_metric.compute(predictions=pred_str, references=label_str)
        # logging.info(f"end evaluation: pred_str: {pred_str}, ref_str: {label_str}, cer: {cer}, wer: {wer}")
        # return {"cer": cer, "wer": wer}

    # Training arguments
    training_args = TrainingArguments(
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
        save_safetensors=False,
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


def evaluate_layer_wise_model(model_path: str, eval_dataset_name: str, config_path: str):
    """Evaluate a trained layer-wise SteerMoE model."""
    config = load_config(config_path)

    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)
    # rank = dist.get_rank()

    logging.info(f"local_rank: {local_rank}, device: {device}")

    # Load model
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

    # Create layer-wise steering Whisper encoder
    print("Creating layer-wise steering Whisper encoder...")
    layer_wise_whisper = EfficientLayerWiseSteeringWhisperEncoder(
        whisper_encoder_path=config['whisper_encoder']['model_path'],
        num_experts=config['steering']['num_experts'],
        steering_scale=config['steering']['steering_scale']
    )

    print("Creating SteerMoE model...")
    model = load_custom(
        model_path, layer_wise_whisper, llm_decoder,
        num_experts=config['steering']['num_experts'],
        max_prompt_tokens=config.get('max_prompt_tokens', 2048),
        use_adapter=config.get('use_adapter', True),
    )

    # model = SteerMoEEfficientLayerWiseModel(
    #     whisper_encoder=layer_wise_whisper,
    #     llm_decoder=llm_decoder,
    #     num_experts=config['steering']['num_experts'],
    #     max_prompt_tokens=config.get('max_prompt_tokens', 2048),
    #     use_adapter=config.get('use_adapter', True),
    # )

    # # 2. LOAD the saved state dictionary from your checkpoint.
    # #    map_location='cpu' is important to avoid one process using all GPU memory.
    # print(f"Loading model state from: {model_path}")
    # state_dict_path = os.path.join(model_path, "pytorch_model.bin")
    # state_dict = torch.load(state_dict_path, map_location="cpu")
    # model.load_state_dict(state_dict)

    model.to(device)
    model.half()
    model.eval()

    # Load evaluation dataset
    print("Loading dataset...")
    parquet_dirs = config.get('parquet_dirs', [])
    batch_size = config['training']['batch_size']

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
    print(f"Dataset: {type(dataset)} {dataset}")

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

    textual_prompt = config.get('textual_prompt', "请转写以下音频内容为文字：")  # Default Chinese prompt
    max_length = config.get('max_text_length', 1024)
    
    # Create data collator with fixed max length for consistent evaluation
    data_collator = DataCollatorSpeechSeqSeqWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        textual_prompt=textual_prompt,
        max_length=max_length,  # Fixed max length
        audio_column="input_features",  # Use preprocessed audio features
        text_column="labels"  # Use preprocessed labels
    )

    # Create a DataLoader for the evaluation set
    # NOTE: You should set a reasonable batch size here. 1 is safe, but you can try larger.
    eval_batch_size = config['training'].get('per_device_eval_batch_size', 1) 
    eval_dataloader = DataLoader(
        processed_val,
        batch_size=eval_batch_size,
        collate_fn=data_collator
    )

    all_preds = []
    all_labels = []
    device=model.device

    print("Starting manual evaluation loop...")
    textual_prompt = config.get('textual_prompt', "请转写以下音频内容为文字：")
    prompt_ids = tokenizer(textual_prompt, return_tensors="pt").input_ids.to(device)

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        # Fallback for some tokenizers
        eos_token_id = tokenizer.pad_token_id
        logging.warning(f"EOS token not found, using PAD token as EOS: {eos_token_id}")

    for step, batch in enumerate(tqdm.tqdm(eval_dataloader)):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_prompt_ids = prompt_ids.expand(batch["input_features"].shape[0], -1)
        
        with torch.no_grad():
            # Use model.generate() for efficient decoding
            # You might need to adjust max_length and other generation parameters
            generated_ids = model.generate(
                input_features=batch["input_features"],
                decoder_input_ids=batch_prompt_ids,
                # should not be the decoder_input_ids as it's used in the autoregressive training and contains the labels
                # decoder_input_ids=batch["decoder_input_ids"],
                max_new_tokens=512,  # Adjust as needed
                eos_token_id=eos_token_id,
            )

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        cleaned_preds = []
        for pred in decoded_preds:
            # Find the prompt in the prediction and take the text after it
            if textual_prompt in pred:
                cleaned_preds.append(pred.split(textual_prompt, 1)[1])
            else:
                cleaned_preds.append(pred) # Fallback if prompt is not found
        # logging.info(f"decoded_preds: {decoded_preds}")
        logging.info(f"decoded_preds: {cleaned_preds}")
        
        # Prepare and decode labels
        labels = batch["labels"]
        labels[labels == -100] = tokenizer.pad_token_id # replace -100 with pad_token_id for decoding
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        logging.info(f"decoded_labels: {decoded_labels}")

        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)
        
        # Optional: clean up memory
        del batch, generated_ids, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Now compute metrics on all the decoded strings
    print("Computing final metrics...")
    cer_metric = load_metric('./scripts/cer.py')
    wer_metric = load_metric('./scripts/wer.py')
    
    cer = cer_metric.compute(predictions=all_preds, references=all_labels)
    wer = wer_metric.compute(predictions=all_preds, references=all_labels)
    
    results = {"cer": cer, "wer": wer}
    print(f"Evaluation results: {results}")

    return results


    # # Create trainer for evaluation
    # trainer = Trainer(
    #     model=model,
    #     args=TrainingArguments(
    #         output_dir="./eval_results",
    #         per_device_eval_batch_size=1,
    #         evaluation_strategy="steps",
    #         deepspeed=deepspeed_config_path if config.get('use_deepspeed', False) else None,
    #         fp16=config.get('fp16', True),
    #     ),
    #     eval_dataset=processed_val,
    #     compute_metrics=compute_metrics_trainer,
    #     data_collator=data_collator,
    # )

    # # Evaluate
    # print("Starting evaluating...")
    # results = trainer.evaluate()
    # print(f"Evaluation results: {results}")

    # return results


def analyze_steering_patterns(model_path: str):
    """Analyze steering patterns of a trained model."""
    model = SteerMoEEfficientLayerWiseModel.from_pretrained(model_path)

    # Get steering analysis
    if hasattr(model, 'whisper_encoder') and hasattr(model.whisper_encoder, 'get_steering_analysis'):
        # Create dummy input for analysis
        dummy_audio = torch.randn(1, 16000 * 10)  # 10 seconds of audio
        with torch.no_grad():
            _, gating_scores = model(dummy_audio, return_gating=True)
            analysis = model.get_steering_analysis(gating_scores)

        print("Steering Analysis:")
        print(f"Layer scales: {analysis['layer_scale_values']}")
        print(f"Expert diversity: {analysis['expert_diversity']}")
        print(f"Steering strength: {analysis['steering_strength']}")

        return analysis

    return None


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
        train_layer_wise_steermoe(
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
