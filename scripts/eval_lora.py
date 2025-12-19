"""
Evaluation script for LoRA-based ablation study model.

This script evaluates a trained LoRA model and computes metrics
for comparison with SteerMoE.

 * author Ruitao Feng
 * created on 16-07-2025
 * github: https://github.com/forfrt
"""

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor
from datasets import load_from_disk
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
from steer_moe.utils import DataCollatorSpeechSeqSeqWithPaddingPrompt
from steer_moe.utils import load_parquet_datasets


def load_config(path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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


def evaluate_lora_model(model_path: str, eval_dataset_name: str, config_path: str):
    """Evaluate a trained LoRA model."""
    config = load_config(config_path)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

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

    print("Loading LoRA model...")
    model = load_custom(
        model_path, lora_whisper, llm_decoder,
        max_prompt_tokens=config.get('max_prompt_tokens', 2048),
        use_adapter=config.get('use_adapter', True),
    )

    model.to(device)
    model.half()
    model.eval()

    # Load evaluation dataset
    print("Loading dataset...")
    parquet_dirs = config.get('parquet_dirs', [])
    batch_size = config['training'].get('per_device_eval_batch_size', 1)

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

    dataset = load_parquet_datasets(expanded_dirs)
    print(f"Dataset: {type(dataset)} {dataset}")

    # Use train split for evaluation (assuming it's actually test data)
    processed_val = dataset['train']

    max_length = config.get('max_text_length', 512)
    
    # Create data collator
    data_collator = DataCollatorSpeechSeqSeqWithPaddingPrompt(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_length=max_length,
        audio_column="input_features",
        text_column="labels",
        prompt_column="text_prompt",
    )

    # Create a DataLoader for the evaluation set
    eval_dataloader = DataLoader(
        processed_val,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    all_preds_decoded = []
    all_labels_decoded = []
    all_preds_token_ids = []
    all_labels_token_ids = []

    print("Starting manual evaluation loop...")

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id
        logging.warning(f"EOS token not found, using PAD token as EOS: {eos_token_id}")

    for step, batch in enumerate(tqdm.tqdm(eval_dataloader)):
        batch_prompt_ids = batch["prompt_input_ids"].to(device)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=batch["input_features"],
                decoder_input_ids=batch_prompt_ids,
                max_new_tokens=512,
                eos_token_id=eos_token_id,
            )

        decoded_prompts_for_cleaning = tokenizer.batch_decode(batch_prompt_ids)
        logging.info(f"decoded_prompts_for_cleaning: {decoded_prompts_for_cleaning}")

        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        logging.info(f"decoded_preds: {decoded_preds}")

        cleaned_preds = []
        for i, pred in enumerate(decoded_preds):
            actual_prompt = decoded_prompts_for_cleaning[i]
            if actual_prompt in pred:
                cleaned_preds.append(pred.split(actual_prompt, 1)[1])
            else:
                cleaned_preds.append(pred)
        logging.info(f"cleaned_preds: {cleaned_preds}")
        
        # Prepare and decode labels
        labels = batch["labels"]
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        logging.info(f"decoded_labels: {decoded_labels}")

        all_preds_decoded.extend(cleaned_preds)
        all_labels_decoded.extend(decoded_labels)

        # Prepare for Accuracy Calculation
        current_labels = batch["labels"].clone()
        current_labels[current_labels == -100] = tokenizer.pad_token_id

        for i in range(len(generated_ids)):
            pred_tokens = generated_ids[i]
            label_tokens = current_labels[i]

            prompt_len = (batch["prompt_input_ids"][i] != tokenizer.pad_token_id).sum().item()
            
            if len(pred_tokens) > prompt_len:
                pred_tokens_without_prompt = pred_tokens[prompt_len:]
            else:
                pred_tokens_without_prompt = torch.tensor([], dtype=torch.long, device=device)

            # Clean padding and EOS from predictions
            cleaned_pred_tokens = []
            for token_id in pred_tokens_without_prompt:
                if token_id == tokenizer.pad_token_id or token_id == eos_token_id:
                    break
                cleaned_pred_tokens.append(token_id.item())

            # Clean padding from labels
            cleaned_label_tokens = []
            for token_id in label_tokens:
                if token_id == tokenizer.pad_token_id or token_id == -100:
                    break
                cleaned_label_tokens.append(token_id.item())

            all_preds_token_ids.append(cleaned_pred_tokens)
            all_labels_token_ids.append(cleaned_label_tokens)
        
        # Optional: clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute metrics
    logging.info("Computing final metrics...")
    from datasets import load_metric
    
    cer_metric = load_metric('./scripts/cer.py')
    wer_metric = load_metric('./scripts/wer.py')
    accuracy_metric = load_metric('accuracy')

    cer = cer_metric.compute(predictions=all_preds_decoded, references=all_labels_decoded)
    wer = wer_metric.compute(predictions=all_preds_decoded, references=all_labels_decoded)

    # Token-level Accuracy
    token_accuracy = 0.0
    
    if all_preds_token_ids and all_labels_token_ids:
        max_len_for_token_acc = 0
        if all_preds_token_ids:
            max_len_for_token_acc = max(max_len_for_token_acc, max((len(l) for l in all_preds_token_ids), default=0))
        if all_labels_token_ids:
            max_len_for_token_acc = max(max_len_for_token_acc, max((len(l) for l in all_labels_token_ids), default=0))

        flat_preds_for_token_acc = []
        flat_labels_for_token_acc = []

        for i in range(len(all_preds_token_ids)):
            pred_tokens = all_preds_token_ids[i]
            label_tokens = all_labels_token_ids[i]

            current_padded_pred = (pred_tokens + [tokenizer.pad_token_id] * max_len_for_token_acc)[:max_len_for_token_acc]
            current_padded_label = (label_tokens + [tokenizer.pad_token_id] * max_len_for_token_acc)[:max_len_for_token_acc]
            
            flat_preds_for_token_acc.extend(current_padded_pred)
            flat_labels_for_token_acc.extend(current_padded_label)

        if flat_preds_for_token_acc and flat_labels_for_token_acc:
            accuracy_results = accuracy_metric.compute(
                predictions=flat_preds_for_token_acc,
                references=flat_labels_for_token_acc
            )
            token_accuracy = accuracy_results.get('accuracy', 0.0)
        else:
            token_accuracy = float('nan')
            logging.warning("No valid tokens after padding/truncating for token-level accuracy calculation.")
    else:
        token_accuracy = float('nan')
        logging.warning("No cleaned token lists available for token-level accuracy calculation.")

    # Raw String-based Accuracy
    correct_matches_count = 0
    total_pairs = len(all_preds_decoded)

    if total_pairs > 0:
        for i in range(total_pairs):
            pred_str = all_preds_decoded[i].strip()
            label_str = all_labels_decoded[i].strip()

            if (pred_str and label_str and (pred_str in label_str or label_str in pred_str)) or \
               (not pred_str and not label_str):
                correct_matches_count += 1
        
        raw_accuracy = correct_matches_count / total_pairs
    else:
        raw_accuracy = float('nan')
        logging.warning("No decoded strings for raw string-based accuracy calculation.")

    results = {
        "cer": cer, 
        "wer": wer, 
        "token_accuracy": token_accuracy, 
        "raw_accuracy": raw_accuracy      
    }
    logging.info(f"Evaluation results: {results}")
    
    del batch, generated_ids, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate LoRA model')
    parser.add_argument('--config', type=str, default='configs/lora_whisper_qwen7b_libri_test.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    evaluate_lora_model(args.model_path, None, args.config)
