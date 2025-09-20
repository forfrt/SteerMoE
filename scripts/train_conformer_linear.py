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


from steer_moe.efficient_layer_wise_conformer import LoadConformerEncoder
from steer_moe.utils import DataCollatorSpeechSeqSeqWithPaddingForConformer
from steer_moe.conformer_module.asr_feat import ASRFeatExtractor


class SteerMoEEfficientLayerWiseModelForConformer(nn.Module):
    """
    Efficient layer-wise steering model using a single router.
    This approach uses one router to assign weights to steering vectors for all layers,
    making it much more parameter-efficient than the multiple router approach.
    """
    def __init__(self, conformer_encoder_path, llm_decoder, num_experts=8,steering_scale = 0.1, 
                 prompt_proj=None, max_prompt_tokens=None, use_adapter=True):
        super().__init__()
        
        # Create efficient layer-wise steering conformer encoder
        self.conformer_encoder = LoadConformerEncoder(conformer_encoder_path)
        
        self.llm_decoder = llm_decoder          # frozen
        self.use_adapter = use_adapter
        self.max_prompt_tokens = max_prompt_tokens

        # Freeze decoder
        for p in self.llm_decoder.parameters():
            p.requires_grad = False
        # self.llm_decoder.eval()
        
        # Optional projection layer if encoder and decoder dimensions don't match
        if prompt_proj is None and use_adapter:
            # Get dimensions from encoder output and decoder input
            encoder_output_dim = self.conformer_encoder.config['d_model']   
            # Try to get decoder input dimension
            if hasattr(llm_decoder, 'config'):
                decoder_input_dim = getattr(llm_decoder.config, 'n_embd', 
                                          getattr(llm_decoder.config, 'hidden_size', encoder_output_dim))
            else:
                decoder_input_dim = encoder_output_dim
            
            if encoder_output_dim != decoder_input_dim:
                self.prompt_proj = nn.Linear(encoder_output_dim, decoder_input_dim)
            else:
                self.prompt_proj = None
        else:
            self.prompt_proj = prompt_proj
            



    def forward(self, audio_waveform=None, input_features=None,input_lengths=None, decoder_input_ids=None, labels=None, 
                prompt_tokens_only=False,prompt_input_ids =None, return_full_output=False):
        """
        Forward pass for efficient layer-wise steering model.
        
        Args:
            audio_waveform: Raw audio waveform input OR preprocessed audio features
            input_features: Alternative preprocessed audio input (for compatibility)
            decoder_input_ids: Text token IDs 
            labels: Target labels for training 
            prompt_tokens_only: If True, only return prompt embeddings without text
            return_gating: If True, return gating scores for analysis
        
        Returns:
            Model output with loss if labels provided, or logits/embeddings
        """
        # Handle both input parameter names for compatibility
        audio_input = audio_waveform if audio_waveform is not None else input_features
        if audio_input is None:
            raise ValueError("Either audio_waveform or input_features must be provided")
        
        # 1. Process audio input - the input from data collator is already preprocessed Whisper features
        # The preprocessing pipeline already called tokenize_waveform, so we have processed features
        # We should apply steering to these features, not call tokenize_waveform again
        if return_full_output:
            # Apply layer-wise steering to the preprocessed features
            h_audio,input_lengths,src_mask = self.conformer_encoder.speech_encoder.forward(audio_input, input_lengths=input_lengths ,pad = True)
        else:
            # Apply layer-wise steering to the preprocessed features  
            h_audio,input_lengths,_ = self.conformer_encoder.speech_encoder.forward(audio_input, input_lengths=input_lengths ,pad = True)
            
            gating_scores = None
        # TODO Logging h_audio: (batch, audio_seq_len, feature_dim)

        # 2. Project to decoder dimension if needed
        if self.use_adapter and self.prompt_proj is not None:
            audio_prompts = self.prompt_proj(h_audio)
        else:
            audio_prompts = h_audio
        # audio_prompts: (batch, audio_seq_len, decoder_dim)

        # 3. Limit prompt length if specified
        if self.max_prompt_tokens is not None:
            audio_prompts = audio_prompts[:, :self.max_prompt_tokens, :]

        # 4. Handle pure audio generation case
        if decoder_input_ids is None:
            if prompt_tokens_only:
                if return_full_output:
                    return audio_prompts, gating_scores
                else:
                    return audio_prompts
            else:
                # For generation without text input
                if return_full_output:
                    return audio_prompts, gating_scores
                else:
                    return audio_prompts

        # 5. Get text embeddings from decoder
        if hasattr(self.llm_decoder, 'model') and hasattr(self.llm_decoder.model, 'embed_tokens'):
            # LLaMA-like decoder (Qwen, LLaMA, etc.)
            text_embeds = self.llm_decoder.model.embed_tokens(decoder_input_ids)
        elif hasattr(self.llm_decoder, 'transformer') and hasattr(self.llm_decoder.transformer, 'wte'):
            # GPT-2 like decoder
            text_embeds = self.llm_decoder.transformer.wte(decoder_input_ids)
        elif hasattr(self.llm_decoder, 'get_input_embeddings'):
            # Generic approach
            text_embeds = self.llm_decoder.get_input_embeddings()(decoder_input_ids)
        else:
            raise ValueError("Could not find embedding method for decoder. Please adapt for your LLM.")

        

            
        # 6. Concatenate audio prompts with text embeddings
        # Format: [audio_prompts, text_embeds]
        # inputs_embeds = torch.cat([text_embeds,audio_prompts ], dim=1)
        inputs_embeds = torch.cat([audio_prompts, text_embeds], dim=1)   
        # inputs_embeds: (batch, audio_seq_len + text_seq_len, decoder_dim)

        # 7. Handle labels for training
        if labels is not None:
            audio_prompt_len = audio_prompts.size(1)
            
            # Create attention mask for the combined sequence
            # We want to attend to both audio prompts and text, but only compute loss on text
            batch_size, total_seq_len = inputs_embeds.shape[:2]
            attention_mask = torch.ones(batch_size, total_seq_len, device=inputs_embeds.device, dtype=torch.long)
            
            # for i in range(batch_size):
            #     length_i = input_lengths[i]
            #     attention_mask[i,length_i:audio_prompt_len] = 0
            
            # Mask audio prompt tokens with -100 (ignored in loss computation)
            # The labels should correspond only to the text portion
            if labels.size(1) != text_embeds.size(1):
                # If labels length doesn't match text length, truncate or pad
                if labels.size(1) > text_embeds.size(1):
                    labels = labels[:, :text_embeds.size(1)]
                else:
                    # Pad with -100
                    pad_length = text_embeds.size(1) - labels.size(1)
                    labels = torch.cat([
                        labels,
                        labels.new_full((labels.size(0), pad_length), -100)
                    ], dim=1)
            
            # Prepend -100 tokens for audio prompts (these will be ignored in loss)
            full_labels = torch.cat([
                labels.new_full((labels.size(0), audio_prompt_len), -100),
                labels
            ], dim=1)
            
            # Pass to decoder with masked labels
            output = self.llm_decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=full_labels
            )
        else:
            # No labels - inference mode
            output = self.llm_decoder(
                inputs_embeds=inputs_embeds
            )

        # 8. Return output with gating scores if requested
        if return_full_output:
            if hasattr(output, 'loss'):
                # Training mode - return loss and gating scores
                return {
                    'loss': output.loss,
                    'logits': output.logits,
                    'gating_scores': gating_scores
                }
            else:
                # Inference mode
                return output, gating_scores
        else:
            return output

    # def get_steering_analysis(self, gating_scores):
    #     """
    #     Analyze steering patterns across layers.
        
    #     Args:
    #         gating_scores: Gating scores from forward pass
            
    #     Returns:
    #         Dictionary with steering analysis
    #     """
    #     return self.conformer_encoder.get_steering_analysis(gating_scores)

    def get_device(self):
        return next(self.parameters()).device
    
    def generate(self,input_features=None,input_lengths=None, decoder_input_ids=None, **kwargs):
        """
        Generates token sequences autoregressively.
        This method prepares the audio prompt embeddings and then calls the
        underlying LLM decoder's generate method.
        
        Args:
            input_features: The preprocessed audio features from the FeatureExtractor.
            **kwargs: Additional arguments to be passed to the llm_decoder.generate() method,
                      such as max_new_tokens, num_beams, do_sample, etc.
        """
        # Ensure the model is in evaluation mode for generation
        self.eval()
    
        h_audio,input_lengths,_ = self.conformer_encoder.speech_encoder.forward(input_features, input_lengths=input_lengths ,pad = True)


        # 2. Project to decoder dimension if needed
        if self.use_adapter and self.prompt_proj is not None:
            audio_prompts = self.prompt_proj(h_audio)
        else:
            audio_prompts = h_audio


        if self.max_prompt_tokens is not None:
            audio_prompts = audio_prompts[:, :self.max_prompt_tokens, :]


        if hasattr(self.llm_decoder, 'model') and hasattr(self.llm_decoder.model, 'embed_tokens'):
            # LLaMA-like decoder (Qwen, LLaMA, etc.)
            text_embeds = self.llm_decoder.model.embed_tokens(decoder_input_ids)
        elif hasattr(self.llm_decoder, 'transformer') and hasattr(self.llm_decoder.transformer, 'wte'):
            # GPT-2 like decoder
            text_embeds = self.llm_decoder.transformer.wte(decoder_input_ids)
        elif hasattr(self.llm_decoder, 'get_input_embeddings'):
            # Generic approach
            text_embeds = self.llm_decoder.get_input_embeddings()(decoder_input_ids)
        else:
            raise ValueError("Could not find embedding method for decoder. Please adapt for your LLM.")

        

            

        inputs_embeds = torch.cat([audio_prompts, text_embeds], dim=1)   
        # inputs_embeds: (batch, audio_seq_len + text_seq_len, decoder_dim)


        batch_size, total_seq_len = inputs_embeds.shape[:2]
        attention_mask = torch.ones(batch_size, total_seq_len, device=inputs_embeds.device, dtype=torch.long)
            
        return self.llm_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs  # Pass through all other generation settings
        )

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

@staticmethod
def load_custom(path, conformer_encoder_path, llm_decoder, **kwargs):
    model = SteerMoEEfficientLayerWiseModelForConformer(
        conformer_encoder_path=conformer_encoder_path,
        llm_decoder=llm_decoder,
        **kwargs
    )
    sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    return model



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
        conformer_encoder_path=config['conformer_encoder']['model_path']+"/model.pth.tar",
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
        # prompt_column= "text_prompt",
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



def evaluate_layer_wise_model(model_path: str, eval_dataset_name: str, config_path: str, local_rank):
    """Evaluate a trained layer-wise SteerMoE model."""
    device = torch.device("cuda", local_rank)
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
    cmvn_path = os.path.join(config['conformer_encoder']['model_path'], "cmvn.ark") 
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




    print("Creating SteerMoE model...")
    model = load_custom(
        model_path, config['conformer_encoder']['model_path']+"/model.pth.tar", llm_decoder,
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
    # model.half()
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
                input_lengths=batch["input_lengths"],
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
        print(args.model_path)
        if args.model_path is None:
            raise ValueError("Model path required for evaluation")
        evaluate_layer_wise_model(args.model_path, args.eval_dataset, args.config, args.local_rank)
    elif args.mode == 'analyze':
        if args.model_path is None:
            raise ValueError("Model path required for analysis")
        analyze_steering_patterns(args.model_path)
