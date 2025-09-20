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
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(process)d - %(levelname)s - %(filename)s>%(funcName)s>%(lineno)d - %(message)s')

# Import our models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from steer_moe.efficient_layer_wise_conformer import EfficientLayerWiseSteeringConformerEncoder
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
        self.conformer_encoder = EfficientLayerWiseSteeringConformerEncoder(
            conformer_encoder_path, num_experts=num_experts,steering_scale=steering_scale
        )
        
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
            encoder_output_dim = self.conformer_encoder.feature_dim
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
            h_audio,input_lengths,_, gating_scores = self.conformer_encoder._forward_with_steering(audio_input, input_lengths=input_lengths ,pad = True, return_full_output=True)
        else:
            # Apply layer-wise steering to the preprocessed features  
            h_audio,input_lengths,_,_ = self.conformer_encoder._forward_with_steering(audio_input, input_lengths=input_lengths ,pad = True, return_full_output=True)
            
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
    
        h_audio,input_lengths,_,_ = self.conformer_encoder._forward_with_steering(input_features, input_lengths=input_lengths ,pad = True, return_full_output=True)


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

@dataclass
class DataCollatorSpeechSeqSeqWithPaddingForConformer:
    """
    Enhanced data collator for SteerMoE model that handles preprocessed audio features and labels.
    Works with preprocessed datasets that have 'input_features' and 'labels' columns.
    """
    feature_extractor: Any
    tokenizer: Any
    # textual_prompt: Optional[str] = None
    # max_length: int = 1024
    max_length: int = 448
    prompt_column: str = "text_prompt"
    audio_column: str = "input_features"  # Preprocessed audio features
    text_column: str = "labels"  # Preprocessed tokenized labels
    return_attention_mask: bool = False
    
    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of preprocessed features for SteerMoE training.
        
        Args:
            features: List of dictionaries with input_features, labels, etc.
            
        Returns:
            Batch dictionary with padded tensors
        """
        # We assume all samples have the same sample rate,
        # so we use the rate from the first sample (sample[0]) as the reference.
        batch_size = len(features)
        if "sample_rate" in features[0]:
            Sample_Rate = features[0]["sample_rate"]
        else:
            raise ValueError("Features must contain 'sample_rate' for processing, please check your dataset")
            
        
        # Handle preprocessed audio features
        audio_features = []
        for feature in features:
            if self.audio_column in feature:
                audio_feat = feature[self.audio_column]
                audio_features.append(np.array(audio_feat, dtype=np.int16))
            else:
                raise KeyError(f"Expected '{self.audio_column}' in features")

        
        # Feature Extraction
        batch_audio,lengths,duration = self.feature_extractor(audio_features,Sample_Rate)
        

        # logging.info(f"batch_audio: {batch_audio.shape} {batch_audio.dtype} {batch_audio}")
        
        # Handle preprocessed labels (already tokenized)
        labels = []
        decoder_input_ids = []
        prompt_token_ids=[]

        # Get the EOS token ID once
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token ID.")
        eos_tensor = torch.tensor([eos_token_id], dtype=torch.long)
        
        for feature in features:
            if self.text_column in feature:
                label_ids = feature[self.text_column]
                if isinstance(label_ids, torch.Tensor):
                    label_ids = label_ids.squeeze()
                else:
                    label_ids = torch.tensor(label_ids, dtype=torch.long).squeeze()

                # logging.debug(f"label_ids: {label_ids.shape}, {label_ids.dtype}, {label_ids}")
                
                # Remove padding tokens and special tokens for processing
                if isinstance(label_ids, torch.Tensor):
                    # Remove padding tokens (-100 and pad_token_id)
                    valid_mask = (label_ids != -100) & (label_ids != self.tokenizer.pad_token_id)
                    if valid_mask.any():
                        clean_labels = label_ids[valid_mask]
                    else:
                        clean_labels = torch.tensor([], dtype=torch.long)
                else:
                    clean_labels = torch.tensor(label_ids, dtype=torch.long)

                clean_labels_with_eos = torch.cat([clean_labels, eos_tensor])

                # logging.debug(f"clean_labels: {clean_labels.shape}, {clean_labels.dtype}, {clean_labels}")
                textual_prompt=feature[self.prompt_column]
                # Create decoder input with optional textual prompt
                if textual_prompt is not None and len(clean_labels) > 0:
                    # Tokenize the textual prompt
                    prompt_tokens = self.tokenizer.encode(
                        textual_prompt, 
                        add_special_tokens=False,
                        return_tensors="pt"
                    ).squeeze(0)

                    # logging.debug(f"prompt_tokens: {prompt_tokens.shape}, {prompt_tokens.dtype}, {prompt_tokens}")
                    
                    # Combine prompt + clean labels for decoder input

                    decoder_input = torch.cat([prompt_tokens, clean_labels_with_eos])
                    empty_prompt = torch.full_like(prompt_tokens, fill_value=-100)
                    label = torch.cat([empty_prompt, clean_labels_with_eos])

                    # decoder_input = torch.cat([prompt_tokens, clean_labels])
                    # Labels should be the original clean labels only (prompt gets masked in model)
                    # empty_prompt=torch.full_like(prompt_tokens, fill_value=-100)
                    # label=torch.cat([empty_prompt, clean_labels])

                    # label = clean_labels.clone()
                else:
                    # No prompt, use clean labels directly
                    
                    decoder_input = clean_labels_with_eos.clone()
                    label = clean_labels_with_eos.clone()

                if decoder_input.numel() > 0 and decoder_input.size(0) > self.max_length:
                    decoder_input = decoder_input[: self.max_length]
                if label.numel() > 0 and label.size(0) > self.max_length:
                    label = label[: self.max_length]
                
                # logging.debug(f"decoder_input: {decoder_input.shape}, {decoder_input.dtype}, {decoder_input}")
                # logging.debug(f"label: {label.shape}, {label.dtype}, {label}")
                prompt_token_ids.append(prompt_tokens)
                decoder_input_ids.append(decoder_input)
                labels.append(label)
            else:
                # No labels provided - create empty tensors
                prompt_token_ids.append(torch.tensor([], dtype=torch.long))
                decoder_input_ids.append(torch.tensor([], dtype=torch.long))
                labels.append(torch.tensor([], dtype=torch.long))

        # logging.debug(f"label: {[len(label) for label in labels]}")
        # logging.debug(f"decoder_input_ids: {[len(decoder_input_id) for decoder_input_id in decoder_input_ids]}")
        # Pad lables 
        label_features=[{"input_ids": input_ids} for input_ids in labels]
        if label_features:
            # labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
            labels_batch = self.tokenizer.pad(
                label_features, 
                return_tensors="pt",
                # padding_value=-100,
                max_length=self.max_length,
                padding="max_length",
                # truncation=True,
            )
            # logging.debug(f"labels_batch: {labels_batch}")
            # logging.debug(f"labels_batch['input_ids']: {labels_batch['input_ids'].shape}, {labels_batch['input_ids'].dtype}, {labels_batch['input_ids']}")
            # logging.debug(f"labels_batch['attention_mask']: {labels_batch['attention_mask'].shape}, {labels_batch['attention_mask'].dtype}, {labels_batch['attention_mask']}")

            batch_labels = labels_batch["input_ids"]
            batch_labels = batch_labels.masked_fill(labels_batch.attention_mask.ne(1), -100)
            # if (batch_labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            #     batch_labels = batch_labels[:, 1:]
        else:
            batch_labels = torch.empty(batch_size, 0, dtype=torch.long)

        # logging.debug(f"batch_labels: {batch_labels.shape}, {batch_labels.dtype}, {batch_labels}")

        # Pad prompt_tokens+labels
        input_features=[{"input_ids": input_ids} for input_ids in decoder_input_ids]
        if label_features:
            # input_ids_batch = self.tokenizer.pad(input_features, return_tensors="pt")
            input_ids_batch = self.tokenizer.pad(
                input_features, 
                return_tensors="pt",
                # padding_value=-100,
                max_length=self.max_length,
                padding="max_length",
                # truncation=True,
            )
            # logging.debug(f"input_ids_batch: {input_ids_batch}")
            # logging.debug(f"input_ids_batch['input_ids']: {input_ids_batch['input_ids'].shape}, {input_ids_batch['input_ids'].dtype}, {input_ids_batch['input_ids']}")
            # logging.debug(f"input_ids_batch['attention_mask']: {input_ids_batch['attention_mask'].shape}, {input_ids_batch['attention_mask'].dtype}, {input_ids_batch['attention_mask']}")

            batch_input_ids = input_ids_batch["input_ids"]
            # do not turn pads into -100, keep pad_token_id; the llm embed cannot understand -100
            # batch_input_ids = batch_input_ids.masked_fill(input_ids_batch.attention_mask.ne(1), -100)
        else:
            batch_input_ids = torch.empty(batch_size, 0, dtype=torch.long)

        prompt_features=[{"input_ids": input_ids} for input_ids in prompt_token_ids]
        if label_features:
            # input_ids_batch = self.tokenizer.pad(input_features, return_tensors="pt")
            prompt_ids_batch = self.tokenizer.pad(
                prompt_features,
                return_tensors="pt",
                # padding_value=-100,
                # max_length=self.max_length,
                padding="longest",
                # truncation=True,
            )
            # logging.debug(f"input_ids_batch: {input_ids_batch}")
            # logging.debug(f"input_ids_batch['input_ids']: {input_ids_batch['input_ids'].shape}, {input_ids_batch['input_ids'].dtype}, {input_ids_batch['input_ids']}")
            # logging.debug(f"input_ids_batch['attention_mask']: {input_ids_batch['attention_mask'].shape}, {input_ids_batch['attention_mask'].dtype}, {input_ids_batch['attention_mask']}")

            batch_prompt_ids = prompt_ids_batch["input_ids"]
            # do not turn pads into -100, keep pad_token_id; the llm embed cannot understand -100
            # batch_input_ids = batch_input_ids.masked_fill(input_ids_batch.attention_mask.ne(1), -100)
        else:
            batch_prompt_ids = torch.empty(batch_size, 0, dtype=torch.long)
        # Create final batch - use different key name for audio to match model expectations
        batch = {
            "input_features": batch_audio,  # Model expects audio_waveform parameter
            "decoder_input_ids": batch_input_ids,
            "labels": batch_labels,
            "input_lengths":lengths,
            "prompt_input_ids": batch_prompt_ids
        }
        
        # if self.return_attention_mask:
        #     batch["attention_mask"] = batch_attention_mask
            
        return batch

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
    # textual_prompt = config.get('textual_prompt', "请转写以下音频内容为文字：") 
    max_length = config.get('max_text_length', 512)


    # Create data collator with fixed max length for consistent evaluation
    data_collator = DataCollatorSpeechSeqSeqWithPaddingForConformer(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        # textual_prompt=textual_prompt,
        max_length=max_length,  # Fixed max length
        prompt_column= "text_prompt",
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

    # textual_prompt = config.get('textual_prompt', "请转写以下音频内容为文字：")  # Default Chinese prompt
    max_length = config.get('max_text_length', 512)
    
    # Create data collator with fixed max length for consistent evaluation
    data_collator = DataCollatorSpeechSeqSeqWithPaddingForConformer(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        # textual_prompt=textual_prompt,
        max_length=max_length,  # Fixed max length
        audio_column="input_features",  # Use preprocessed audio features
        text_column="labels",  # Use preprocessed labels
        prompt_column="text_prompt",
    )

    # Create a DataLoader for the evaluation set
    # NOTE: You should set a reasonable batch size here. 1 is safe, but you can try larger.
    eval_batch_size = config['training'].get('per_device_eval_batch_size', 1) 
    eval_dataloader = DataLoader(
        processed_val,
        batch_size=eval_batch_size,
        collate_fn=data_collator
    )

    all_preds_decoded = []
    all_labels_decoded = []
    all_preds_token_ids = []
    all_labels_token_ids = []
    print("Starting manual evaluation loop...")
    # textual_prompt = config.get('textual_prompt', "请转写以下音频内容为文字：")
    # prompt_ids = tokenizer(textual_prompt, return_tensors="pt").input_ids.to(device)

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        # Fallback for some tokenizers
        eos_token_id = tokenizer.pad_token_id
        logging.warning(f"EOS token not found, using PAD token as EOS: {eos_token_id}")

    for step, batch in enumerate(tqdm.tqdm(eval_dataloader)):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_prompt_ids = batch["prompt_input_ids"].to(device)
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
        decoded_prompts_for_cleaning = tokenizer.batch_decode(batch_prompt_ids)
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        cleaned_preds = []
        for i, pred in enumerate(decoded_preds):
            actual_prompt=decoded_prompts_for_cleaning[i]
            # Find the prompt in the prediction and take the text after it
            if actual_prompt in pred:
                cleaned_preds.append(pred.split(actual_prompt, 1)[1])
            else:
                cleaned_preds.append(pred) # Fallback if prompt is not found
        logging.info(f"decoded_preds: {cleaned_preds}")
        
        # Prepare and decode labels
        labels = batch["labels"]
        labels[labels == -100] = tokenizer.pad_token_id # replace -100 with pad_token_id for decoding
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
                pred_tokens_without_prompt = torch.tensor([], dtype=torch.long, device=device) # Empty if only prompt was returned

            # Clean padding and EOS from predictions
            cleaned_pred_tokens = []
            for token_id in pred_tokens_without_prompt:
                if token_id == tokenizer.pad_token_id or token_id == eos_token_id:
                    break
                cleaned_pred_tokens.append(token_id.item())

            # Clean padding from labels
            cleaned_label_tokens = []
            for token_id in label_tokens:
                if token_id == tokenizer.pad_token_id or token_id == -100: # Ensure -100 also handled
                    break
                cleaned_label_tokens.append(token_id.item())

            all_preds_token_ids.append(cleaned_pred_tokens)
            all_labels_token_ids.append(cleaned_label_tokens)

        
        # Optional: clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Now compute metrics on all the decoded strings
    logging.info("Computing final metrics...")
    cer_metric = load_metric('./scripts/cer.py')
    wer_metric = load_metric('./scripts/wer.py')
    accuracy_metric = load_metric('accuracy')

    cer = cer_metric.compute(predictions=all_preds_decoded, references=all_labels_decoded)
    wer = wer_metric.compute(predictions=all_preds_decoded, references=all_labels_decoded)

    # --- Token-level Accuracy (from cleaned_pred and decoded_labels tokens, excluding prompt) ---
    # We will compute token-level accuracy on `all_preds_token_ids` and `all_labels_token_ids`
    # which have already had prompts removed and padding/EOS cleaned.
    # To use `accuracy_metric.compute`, we need to ensure lists are of comparable length,
    # typically by padding or truncating to a common maximum.

    token_accuracy = 0.0
    
    # Check if there are any tokens to compare
    if all_preds_token_ids and all_labels_token_ids:
        # Determine the maximum length among all *cleaned* token sequences
        max_len_for_token_acc = 0
        if all_preds_token_ids:
            max_len_for_token_acc = max(max_len_for_token_acc, max((len(l) for l in all_preds_token_ids), default=0))
        if all_labels_token_ids:
            max_len_for_token_acc = max(max_len_for_token_acc, max((len(l) for l in all_labels_token_ids), default=0))

        # Flattened lists to hold padded tokens for the accuracy metric
        flat_preds_for_token_acc = []
        flat_labels_for_token_acc = []

        for i in range(len(all_preds_token_ids)):
            pred_tokens = all_preds_token_ids[i]
            label_tokens = all_labels_token_ids[i]

            # Pad or truncate to max_len_for_token_acc for element-wise comparison
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
        token_accuracy = float('nan') # If initial cleaned lists are empty
        logging.warning("No cleaned token lists available for token-level accuracy calculation.")


    # --- Raw String-based Accuracy (Substring Matching) ---
    # For a prediction-label pair, if the prediction is contained in the label string,
    # or the label is contained in the prediction string, we record it as correct.
    # Calculate accuracy by correct number / total number of pairs.

    correct_matches_count = 0
    total_pairs = len(all_preds_decoded) # Use the count of decoded string pairs

    if total_pairs > 0:
        for i in range(total_pairs):
            pred_str = all_preds_decoded[i].strip()
            label_str = all_labels_decoded[i].strip()

            # Check if prediction is contained in label, OR label is contained in prediction.
            # Handles cases where one or both might be empty strings.
            if (pred_str and label_str and (pred_str in label_str or label_str in pred_str)) or \
               (not pred_str and not label_str): # Both empty strings count as a match
                correct_matches_count += 1
        
        raw_accuracy = correct_matches_count / total_pairs
    else:
        raw_accuracy = float('nan') # If no decoded strings
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
