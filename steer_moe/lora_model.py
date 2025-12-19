"""
LoRA-based ablation study model for SteerMoE comparison.

This module implements a LoRA (Low-Rank Adaptation) baseline that:
- Freezes the pretrained LLM decoder
- Applies LoRA adapters to the speech encoder layers
- Provides similar interface to SteerMoE models for fair comparison

Why not use PEFT library?
--------------------------
While PEFT (Parameter-Efficient Fine-Tuning) from HuggingFace could be used,
we implement LoRA from scratch because:
1. Our Whisper encoder uses custom structure (speech_encoder.layers) not compatible
   with PEFT's automatic module detection
2. Custom implementation provides better control for ablation studies
3. Keeps codebase consistent with SteerMoE's custom implementation
4. Easier to debug and modify for research purposes
5. No external dependencies beyond PyTorch

For fair comparison, both SteerMoE and LoRA use similar custom implementation
patterns, making the ablation study more meaningful.

 * author Ruitao Feng
 * created on 16-07-2025
 * github: https://github.com/forfrt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(process)d - %(levelname)s - %(filename)s>%(funcName)s>%(lineno)d - %(message)s')

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from steer_moe.tokenizer.whisper_Lv3.whisper import WhisperEncoder


def _get_num_layers(encoder):
    """Recursively find the number of layers in the encoder."""
    if hasattr(encoder, 'speech_encoder'):
        return len(encoder.speech_encoder.layers)
    elif hasattr(encoder, '_modules') and 'speech_encoder' in encoder._modules:
        return len(encoder._modules['speech_encoder'].layers)
    elif hasattr(encoder, 'original_encoder'):
        return _get_num_layers(encoder.original_encoder)
    else:
        raise AttributeError("Cannot find layers in the provided encoder object.")


def _get_layers(encoder):
    """Recursively find the layers in the encoder."""
    if hasattr(encoder, 'speech_encoder'):
        return encoder.speech_encoder.layers
    elif hasattr(encoder, '_modules') and 'speech_encoder' in encoder._modules:
        return encoder._modules['speech_encoder'].layers
    elif hasattr(encoder, 'original_encoder'):
        return _get_layers(encoder.original_encoder)
    else:
        raise AttributeError("Cannot find layers in the provided encoder object.")


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) linear layer.
    
    Wraps a frozen linear layer with trainable low-rank adapters:
    output = W_original(x) + (B @ A) * scaling * x
    
    Where:
    - W_original: Frozen original weight matrix
    - A: Low-rank matrix (rank x in_features)
    - B: Low-rank matrix (out_features x rank)
    - scaling: Scaling factor (1/rank by default)
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        scaling: Optional[float] = None,
        alpha: float = 1.0
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Default scaling is 1/rank, but can be overridden
        if scaling is None:
            self.scaling = alpha / rank
        else:
            self.scaling = scaling
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA adapters
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Initialize A with small random values, B with zeros
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original + LoRA adaptation.
        
        Args:
            x: Input tensor (batch, seq_len, in_features)
            
        Returns:
            Output tensor (batch, seq_len, out_features)
        """
        # Original layer output (frozen)
        original_output = self.original_layer(x)
        
        # LoRA adaptation: x @ A^T @ B^T
        # x: (batch, seq_len, in_features)
        # A: (rank, in_features) -> (in_features, rank)
        # B: (out_features, rank)
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B.t())
        
        # Scale and add
        return original_output + self.scaling * lora_output


class LoRAWhisperEncoder(nn.Module):
    """
    Whisper encoder with LoRA adapters applied to attention and FFN layers.
    
    This is an ablation study baseline that applies LoRA to encoder layers
    instead of using steering vectors like SteerMoE.
    """
    
    def __init__(
        self,
        whisper_encoder_path: str,
        lora_rank: int = 8,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        target_modules: Optional[list] = None,
        pooling_kernel_size: Optional[int] = None,
        pooling_type: Optional[str] = None,
        pooling_position: int = 32
    ):
        """
        Initialize LoRA Whisper encoder.
        
        Args:
            whisper_encoder_path: Path to pretrained Whisper encoder
            lora_rank: Rank of LoRA adapters (default: 8)
            lora_alpha: Scaling factor for LoRA (default: 1.0)
            lora_dropout: Dropout rate for LoRA adapters (default: 0.0)
            target_modules: List of module types to apply LoRA to.
                          Default: ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
            pooling_kernel_size: Optional pooling kernel size for downsampling
            pooling_type: Pooling type ('max' or 'avg')
            pooling_position: Layer index to apply pooling (if enabled)
        """
        super().__init__()
        
        # Load and freeze original encoder
        self.original_encoder = WhisperEncoder(whisper_encoder_path)
        for param in self.original_encoder.parameters():
            param.requires_grad = False
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_type = pooling_type
        self.pooling_position = pooling_position
        
        # Default target modules: attention projections and FFN layers
        if target_modules is None:
            self.target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
        else:
            self.target_modules = target_modules
        
        # Get encoder structure
        self.num_layers = _get_num_layers(self.original_encoder)
        
        # Get feature dimension
        if hasattr(self.original_encoder, 'config'):
            self.feature_dim = self.original_encoder.config.d_model
        else:
            self.feature_dim = 1280  # Default for Whisper large
        
        # Apply LoRA to encoder layers
        self._apply_lora_to_layers()
        
        # Initialize pooling layer if needed
        self.pooling_layer = None
        if pooling_kernel_size is not None and pooling_type is not None:
            self.init_pooling_layer(pooling_type, pooling_kernel_size)
    
    def _apply_lora_to_layers(self):
        """Apply LoRA adapters to target modules in encoder layers."""
        # Get the speech encoder
        if hasattr(self.original_encoder, 'speech_encoder'):
            speech_encoder = self.original_encoder.speech_encoder
        else:
            speech_encoder = self.original_encoder
        
        encoder_layers = speech_encoder.layers
        
        # Store LoRA adapters for each layer
        self.lora_adapters = nn.ModuleDict()
        
        for layer_idx, layer in enumerate(encoder_layers):
            layer_key = f"layer_{layer_idx}"
            self.lora_adapters[layer_key] = nn.ModuleDict()
            
            # Apply LoRA to self-attention modules
            if hasattr(layer, 'self_attn'):
                self_attn = layer.self_attn
                
                for module_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                    if hasattr(self_attn, module_name):
                        original_module = getattr(self_attn, module_name)
                        if isinstance(original_module, nn.Linear):
                            lora_module = LoRALinear(
                                original_module,
                                rank=self.lora_rank,
                                alpha=self.lora_alpha
                            )
                            self.lora_adapters[layer_key][f"self_attn_{module_name}"] = lora_module
                            # Replace original module with LoRA wrapper
                            setattr(self_attn, module_name, lora_module)
            
            # Apply LoRA to feed-forward network modules
            if hasattr(layer, 'fc1'):
                original_fc1 = layer.fc1
                if isinstance(original_fc1, nn.Linear):
                    lora_fc1 = LoRALinear(
                        original_fc1,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha
                    )
                    self.lora_adapters[layer_key]["fc1"] = lora_fc1
                    layer.fc1 = lora_fc1
            
            if hasattr(layer, 'fc2'):
                original_fc2 = layer.fc2
                if isinstance(original_fc2, nn.Linear):
                    lora_fc2 = LoRALinear(
                        original_fc2,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha
                    )
                    self.lora_adapters[layer_key]["fc2"] = lora_fc2
                    layer.fc2 = lora_fc2
    
    def init_pooling_layer(self, pooling_type: str, pooling_kernel_size: int):
        """Initialize pooling layer for downsampling."""
        if pooling_kernel_size is not None:
            if pooling_type == "max":
                self.pooling_layer = nn.MaxPool1d(kernel_size=pooling_kernel_size)
            elif pooling_type == "avg":
                self.pooling_layer = nn.AvgPool1d(kernel_size=pooling_kernel_size)
            else:
                raise NotImplementedError(f"Pooling type {pooling_type} not implemented")
    
    def forward(self, mel_features: torch.Tensor, return_gating: bool = False) -> torch.Tensor:
        """
        Forward pass through LoRA-adapted encoder.
        
        Args:
            mel_features: Preprocessed mel-spectrogram features
                         Shape: (batch, 128, seq_len) or (batch, seq_len, feature_dim)
            return_gating: For compatibility with SteerMoE interface (ignored)
            
        Returns:
            Encoded features: (batch, seq_len, feature_dim)
        """
        # Get the speech encoder
        if hasattr(self.original_encoder, 'speech_encoder'):
            speech_encoder = self.original_encoder.speech_encoder
        else:
            speech_encoder = self.original_encoder
        
        # Device and dtype management
        model_device = next(self.lora_adapters.parameters()).device
        model_dtype = next(self.lora_adapters.parameters()).dtype
        
        speech_encoder = speech_encoder.to(model_device)
        x = mel_features.to(device=model_device, dtype=model_dtype)
        
        # Initial processing (conv layers, positional embedding)
        if hasattr(speech_encoder, 'conv1'):
            x = F.gelu(speech_encoder.conv1(x))
            x = F.gelu(speech_encoder.conv2(x))
            x = x.permute(0, 2, 1)
        else:
            logging.error("no conv1 layer found")
        
        if hasattr(speech_encoder, 'embed_positions'):
            positions = speech_encoder.embed_positions.weight[:x.size(1)].to(model_device)
            x = (x + positions) * speech_encoder.embed_scale
        else:
            logging.error("no embed_positions layer found")
        
        # Apply dropout if present
        if hasattr(speech_encoder, 'dropout'):
            x = F.dropout(x, p=speech_encoder.dropout, training=speech_encoder.training)
        
        # Process through LoRA-adapted layers
        encoder_layers = speech_encoder.layers
        for layer_idx, layer in enumerate(encoder_layers):
            layer = layer.to(model_device)
            
            # Forward through layer (LoRA adapters are already integrated)
            layer_output = layer(x)[0]
            x = layer_output
            
            # Optional pooling at specific layer
            if layer_idx + 1 == self.pooling_position and self.pooling_layer is not None:
                x = x.permute(0, 2, 1)  # (batch, feature_dim, seq)
                if x.shape[-1] % self.pooling_kernel_size != 0:
                    x = F.pad(x, (0, self.pooling_kernel_size - x.shape[-1] % self.pooling_kernel_size))
                x = self.pooling_layer(x).permute(0, 2, 1)  # (batch, seq', feature_dim)
        
        # Final layer norm
        if hasattr(speech_encoder, 'layer_norm'):
            x = speech_encoder.layer_norm(x)
        else:
            logging.error("no layer_norm found")
        
        return x


class LoRAModel(nn.Module):
    """
    Main LoRA model for ablation study comparison with SteerMoE.
    
    Architecture:
    Audio → LoRA-adapted Encoder → Projection → Frozen LLM Decoder → Text
    
    This model freezes the LLM decoder and applies LoRA to the encoder,
    providing a baseline for comparison with SteerMoE.
    """
    
    def __init__(
        self,
        whisper_encoder: LoRAWhisperEncoder,
        llm_decoder: nn.Module,
        prompt_proj: Optional[nn.Module] = None,
        max_prompt_tokens: Optional[int] = None,
        use_adapter: bool = True
    ):
        """
        Initialize LoRA model.
        
        Args:
            whisper_encoder: LoRA-adapted Whisper encoder
            llm_decoder: Frozen LLM decoder (Qwen, LLaMA, etc.)
            prompt_proj: Optional projection layer (auto-created if None and use_adapter=True)
            max_prompt_tokens: Maximum number of audio prompt tokens
            use_adapter: Whether to use projection layer
        """
        super().__init__()
        
        self.whisper_encoder = whisper_encoder
        self.llm_decoder = llm_decoder
        self.use_adapter = use_adapter
        self.max_prompt_tokens = max_prompt_tokens
        
        # Cache vocab size
        self.vocab_size = self.llm_decoder.get_input_embeddings().num_embeddings
        self.pad_token_id = getattr(
            getattr(self.llm_decoder, "config", None), "pad_token_id", None
        )
        
        # Freeze decoder
        for p in self.llm_decoder.parameters():
            p.requires_grad = False
        
        # Optional projection layer
        if prompt_proj is None and use_adapter:
            encoder_output_dim = self.whisper_encoder.feature_dim
            if hasattr(llm_decoder, 'config'):
                decoder_input_dim = getattr(
                    llm_decoder.config, 'n_embd',
                    getattr(llm_decoder.config, 'hidden_size', encoder_output_dim)
                )
            else:
                decoder_input_dim = encoder_output_dim
            
            if encoder_output_dim != decoder_input_dim:
                self.prompt_proj = nn.Linear(encoder_output_dim, decoder_input_dim)
            else:
                self.prompt_proj = None
        else:
            self.prompt_proj = prompt_proj
    
    def forward(
        self,
        audio_waveform: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        prompt_tokens_only: bool = False,
        return_gating: bool = False
    ):
        """
        Forward pass for LoRA model.
        
        Args:
            audio_waveform: Raw audio waveform (not used, kept for compatibility)
            input_features: Preprocessed audio features
            decoder_input_ids: Text token IDs
            labels: Target labels for training
            prompt_tokens_only: If True, only return prompt embeddings
            return_gating: For compatibility (ignored, LoRA doesn't have gating)
            
        Returns:
            Model output with loss if labels provided, or logits/embeddings
        """
        # Handle input
        audio_input = audio_waveform if audio_waveform is not None else input_features
        if audio_input is None:
            raise ValueError("Either audio_waveform or input_features must be provided")
        
        # Process audio through LoRA-adapted encoder
        h_audio = self.whisper_encoder.forward(audio_input, return_gating=False)
        # h_audio: (batch, audio_seq_len, feature_dim)
        
        # Project to decoder dimension if needed
        if self.use_adapter and self.prompt_proj is not None:
            audio_prompts = self.prompt_proj(h_audio)
        else:
            audio_prompts = h_audio
        # audio_prompts: (batch, audio_seq_len, decoder_dim)
        
        # Limit prompt length if specified
        if self.max_prompt_tokens is not None:
            audio_prompts = audio_prompts[:, :self.max_prompt_tokens, :]
        
        # Handle pure audio generation case
        if decoder_input_ids is None:
            if prompt_tokens_only:
                return audio_prompts
            else:
                return audio_prompts
        
        # Defensive checks
        assert (decoder_input_ids >= 0).all()
        assert (decoder_input_ids < self.vocab_size).all()
        
        # Get text embeddings from decoder
        if hasattr(self.llm_decoder, 'model') and hasattr(self.llm_decoder.model, 'embed_tokens'):
            text_embeds = self.llm_decoder.model.embed_tokens(decoder_input_ids)
        elif hasattr(self.llm_decoder, 'transformer') and hasattr(self.llm_decoder.transformer, 'wte'):
            text_embeds = self.llm_decoder.transformer.wte(decoder_input_ids)
        elif hasattr(self.llm_decoder, 'get_input_embeddings'):
            text_embeds = self.llm_decoder.get_input_embeddings()(decoder_input_ids)
        else:
            raise ValueError("Could not find embedding method for decoder. Please adapt for your LLM.")
        
        # Concatenate audio prompts with text embeddings
        inputs_embeds = torch.cat([audio_prompts, text_embeds], dim=1)
        # inputs_embeds: (batch, audio_seq_len + text_seq_len, decoder_dim)
        
        # Handle labels for training
        if labels is not None:
            audio_prompt_len = audio_prompts.size(1)
            
            # Create attention mask
            batch_size, total_seq_len = inputs_embeds.shape[:2]
            attention_mask = torch.ones(
                batch_size, total_seq_len,
                device=inputs_embeds.device, dtype=torch.long
            )
            
            # Align labels with text length
            if labels.size(1) != text_embeds.size(1):
                if labels.size(1) > text_embeds.size(1):
                    labels = labels[:, :text_embeds.size(1)]
                else:
                    pad_length = text_embeds.size(1) - labels.size(1)
                    labels = torch.cat([
                        labels,
                        labels.new_full((labels.size(0), pad_length), -100)
                    ], dim=1)
            
            # Prepend -100 tokens for audio prompts (ignored in loss)
            full_labels = torch.cat([
                labels.new_full((labels.size(0), audio_prompt_len), -100),
                labels
            ], dim=1)
            
            # Pass to decoder
            output = self.llm_decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=full_labels
            )
        else:
            # No labels - inference mode
            output = self.llm_decoder(inputs_embeds=inputs_embeds)
        
        return output
    
    def generate(self, input_features: torch.Tensor, decoder_input_ids: Optional[torch.Tensor] = None, **kwargs):
        """
        Generate token sequences autoregressively.
        
        Args:
            input_features: Preprocessed audio features
            decoder_input_ids: Optional initial text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs
        """
        self.eval()
        
        # Get audio embeddings
        h_audio = self.whisper_encoder.forward(input_features, return_gating=False)
        
        # Project to LLM dimension
        if self.use_adapter and self.prompt_proj is not None:
            audio_prompts = self.prompt_proj(h_audio)
        else:
            audio_prompts = h_audio
        
        # Limit prompt length
        if self.max_prompt_tokens is not None:
            audio_prompts = audio_prompts[:, :self.max_prompt_tokens, :]
        
        # Get text embeddings if provided
        if decoder_input_ids is not None:
            if hasattr(self.llm_decoder, 'model') and hasattr(self.llm_decoder.model, 'embed_tokens'):
                text_embeds = self.llm_decoder.model.embed_tokens(decoder_input_ids)
            elif hasattr(self.llm_decoder, 'transformer') and hasattr(self.llm_decoder.transformer, 'wte'):
                text_embeds = self.llm_decoder.transformer.wte(decoder_input_ids)
            elif hasattr(self.llm_decoder, 'get_input_embeddings'):
                text_embeds = self.llm_decoder.get_input_embeddings()(decoder_input_ids)
            else:
                raise ValueError("Could not find embedding method for decoder.")
            
            inputs_embeds = torch.cat([audio_prompts, text_embeds], dim=1)
        else:
            inputs_embeds = audio_prompts
        
        # Create attention mask
        batch_size, total_seq_len = inputs_embeds.shape[:2]
        attention_mask = torch.ones(
            batch_size, total_seq_len,
            device=inputs_embeds.device, dtype=torch.long
        )
        
        # Generate with LLM
        return self.llm_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.llm_decoder, 'gradient_checkpointing_enable'):
            self.llm_decoder.gradient_checkpointing_enable()
        if hasattr(self.whisper_encoder, 'original_encoder'):
            if hasattr(self.whisper_encoder.original_encoder, 'gradient_checkpointing_enable'):
                self.whisper_encoder.original_encoder.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.llm_decoder, 'gradient_checkpointing_disable'):
            self.llm_decoder.gradient_checkpointing_disable()
        if hasattr(self.whisper_encoder, 'original_encoder'):
            if hasattr(self.whisper_encoder.original_encoder, 'gradient_checkpointing_disable'):
                self.whisper_encoder.original_encoder.gradient_checkpointing_disable()
    
    def get_device(self):
        """Get device of model parameters."""
        return next(self.parameters()).device
    
    @property
    def device(self):
        """Get device property."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            if hasattr(self, 'whisper_encoder') and self.whisper_encoder:
                return next(self.whisper_encoder.parameters()).device
            if hasattr(self, 'llm_decoder') and self.llm_decoder:
                return next(self.llm_decoder.parameters()).device
            return torch.device("cpu")
