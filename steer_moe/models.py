"""
 * author Ruitao Feng
 * created on 16-07-2025
 * github: https://github.com/forfrt
"""

import torch
import torch.nn as nn
import logging

# Import WhisperEncoder from Kimi-Audio
from .tokenizer.whisper_Lv3.whisper import WhisperEncoder

# Import SteerMoEAligner
from .aligner import SteerMoEAligner

# Import layer-wise steering
from .layer_wise_whisper import LayerWiseSteeringWhisperEncoder
from .efficient_layer_wise_whisper import EfficientLayerWiseSteeringWhisperEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

# TODO: Import or load LLM decoder (e.g., from transformers)

# Audio → Encoder → MoE_Steering → LLM → Output
class SteerMoEModel(nn.Module):
    def __init__(self, whisper_encoder, aligner, llm_decoder):
        super().__init__()
        self.whisper_encoder = whisper_encoder  # frozen
        self.aligner = aligner                  # trainable
        self.llm_decoder = llm_decoder          # frozen

        # Freeze encoder and decoder
        for p in self.whisper_encoder.parameters():
            p.requires_grad = False
        for p in self.llm_decoder.parameters():
            p.requires_grad = False

    def forward(self, audio_waveform, *args, **kwargs):
        # 1. Extract continuous features from audio using frozen Whisper encoder
        with torch.no_grad():
            h_audio = self.whisper_encoder.tokenize_waveform(audio_waveform)
            # h_audio: (batch, seq_len, feature_dim)

        # 2. Align features using trainable SteerMoE aligner
        h_aligned = self.aligner(h_audio)
        # h_aligned: (batch, seq_len, feature_dim)

        # 3. Feed aligned features as prompt to frozen LLM decoder
        with torch.no_grad():
            output = self.llm_decoder(inputs_embeds=h_aligned)

        return output


# Audio → WhisperEncoder → SteerMoEAligner → [Optional Projection] → Continuous Prompts + Text Embeddings → LLM Decoder
class SteerMoEHybridModel(nn.Module):
    """
    Hybrid model that combines SteerMoE aligner with continuous prompt approach.
    Uses SteerMoE to process audio features, then concatenates them as continuous
    prompts with text embeddings for the LLM decoder.
    """
    def __init__(self, whisper_encoder, aligner, llm_decoder, prompt_proj=None, 
                 max_prompt_tokens=None, use_adapter=True):
        super().__init__()
        self.whisper_encoder = whisper_encoder  # frozen
        self.aligner = aligner                  # trainable
        self.llm_decoder = llm_decoder          # frozen
        self.use_adapter = use_adapter
        self.max_prompt_tokens = max_prompt_tokens

        # Freeze encoder and decoder
        for p in self.whisper_encoder.parameters():
            p.requires_grad = False
        for p in self.llm_decoder.parameters():
            p.requires_grad = False

        # Optional projection layer if encoder and decoder dimensions don't match
        if prompt_proj is None and use_adapter:
            # Get dimensions from aligner output and decoder input
            aligner_output_dim = aligner.feature_dim
            # Try to get decoder input dimension
            if hasattr(llm_decoder, 'config'):
                decoder_input_dim = getattr(llm_decoder.config, 'n_embd', 
                                          getattr(llm_decoder.config, 'hidden_size', aligner_output_dim))
            else:
                decoder_input_dim = aligner_output_dim
            
            if aligner_output_dim != decoder_input_dim:
                self.prompt_proj = nn.Linear(aligner_output_dim, decoder_input_dim)
            else:
                self.prompt_proj = None
        else:
            self.prompt_proj = prompt_proj

    def forward(self, audio_waveform, decoder_input_ids=None, labels=None, 
                prompt_tokens_only=False):
        """
        Forward pass for hybrid SteerMoE model.
        
        Args:
            audio_waveform: Audio input tensor
            decoder_input_ids: Text token IDs (optional for pure audio generation)
            labels: Target labels for training (optional)
            prompt_tokens_only: If True, only return prompt embeddings without text
        
        Returns:
            Model output (logits or loss depending on labels)
        """
        # 1. Extract continuous features from audio using frozen Whisper encoder
        with torch.no_grad():
            h_audio = self.whisper_encoder.tokenize_waveform(audio_waveform)
            # h_audio: (batch, seq_len, feature_dim)

        # 2. Align features using trainable SteerMoE aligner
        h_aligned = self.aligner(h_audio)
        # h_aligned: (batch, seq_len, feature_dim)

        # 3. Project to decoder dimension if needed
        if self.use_adapter and self.prompt_proj is not None:
            prompts = self.prompt_proj(h_aligned)
        else:
            prompts = h_aligned
        # prompts: (batch, seq_len, decoder_dim)

        # 4. Limit prompt length if specified
        if self.max_prompt_tokens is not None:
            prompts = prompts[:, :self.max_prompt_tokens, :]

        # 5. Handle text embeddings and concatenation
        if decoder_input_ids is not None:
            # Get text embeddings from decoder
            if hasattr(self.llm_decoder, 'model') and hasattr(self.llm_decoder.model, 'embed_tokens'):
                # LLaMA-like decoder
                input_embeds = self.llm_decoder.model.embed_tokens(decoder_input_ids)
            elif hasattr(self.llm_decoder, 'transformer') and hasattr(self.llm_decoder.transformer, 'wte'):
                # GPT-2 like decoder
                input_embeds = self.llm_decoder.transformer.wte(decoder_input_ids)
            else:
                raise ValueError("Decoder embedding method not found. Adapt this part.")

            # Concatenate prompts with text embeddings
            inputs_embeds = torch.cat([prompts, input_embeds], dim=1)
            # inputs_embeds: (batch, prompt_len + text_len, decoder_dim)

            # Handle labels for training
            if labels is not None:
                prompt_len = prompts.size(1)
                # Mask prompt tokens with -100 (ignored in loss)
                labels = torch.cat([
                    labels.new_full((labels.size(0), prompt_len), -100),
                    labels
                ], dim=1)

            # Pass to decoder
            output = self.llm_decoder(
                inputs_embeds=inputs_embeds,
                labels=labels
            )
        else:
            # Pure audio generation - only use prompts
            if prompt_tokens_only:
                return prompts
            else:
                # For generation, we'll handle this in the generation loop
                # For now, just return the prompts
                return prompts

        return output

    def get_device(self):
        return next(self.parameters()).device

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            if hasattr(self, 'whisper_encoder') and self.whisper_encoder:
                return next(self.whisper_encoder.parameters()).device
            if hasattr(self, 'llm_decoder') and self.llm_decoder:
                return next(self.llm_decoder.parameters()).device
            return torch.device("cpu")


# NEW: Layer-wise steering model (Original implementation with multiple routers)
class SteerMoELayerWiseModel(nn.Module):
    """
    Model that uses layer-wise steering within the Whisper encoder.
    This approach adds steering vectors to each layer of the Whisper encoder,
    providing fine-grained control over audio processing.
    """
    def __init__(self, whisper_encoder_path, llm_decoder, num_experts=8, 
                 prompt_proj=None, max_prompt_tokens=None, use_adapter=True):
        super().__init__()
        
        # Create layer-wise steering Whisper encoder
        self.whisper_encoder = LayerWiseSteeringWhisperEncoder(
            whisper_encoder_path, num_experts=num_experts
        )
        
        self.llm_decoder = llm_decoder          # frozen
        self.use_adapter = use_adapter
        self.max_prompt_tokens = max_prompt_tokens

        # Freeze decoder
        for p in self.llm_decoder.parameters():
            p.requires_grad = False

        # Optional projection layer if encoder and decoder dimensions don't match
        if prompt_proj is None and use_adapter:
            # Get dimensions from encoder output and decoder input
            encoder_output_dim = self.whisper_encoder.feature_dim
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

    def forward(self, audio_waveform, decoder_input_ids=None, labels=None, 
                prompt_tokens_only=False, return_gating=False):
        """
        Forward pass for layer-wise steering model.
        
        Args:
            audio_waveform: Audio input tensor
            decoder_input_ids: Text token IDs (optional for pure audio generation)
            labels: Target labels for training (optional)
            prompt_tokens_only: If True, only return prompt embeddings without text
            return_gating: If True, return gating scores for analysis
        
        Returns:
            Model output (logits or loss depending on labels)
        """
        # 1. Extract continuous features from audio using layer-wise steering Whisper encoder
        if return_gating:
            h_audio, gating_scores = self.whisper_encoder.tokenize_waveform(
                audio_waveform, return_gating=True
            )
        else:
            h_audio = self.whisper_encoder.tokenize_waveform(audio_waveform)
        # h_audio: (batch, seq_len, feature_dim)

        # 2. Project to decoder dimension if needed
        if self.use_adapter and self.prompt_proj is not None:
            prompts = self.prompt_proj(h_audio)
        else:
            prompts = h_audio
        # prompts: (batch, seq_len, decoder_dim)

        # 3. Limit prompt length if specified
        if self.max_prompt_tokens is not None:
            prompts = prompts[:, :self.max_prompt_tokens, :]

        # 4. Handle text embeddings and concatenation
        if decoder_input_ids is not None:
            # Get text embeddings from decoder
            if hasattr(self.llm_decoder, 'model') and hasattr(self.llm_decoder.model, 'embed_tokens'):
                # LLaMA-like decoder
                input_embeds = self.llm_decoder.model.embed_tokens(decoder_input_ids)
            elif hasattr(self.llm_decoder, 'transformer') and hasattr(self.llm_decoder.transformer, 'wte'):
                # GPT-2 like decoder
                input_embeds = self.llm_decoder.transformer.wte(decoder_input_ids)
            else:
                raise ValueError("Decoder embedding method not found. Adapt this part.")

            # Concatenate prompts with text embeddings
            inputs_embeds = torch.cat([prompts, input_embeds], dim=1)
            # inputs_embeds: (batch, prompt_len + text_len, decoder_dim)

            # Handle labels for training
            if labels is not None:
                prompt_len = prompts.size(1)
                # Mask prompt tokens with -100 (ignored in loss)
                labels = torch.cat([
                    labels.new_full((labels.size(0), prompt_len), -100),
                    labels
                ], dim=1)

            # Pass to decoder
            output = self.llm_decoder(
                inputs_embeds=inputs_embeds,
                labels=labels
            )
        else:
            # Pure audio generation - only use prompts
            if prompt_tokens_only:
                if return_gating:
                    return prompts, gating_scores
                else:
                    return prompts
            else:
                # For generation, we'll handle this in the generation loop
                if return_gating:
                    return prompts, gating_scores
                else:
                    return prompts

        if return_gating:
            return output, gating_scores
        return output

    def get_steering_analysis(self, gating_scores):
        """
        Analyze steering patterns across layers.
        
        Args:
            gating_scores: Gating scores from forward pass
            
        Returns:
            Dictionary with steering analysis
        """
        return self.whisper_encoder.get_steering_analysis(gating_scores)

    def get_device(self):
        return next(self.parameters()).device

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            if hasattr(self, 'whisper_encoder') and self.whisper_encoder:
                return next(self.whisper_encoder.parameters()).device
            if hasattr(self, 'llm_decoder') and self.llm_decoder:
                return next(self.llm_decoder.parameters()).device
            return torch.device("cpu")


# NEW: Efficient layer-wise steering model (Single router approach)
class SteerMoEEfficientLayerWiseModel(nn.Module):
    """
    Efficient layer-wise steering model using a single router.
    This approach uses one router to assign weights to steering vectors for all layers,
    making it much more parameter-efficient than the multiple router approach.
    """
    def __init__(self, whisper_encoder_path, llm_decoder, num_experts=8, 
                 prompt_proj=None, max_prompt_tokens=None, use_adapter=True):
        super().__init__()
        
        # Create efficient layer-wise steering Whisper encoder
        self.whisper_encoder = EfficientLayerWiseSteeringWhisperEncoder(
            whisper_encoder_path, num_experts=num_experts
        )
        
        self.llm_decoder = llm_decoder          # frozen
        self.use_adapter = use_adapter
        self.max_prompt_tokens = max_prompt_tokens

        # Freeze decoder
        for p in self.llm_decoder.parameters():
            p.requires_grad = False

        # Optional projection layer if encoder and decoder dimensions don't match
        if prompt_proj is None and use_adapter:
            # Get dimensions from encoder output and decoder input
            encoder_output_dim = self.whisper_encoder.feature_dim
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


    def forward(self, audio_waveform=None, input_features=None, decoder_input_ids=None, labels=None, 
                prompt_tokens_only=False, return_gating=False):
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
        if return_gating:
            # Apply layer-wise steering to the preprocessed features
            h_audio, gating_scores = self.whisper_encoder._forward_with_steering(audio_input, return_gating=True)
        else:
            # Apply layer-wise steering to the preprocessed features  
            h_audio = self.whisper_encoder._forward_with_steering(audio_input, return_gating=False)
            gating_scores = None
        
        # h_audio: (batch, audio_seq_len, feature_dim)

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
                if return_gating:
                    return audio_prompts, gating_scores
                else:
                    return audio_prompts
            else:
                # For generation without text input
                if return_gating:
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
        inputs_embeds = torch.cat([audio_prompts, text_embeds], dim=1)
        # inputs_embeds: (batch, audio_seq_len + text_seq_len, decoder_dim)

        # 7. Handle labels for training
        if labels is not None:
            audio_prompt_len = audio_prompts.size(1)
            
            # Create attention mask for the combined sequence
            # We want to attend to both audio prompts and text, but only compute loss on text
            batch_size, total_seq_len = inputs_embeds.shape[:2]
            attention_mask = torch.ones(batch_size, total_seq_len, device=inputs_embeds.device, dtype=torch.long)
            
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
        if return_gating:
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

    def get_steering_analysis(self, gating_scores):
        """
        Analyze steering patterns across layers.
        
        Args:
            gating_scores: Gating scores from forward pass
            
        Returns:
            Dictionary with steering analysis
        """
        return self.whisper_encoder.get_steering_analysis(gating_scores)

    def get_device(self):
        return next(self.parameters()).device

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            if hasattr(self, 'whisper_encoder') and self.whisper_encoder:
                return next(self.whisper_encoder.parameters()).device
            if hasattr(self, 'llm_decoder') and self.llm_decoder:
                return next(self.llm_decoder.parameters()).device
            return torch.device("cpu")


def hybrid_model_test():
    # Initialize model
    model = SteerMoEHybridModel(
        whisper_encoder=whisper_encoder,
        aligner=aligner,
        llm_decoder=llm_decoder,
        max_prompt_tokens=512  # Limit audio prompt length
    )

    # Forward pass with audio and text
    output = model(
        audio_waveform=audio,
        decoder_input_ids=text_tokens,
        labels=target_labels
    )
    # Labels are automatically masked for prompt tokens

    # Get prompt embeddings only
    prompts = model(
        audio_waveform=audio,
        prompt_tokens_only=True
    )

    # Use prompts for generation
    generated = llm_decoder.generate(inputs_embeds=prompts)

    # Audio + instruction + response
    output = model(
        audio_waveform=audio,
        decoder_input_ids=instruction_tokens + response_tokens,
        labels=response_labels
    )

def extract_whisper_feat(self, wav: torch.Tensor | str):
    if isinstance(wav, str):
        wav = librosa.load(wav, sr=16000)[0]

        wav_tensor = torch.tensor(wav).unsqueeze(0)[:, :]
    elif isinstance(wav, torch.Tensor):
        wav_tensor = wav
    else:
        raise ValueError(f"Invalid wav type: {type(wav)}")
    assert self.whisper_model is not None
    wav_tensor = wav_tensor.to(torch.cuda.current_device())
    continous_feature = self.whisper_model.tokenize_waveform(wav_tensor)
    continous_feature = continous_feature.reshape(
        continous_feature.shape[0],
        int(continous_feature.shape[1] // 4),
        continous_feature.shape[2] * 4,
    )
    return continous_feature

if __name__ == "__main__":

    test_audio="/root/autodl-nas/ruitao/SteerMoE/tests/output_audio.wav"
    input_features=extract_whisper_feat(test_audio)
    logging.info(f"input_features shape: {input_features.shape}")
    steer_moe_model=SteerMoEEfficientLayerWiseModel(
        whisper_encoder_path="/root/autodl-nas/ruitao/SteerMoE/steer_moe/tokenizer/whisper_Lv3/whisper_large-v2",
        llm_decoder=llm_decoder,
        num_experts=8,
        prompt_proj=None,
        max_prompt_tokens=512,
    )
    output=steer_moe_model(input_features)
    logging.info(f"output shape: {output.shape}")


    
