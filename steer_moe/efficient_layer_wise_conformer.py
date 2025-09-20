"""
 * author Bixi Zhang
 * created on 21-08-2025
 * github: https://github.com/zbxforward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(process)d - %(levelname)s - %(filename)s>%(funcName)s>%(lineno)d - %(message)s')


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from steer_moe.fireredasr_aed import FireRedAsrAed 


def _get_num_layers(encoder):
    # Recursively find the underlying WhisperEncoder and return the number of layers
    if hasattr(encoder, 'speech_encoder'):
        if hasattr(encoder.speech_encoder, 'layers'):
            return len(encoder.speech_encoder.layers)
        elif hasattr(encoder.speech_encoder, 'layer_stack'):
            return len(encoder.speech_encoder.layer_stack)
        else:
            raise AttributeError("Cannot find layers in the speech encoder.")
    elif hasattr(encoder, '_modules') and 'speech_encoder' in encoder._modules:
        return len(encoder._modules['speech_encoder'].layers)
    elif hasattr(encoder, 'original_encoder'):
        return _get_num_layers(encoder.original_encoder)
    else:
        raise AttributeError("Cannot find layers in the provided encoder object.")

def _get_layers(encoder):
    if hasattr(encoder, 'speech_encoder'):
        return encoder.speech_encoder.layers
    elif hasattr(encoder, '_modules') and 'speech_encoder' in encoder._modules:
        return encoder._modules['speech_encoder'].layers
    elif hasattr(encoder, 'original_encoder'):
        return _get_layers(encoder.original_encoder)
    else:
        raise AttributeError("Cannot find layers in the provided encoder object.")
    
class LoadConformerEncoder(nn.Module):
    def load_encoder(self, model_path):
        assert os.path.exists(model_path)
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = FireRedAsrAed.from_args(package["args"])  
        if "model_state_dict" in package:
            model.load_state_dict(package["model_state_dict"], strict=False)
        encoder = model.encoder
        encoder_dim = encoder.odim
        return encoder, encoder_dim   # return the conformer encoder and its output dimension

    def __init__(self, conformer_encoder_path, freeze = True):
        super().__init__()
        self.speech_encoder, encoder_dim = self.load_encoder(conformer_encoder_path)
        self.config = {}
        self.config['d_model'] = encoder_dim   # output dimension of the conformer encoder, for compatibility
        if freeze:
            for param in self.speech_encoder.parameters():
                param.requires_grad = False

        
    
class EfficientLayerWiseSteeringConformerEncoder(nn.Module):
    """
    Efficient layer-wise steering implementation using a single router.
    This approach uses one router to assign weights to steering vectors for all layers.
    """
    def __init__(self, conformer_encoder_path,
                 num_experts: int = 8,
                 steering_scale: float = 0.1,
                 # steering_scale: float = 0.5,
                 pooling_kernel_size: int = 2,
                 pooling_type: str = "max"):
        super().__init__()
        # self.original_encoder = original_whisper_encoder
        self.original_encoder = LoadConformerEncoder(conformer_encoder_path)    # Use conformer encoder instead
        self.num_experts = num_experts
        self.steering_scale = steering_scale
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_type = pooling_type
        
        # logging.debug(f"self.original_encoder: {self.original_encoder}")
        
        # Get the number of layers from the speech encoder
        self.num_layers = _get_num_layers(self.original_encoder)
        
        # Get the feature dimension from the original encoder
        if hasattr(self.original_encoder, 'config'):
            self.feature_dim = self.original_encoder.config['d_model']    
        else:
            # Default to 1280 for Whisper large
            self.feature_dim = 1280
        
        '''
            Steering modules for each layer, this part is identical to Whisper
        '''
        # Shape: (num_layers, num_experts, feature_dim)
        self.steering_vectors = nn.Parameter(
            torch.randn(self.num_layers, num_experts, self.feature_dim) * 0.01
            # torch.randn(self.num_layers, num_experts, self.feature_dim) * 0.1
        )
        
        # SINGLE router for all layers
        # Output: (batch, seq_len, num_experts * num_layers)
        self.router = nn.Linear(self.feature_dim, num_experts * self.num_layers)
        
        # Layer-specific scaling factors
        self.layer_scales = nn.Parameter(torch.ones(self.num_layers) * steering_scale)
        
        # for para in self.router.parameters():
        #     para.requires_grad = False
        # self.layer_scales.requires_grad = False
        # self.steering_vectors.requires_grad = False
        
        # Freeze the original encoder
        for param in self.original_encoder.parameters():
            param.requires_grad = False
            
        self.original_encoder.eval()
        
        # uncomment the following line to use the pooling layer to downsample the features
        self.pooling_layer = None
        self.init_pooling_layer(pooling_type=self.pooling_type, pooling_kernel_size=self.pooling_kernel_size)


    def init_pooling_layer(self, pooling_type, pooling_kernel_size):
        if pooling_kernel_size is not None:
            if pooling_type == "max":
                self.pooling_layer = nn.MaxPool1d(kernel_size=pooling_kernel_size)
            elif pooling_type == "avg":
                self.pooling_layer = nn.AvgPool1d(kernel_size=pooling_kernel_size)
            else:
                raise NotImplementedError(f"Pooling type {pooling_type} not implemented")

    def _forward_with_steering(self, mel_features, input_lengths, pad=True,return_full_output=False):
        """
        Forward pass through the speech encoder with steering applied.
        
        Args:
            mel_features: Mel spectrogram features from original encoder
            input_lengths: Lengths of the input sequences (one dimension list or array like variable)
            pad: Whether to apply padding before processing.
                 Padding is required for preprocessing conv layers to reduce boundary effect.
                 Padding length shall align receptive field of conv kernel (speech_encoder.input_preprocessor.context - 1).
            return_full_output: Whether to return full output including "enc_output, input_lengths, src_mask, gating_scores_list"
            
        Returns"""  """:
            Steered features with optional gating scores
        """
        # Get the speech encoder
        if hasattr(self.original_encoder, 'speech_encoder'):  
            speech_encoder = self.original_encoder.speech_encoder
        else:
            speech_encoder = self.original_encoder
            
        # Apply initial processing (conv layers, positional embedding, etc.)
        model_device = self.steering_vectors.device
        model_dtype = self.steering_vectors.dtype 

        # logging.debug(f"model_device: {model_device}, model_dtype: {model_dtype}")
        speech_encoder=speech_encoder.to(model_device)
        x = mel_features.to(device=model_device, dtype=model_dtype)
        # logging.debug(f"original x: {x.shape}, {x.dtype}, {x.device}, {x}")

        # x=self.original_encoder._mask_input_features
        
        '''
        --------------------------------------------------------------------------
        Conformer Preprocessing
        --------------------------------------------------------------------------
        '''
        if hasattr(speech_encoder, 'input_preprocessor'):
            if pad:
                x = F.pad(x,
                    (0, 0, 0, speech_encoder.input_preprocessor.context - 1), 'constant', 0.0)
            src_mask = speech_encoder.padding_position_is_0(x, input_lengths)
            x, input_lengths, src_mask = speech_encoder.input_preprocessor(x, src_mask)
        else:
            logging.error(f"no input_preprocessor layer found")
        # logging.debug(f"input_preprocessor x: {x.shape}, {x.dtype}, {x.device}, {x}")
        
            

        if hasattr(speech_encoder, 'dropout'):
            enc_output = speech_encoder.dropout(x)
        else:
            logging.error(f"no dropout layer found")
        # logging.debug(f"dropout x: {enc_output.shape}, {enc_output.dtype}, {enc_output.device}, {enc_output}")

        if hasattr(speech_encoder, 'positional_encoding'):
            pos_emb = speech_encoder.positional_encoding(x)
        else:
            logging.error(f"no positional_encoding layer found")
        # logging.debug(f"positional_encoding x: {pos_emb.shape}, {pos_emb.dtype}, {pos_emb.device}, {pos_emb}")


        if hasattr(speech_encoder, 'dropout'):
            pos_emb = speech_encoder.dropout(pos_emb)
        else:
            logging.error(f"no dropout layer found")
        # logging.debug(f"dropout x: {pos_emb.shape}, {pos_emb.dtype}, {pos_emb.device}, {pos_emb}")

        '''
        --------------------------------------------------------------------------
        Conformer Preprocessing
        --------------------------------------------------------------------------
        '''

        # Process through layers with steering
        gating_scores_list = []
        
        # Get the encoder layers
        if hasattr(speech_encoder, 'layer_stack'):
            encoder_layers = speech_encoder.layer_stack   # layer_stack in conformer
        else:
            logging.error(f"no attention layers found")
            encoder_layers = []
        
        '''
        --------------------------------------------------------------------------
            Layerwise steering process should be identical to Whisper
        -----------------------------------------------------------------------------
        '''



        for layer_idx, layer in enumerate(encoder_layers):
            layer=layer.to(model_device)
            if layer_idx >= self.num_layers:
                # More layers than we have steering for, process normally
                enc_output = layer(enc_output, pos_emb, slf_attn_mask=src_mask,
                pad_mask=src_mask)
                continue
                
            # Apply original layer
            layer_output = layer(enc_output, pos_emb, slf_attn_mask=src_mask,
                                 pad_mask=src_mask)
            # print(layer_output.shape)
            
            # Compute steering for this layer using the single router
            self.router = self.router.to(model_device)
            router_output = self.router(layer_output)  # (batch, seq_len, num_experts * num_layers)
            
            # Extract gating scores for this layer
            start_idx = layer_idx * self.num_experts
            end_idx = (layer_idx + 1) * self.num_experts
            layer_gating_logits = router_output[:, :, start_idx:end_idx]  # (batch, seq_len, num_experts)
            
            # Apply softmax to get gating scores
            gating_scores = F.softmax(layer_gating_logits, dim=-1)
            
            # Get steering vectors for this layer
            steering_vectors = self.steering_vectors[layer_idx]  # (num_experts, feature_dim)
            layer_scale = self.layer_scales[layer_idx]
            
            # Compute steering adjustment
            steering_adjustment = torch.einsum('bte,ef->btf', gating_scores, steering_vectors)
            
            # Apply steering with layer-specific scaling
            enc_output = layer_output + layer_scale * steering_adjustment
            
            # Store gating scores for analysis
            gating_scores_list.append(gating_scores)
        '''
         --------------------------------------------------------------------------
            Steering procecss Completed
         --------------------------------------------------------------------------
        '''

        # Pooling
        enc_output = enc_output.permute(0, 2, 1)
        if enc_output.shape[-1] % self.pooling_kernel_size != 0:
            enc_output = torch.nn.functional.pad(enc_output, (
            0, self.pooling_kernel_size - enc_output.shape[-1] % self.pooling_kernel_size))
        enc_output = self.pooling_layer(enc_output).permute(0, 2, 1)
            
        # logging.debug(f"layers enc_output: {enc_output.shape}, {enc_output.dtype}, {enc_output}")
        input_lengths_f = [i//self.pooling_kernel_size for i in input_lengths]
        if return_full_output:
            return enc_output, input_lengths_f, src_mask, gating_scores_list
        return enc_output
            

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
            
        # for para in self.prompt_proj.parameters():
        #     para.requires_grad = False


    def forward(self, audio_waveform=None, input_features=None,input_lengths=None, decoder_input_ids=None, labels=None, 
                prompt_tokens_only=False, return_full_output=False):
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

# # Comparison of approaches
# def compare_router_approaches():
#     """
#     Compare different router approaches for layer-wise steering.
#     """
#     print("=== Router Approach Comparison ===\n")
    
#     print("1. MULTIPLE ROUTERS (Original Implementation):")
#     print("   - 32 separate routers (one per layer)")
#     print("   - Each router: Linear(feature_dim, num_experts)")
#     print("   - Parameters: 32 * feature_dim * num_experts")
#     print("   - Pros: Independent routing per layer")
#     print("   - Cons: High parameter count, no parameter sharing")
#     print()
    
#     print("2. SINGLE ROUTER (Your Proposed Approach):")
#     print("   - 1 router: Linear(feature_dim, num_experts * num_layers)")
#     print("   - Parameters: feature_dim * num_experts * num_layers")
#     print("   - Pros: Efficient, parameter sharing")
#     print("   - Cons: Less flexibility, all layers share same router")
#     print()
    
#     print("3. SHARED ROUTER WITH LAYER WEIGHTS (Alternative):")
#     print("   - 1 shared router: Linear(feature_dim, num_experts)")
#     print("   - Layer-specific weights: (num_layers, num_experts, num_experts)")
#     print("   - Parameters: feature_dim * num_experts + num_layers * num_experts^2")
#     print("   - Pros: Parameter sharing + layer-specific adaptation")
#     print("   - Cons: More complex, potential redundancy")
#     print()
    
#     # Parameter count comparison
#     feature_dim = 1280
#     num_experts = 8
#     num_layers = 32
    
#     params_multiple = 32 * feature_dim * num_experts
#     params_single = feature_dim * num_experts * num_layers
#     params_shared = feature_dim * num_experts + num_layers * num_experts * num_experts
    
#     print("PARAMETER COUNT COMPARISON:")
#     print(f"Multiple routers: {params_multiple:,} parameters")
#     print(f"Single router: {params_single:,} parameters")
#     print(f"Shared router: {params_shared:,} parameters")
#     print()
    
#     print("RECOMMENDATION:")
#     print("- Use single router approach for efficiency")
#     print("- Use shared router with layer weights if you need more flexibility")
#     print("- Use multiple routers only if you need completely independent routing")


# if __name__ == "__main__":
#     compare_router_approaches() 
