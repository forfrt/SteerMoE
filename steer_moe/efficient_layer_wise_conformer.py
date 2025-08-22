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

from .fireredasr_aed import FireRedAsrAed

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
    def __init__(self, conformer_encoder_path, num_experts: int = 8, steering_scale: float = 0.1):
        super().__init__()
        # self.original_encoder = original_whisper_encoder
        self.original_encoder = LoadConformerEncoder(conformer_encoder_path)    # Use conformer encoder instead
        self.num_experts = num_experts
        self.steering_scale = steering_scale
        logging.debug(f"self.original_encoder: {self.original_encoder}")
        
        # Get the number of layers from the speech encoder
        self.num_layers = _get_num_layers(self.original_encoder)
        
        # Get the feature dimension from the original encoder
        if hasattr(self.original_encoder, 'config'):
            self.feature_dim = self.original_encoder.config.d_model    
        else:
            # Default to 1280 for Whisper large
            self.feature_dim = 1280
        
        '''
            Steering modules for each layer, this part is identical to Whisper
        '''
        # Shape: (num_layers, num_experts, feature_dim)
        self.steering_vectors = nn.Parameter(
            torch.randn(self.num_layers, num_experts, self.feature_dim) * 0.01
        )
        
        # SINGLE router for all layers
        # Output: (batch, seq_len, num_experts * num_layers)
        self.router = nn.Linear(self.feature_dim, num_experts * self.num_layers)
        
        # Layer-specific scaling factors
        self.layer_scales = nn.Parameter(torch.ones(self.num_layers) * steering_scale)
        
        # Freeze the original encoder
        for param in self.original_encoder.parameters():
            param.requires_grad = False

    def forward_with_steering(self, mel_features, input_lengths, pad=True,return_full_output=False):
        """
        Forward pass through the speech encoder with steering applied.
        
        Args:
            mel_features: Mel spectrogram features from original encoder
            return_gating: Whether to return gating scores
            
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

        logging.debug(f"model_device: {model_device}, model_dtype: {model_dtype}")
        speech_encoder=speech_encoder.to(model_device)
        x = mel_features.to(device=model_device, dtype=model_dtype)
        logging.debug(f"original x: {x.shape}, {x.dtype}, {x.device}, {x}")

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
        logging.debug(f"input_preprocessor x: {x.shape}, {x.dtype}, {x.device}, {x}")
            

        if hasattr(speech_encoder, 'dropout'):
            enc_output = speech_encoder.dropout(x)
        else:
            logging.error(f"no dropout layer found")
        logging.debug(f"dropout x: {enc_output.shape}, {enc_output.dtype}, {enc_output.device}, {enc_output}")
            

        if hasattr(speech_encoder, 'positional_encoding'):
            pos_emb = speech_encoder.positional_encoding(x)
        else:
            logging.error(f"no positional_encoding layer found")
        logging.debug(f"positional_encoding x: {pos_emb.shape}, {pos_emb.dtype}, {pos_emb.device}, {pos_emb}")


        if hasattr(speech_encoder, 'dropout'):
            pos_emb = speech_encoder.dropout(pos_emb)
        else:
            logging.error(f"no dropout layer found")
        logging.debug(f"dropout x: {pos_emb.shape}, {pos_emb.dtype}, {pos_emb.device}, {pos_emb}")

        '''
        --------------------------------------------------------------------------
        Conformer Preprocessing
        --------------------------------------------------------------------------
        '''

        # Process through layers with steering
        gating_scores_list = []
        
        # Get the encoder layers
        if hasattr(speech_encoder, 'layers'):
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
                                 pad_mask=src_mask)[0]      #TODO check the output format of conformer layer, this might cause error
            
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

        logging.debug(f"layers enc_output: {enc_output.shape}, {enc_output.dtype}, {enc_output}")

            
        if return_full_output:
            return enc_output, input_lengths, src_mask, gating_scores_list
        return enc_output
            



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