"""
 * author Ruitao Feng
 * created on 16-07-2025
 * github: https://github.com/forfrt
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

from steer_moe.tokenizer.whisper_Lv3.whisper import WhisperEncoder

def _get_num_layers(encoder):
    # Recursively find the underlying WhisperEncoder and return the number of layers
    if hasattr(encoder, 'speech_encoder'):
        return len(encoder.speech_encoder.layers)
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

class EfficientLayerWiseSteeringWhisperEncoder(nn.Module):
    """
    Efficient layer-wise steering implementation using a single router.
    This approach uses one router to assign weights to steering vectors for all layers.
    """
    # def __init__(self, original_whisper_encoder, num_experts: int = 8, steering_scale: float = 0.1):
    def __init__(self, whisper_encoder_path, 
        num_experts: int = 8, 
        steering_scale: float = 0.1, 
        pooling_kernel_size: int = 4, 
        pooling_type: str = None, 
        pooling_position: int = 32):

        super().__init__()
        # self.original_encoder = original_whisper_encoder
        self.original_encoder = WhisperEncoder(whisper_encoder_path)
        self.num_experts = num_experts
        self.steering_scale = steering_scale
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_type = pooling_type
        self.pooling_position = pooling_position

        logging.debug(f"self.original_encoder: {self.original_encoder}")
        
        # Get the number of layers from the speech encoder
        self.num_layers = _get_num_layers(self.original_encoder)
        
        # Get the feature dimension from the original encoder
        if hasattr(self.original_encoder, 'config'):
            self.feature_dim = self.original_encoder.config.d_model
        else:
            # Default to 1280 for Whisper large
            self.feature_dim = 1280
        
        # Steering vectors for each layer
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

        # uncomment the following line to use the pooling layer to downsample the features
        self.pooling_layer = None
        self.init_pooling_layer(pooling_type="max", pooling_kernel_size=4)


    def init_pooling_layer(self, pooling_type, pooling_kernel_size):
        if pooling_kernel_size is not None:
            if pooling_type == "max":
                self.pooling_layer = nn.MaxPool1d(kernel_size=pooling_kernel_size)
            elif pooling_type == "avg":
                self.pooling_layer = nn.AvgPool1d(kernel_size=pooling_kernel_size)
            else:
                raise NotImplementedError(f"Pooling type {pooling_type} not implemented")

    
    def tokenize_waveform(self, audio, return_gating=False):
        """
        Tokenize waveform with efficient layer-wise steering.
        This method integrates with the original WhisperEncoder.tokenize_waveform 
        but applies steering during the encoding process.
        
        Args:
            audio: Audio waveform tensor or preprocessed features
            return_gating: Whether to return gating scores for analysis
            
        Returns:
            Steered audio features, optionally with gating scores
        """
        # In the training pipeline, input is always preprocessed features
        # For raw waveform processing (inference), we keep the original logic
        if (audio.dim() == 3 and audio.size(-1) == 1280) or (audio.dim() == 2 and audio.size(-1) == 1280):
            # Input is already preprocessed Whisper features - apply steering directly
            mel_features = audio
        else:
            # Raw waveform - process through original encoder first
            if hasattr(self.original_encoder, 'tokenize_waveform'):
                mel_features = self.original_encoder.tokenize_waveform(audio)
            else:
                # Fallback: assume audio is already processed mel spectrogram
                mel_features = audio
        
        # Now apply our layer-wise steering to the mel features
        # We need to process through the speech encoder with steering
        if hasattr(self.original_encoder, 'speech_encoder'):
            # Process mel features through our steered encoder layers
            steered_features = self._forward_with_steering(mel_features, return_gating)
        else:
            # Fallback to direct forward
            steered_features = self.forward(mel_features, return_gating)
            
        return steered_features
    
    def _forward_with_steering(self, mel_features, return_gating=False):
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
        
        # If the speech encoder has conv layers and embeddings, apply them first
        if hasattr(speech_encoder, 'conv1'):
            # x = torch.nn.functional.gelu(speech_encoder.conv1(x.transpose(1, 2)))
            x = torch.nn.functional.gelu(speech_encoder.conv1(x))
            x = torch.nn.functional.gelu(speech_encoder.conv2(x))
            # x = x.transpose(1, 2)
            x=x.permute(0, 2, 1)
        else:
            logging.error(f"no conv1 layer found")

        logging.debug(f"conv1+conv2 x: {x.shape}, {x.dtype}, {x.device}, {x}")
            
        if hasattr(speech_encoder, 'embed_positions'):
            # positions = speech_encoder.embed_positions.weight[:x.size(1)]
            positions = speech_encoder.embed_positions.weight[:x.size(1)].to(model_device)
            x = (x + positions) * speech_encoder.embed_scale
        else:
            logging.error(f"no embed_positions layer found")

        logging.debug(f"embed_positions x: {x.shape}, {x.dtype}, {x.device}, {x}")
            
        # Apply dropout if present
        if hasattr(speech_encoder, 'dropout'):
            x = nn.functional.dropout(x, p=speech_encoder.dropout, training=speech_encoder.training)
        else:
            logging.error(f"no dropout layer found")
        
        # Now process through layers with steering
        gating_scores_list = []
        
        # Get the encoder layers
        if hasattr(speech_encoder, 'layers'):
            encoder_layers = speech_encoder.layers
        else:
            logging.error(f"no attention layers found")
            encoder_layers = []
            
        for layer_idx, layer in enumerate(encoder_layers):
            layer=layer.to(model_device)
            if layer_idx >= self.num_layers:
                # More layers than we have steering for, process normally
                x = layer(x)
                continue
                
            # Apply original layer
            layer_output = layer(x)[0]
            
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
            x = layer_output + layer_scale * steering_adjustment
            
            # Store gating scores for analysis
            gating_scores_list.append(gating_scores)

            if layer_idx + 1 == self.pooling_position and self.pooling_kernel_size is not None:
                x = x.permute(0, 2, 1)
                if x.shape[-1] % self.pooling_kernel_size != 0:
                    x = torch.nn.functional.pad(x, (
                    0, self.pooling_kernel_size - x.shape[-1] % self.pooling_kernel_size))
                x = self.pooling_layer(x).permute(0, 2, 1)


        logging.debug(f"layers x: {x.shape}, {x.dtype}, {x}")
        
        # Apply final layer norm if exists
        if hasattr(speech_encoder, 'layer_norm'):
            x = speech_encoder.layer_norm(x)
        else:
            logging.error(f"no layer_norm layers found")

        logging.debug(f"layer_norm x: {x.shape}, {x.dtype}, {x}")
            
        if return_gating:
            return x, gating_scores_list
        return x

    def forward(self, mel_features, return_gating=False):
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
        
        # If the speech encoder has conv layers and embeddings, apply them first
        if hasattr(speech_encoder, 'conv1'):
            # x = torch.nn.functional.gelu(speech_encoder.conv1(x.transpose(1, 2)))
            x = torch.nn.functional.gelu(speech_encoder.conv1(x))
            x = torch.nn.functional.gelu(speech_encoder.conv2(x))
            # x = x.transpose(1, 2)
            x=x.permute(0, 2, 1)
        else:
            logging.error(f"no conv1 layer found")

        logging.debug(f"conv1+conv2 x: {x.shape}, {x.dtype}, {x.device}, {x}")
            
        if hasattr(speech_encoder, 'embed_positions'):
            # positions = speech_encoder.embed_positions.weight[:x.size(1)]
            positions = speech_encoder.embed_positions.weight[:x.size(1)].to(model_device)
            x = (x + positions) * speech_encoder.embed_scale
        else:
            logging.error(f"no embed_positions layer found")

        logging.debug(f"embed_positions x: {x.shape}, {x.dtype}, {x.device}, {x}")
            
        # Apply dropout if present
        if hasattr(speech_encoder, 'dropout'):
            x = nn.functional.dropout(x, p=speech_encoder.dropout, training=speech_encoder.training)
        else:
            logging.error(f"no dropout layer found")
        
        # Now process through layers with steering
        gating_scores_list = []
        
        # Get the encoder layers
        if hasattr(speech_encoder, 'layers'):
            encoder_layers = speech_encoder.layers
        else:
            logging.error(f"no attention layers found")
            encoder_layers = []
            
        for layer_idx, layer in enumerate(encoder_layers):
            layer=layer.to(model_device)
            if layer_idx >= self.num_layers:
                # More layers than we have steering for, process normally
                x = layer(x)
                continue
                
            # Apply original layer
            layer_output = layer(x)[0]
            
            x = layer_output

            if layer_idx + 1 == self.pooling_position and self.pooling_kernel_size is not None:
                x = x.permute(0, 2, 1)
                if x.shape[-1] % self.pooling_kernel_size != 0:
                    x = torch.nn.functional.pad(x, (
                    0, self.pooling_kernel_size - x.shape[-1] % self.pooling_kernel_size))
                x = self.pooling_layer(x).permute(0, 2, 1)


        logging.debug(f"layers x: {x.shape}, {x.dtype}, {x}")
        
        # Apply final layer norm if exists
        if hasattr(speech_encoder, 'layer_norm'):
            x = speech_encoder.layer_norm(x)
        else:
            logging.error(f"no layer_norm layers found")

        logging.debug(f"layer_norm x: {x.shape}, {x.dtype}, {x}")
            
        if return_gating:
            return x, gating_scores_list
        return x
    
    def get_steering_analysis(self, gating_scores_list):
        """
        Analyze steering patterns across layers.
        
        Args:
            gating_scores_list: List of gating scores from forward pass
            
        Returns:
            Dictionary with steering analysis
        """
        analysis = {
            'layer_usage': [],
            'expert_diversity': [],
            'steering_strength': [],
            'layer_scale_values': self.layer_scales.detach().cpu().numpy().tolist()
        }
        
        for layer_idx, gating_scores in enumerate(gating_scores_list):
            # Average gating scores across batch and sequence
            avg_gating = gating_scores.mean(dim=(0, 1))  # (num_experts,)
            
            # Which experts are most used
            top_experts = torch.topk(avg_gating, k=3, dim=-1)
            
            # Diversity measure (entropy of gating distribution)
            entropy = -torch.sum(avg_gating * torch.log(avg_gating + 1e-8))
            
            analysis['layer_usage'].append({
                'layer': layer_idx,
                'top_experts': top_experts.indices.tolist(),
                'top_scores': top_experts.values.tolist(),
                'entropy': entropy.item()
            })
            
            analysis['expert_diversity'].append(entropy.item())
            analysis['steering_strength'].append(avg_gating.max().item())
        
        return analysis


# Alternative: Shared Router with Layer-Specific Weights
class SharedRouterLayerWiseSteeringWhisperEncoder(nn.Module):
    """
    Alternative approach: Shared router with layer-specific weight matrices.
    This approach uses one router but applies different weight matrices per layer.
    """
    def __init__(self, original_whisper_encoder, num_experts: int = 8, steering_scale: float = 0.1):
        super().__init__()
        self.original_encoder = original_whisper_encoder
        self.num_experts = num_experts
        self.steering_scale = steering_scale
        self.num_layers = len(original_whisper_encoder.layers)
        
        # Get the feature dimension
        if hasattr(original_whisper_encoder, 'config'):
            self.feature_dim = original_whisper_encoder.config.d_model
        else:
            self.feature_dim = 1280
        
        # Steering vectors for each layer
        self.steering_vectors = nn.Parameter(
            torch.randn(self.num_layers, num_experts, self.feature_dim) * 0.01
        )
        
        # Shared router
        self.shared_router = nn.Linear(self.feature_dim, num_experts)
        
        # Layer-specific weight matrices to transform shared router output
        # Shape: (num_layers, num_experts, num_experts)
        self.layer_weights = nn.Parameter(
            torch.eye(num_experts).unsqueeze(0).repeat(self.num_layers, 1, 1)
        )
        
        # Layer-specific scaling factors
        self.layer_scales = nn.Parameter(torch.ones(self.num_layers) * steering_scale)
        
        # Freeze the original encoder
        for param in self.original_encoder.parameters():
            param.requires_grad = False
            
    def forward(self, mel_spec, return_gating=False):
        """
        Forward pass with shared router and layer-specific weights.
        """
        gating_scores_list = []
        encoder_layers = self.original_encoder.layers
        
        x = mel_spec
        for layer_idx, layer in enumerate(encoder_layers):
            # 1. Apply original layer
            layer_output = layer(x)
            
            # 2. Get shared router output
            shared_gating_logits = self.shared_router(layer_output)  # (batch, seq_len, num_experts)
            
            # 3. Apply layer-specific weight matrix
            layer_weights = self.layer_weights[layer_idx]  # (num_experts, num_experts)
            layer_gating_logits = torch.einsum('bte,ef->btf', shared_gating_logits, layer_weights)
            
            # 4. Apply softmax
            gating_scores = F.softmax(layer_gating_logits, dim=-1)
            
            # 5. Get steering vectors and apply
            steering_vectors = self.steering_vectors[layer_idx]  # (num_experts, feature_dim)
            layer_scale = self.layer_scales[layer_idx]
            
            steering_adjustment = torch.einsum('bte,ef->btf', gating_scores, steering_vectors)
            steered_output = layer_output + layer_scale * steering_adjustment
            
            gating_scores_list.append(gating_scores)
            x = steered_output
        
        # Apply final layer norm if exists
        if hasattr(self.original_encoder, 'layer_norm'):
            final_output = self.original_encoder.layer_norm(x)
        else:
            final_output = x
            
        if return_gating:
            return final_output, gating_scores_list
        return final_output


# Comparison of approaches
def compare_router_approaches():
    """
    Compare different router approaches for layer-wise steering.
    """
    print("=== Router Approach Comparison ===\n")
    
    print("1. MULTIPLE ROUTERS (Original Implementation):")
    print("   - 32 separate routers (one per layer)")
    print("   - Each router: Linear(feature_dim, num_experts)")
    print("   - Parameters: 32 * feature_dim * num_experts")
    print("   - Pros: Independent routing per layer")
    print("   - Cons: High parameter count, no parameter sharing")
    print()
    
    print("2. SINGLE ROUTER (Your Proposed Approach):")
    print("   - 1 router: Linear(feature_dim, num_experts * num_layers)")
    print("   - Parameters: feature_dim * num_experts * num_layers")
    print("   - Pros: Efficient, parameter sharing")
    print("   - Cons: Less flexibility, all layers share same router")
    print()
    
    print("3. SHARED ROUTER WITH LAYER WEIGHTS (Alternative):")
    print("   - 1 shared router: Linear(feature_dim, num_experts)")
    print("   - Layer-specific weights: (num_layers, num_experts, num_experts)")
    print("   - Parameters: feature_dim * num_experts + num_layers * num_experts^2")
    print("   - Pros: Parameter sharing + layer-specific adaptation")
    print("   - Cons: More complex, potential redundancy")
    print()
    
    # Parameter count comparison
    feature_dim = 1280
    num_experts = 8
    num_layers = 32
    
    params_multiple = 32 * feature_dim * num_experts
    params_single = feature_dim * num_experts * num_layers
    params_shared = feature_dim * num_experts + num_layers * num_experts * num_experts
    
    print("PARAMETER COUNT COMPARISON:")
    print(f"Multiple routers: {params_multiple:,} parameters")
    print(f"Single router: {params_single:,} parameters")
    print(f"Shared router: {params_shared:,} parameters")
    print()
    
    print("RECOMMENDATION:")
    print("- Use single router approach for efficiency")
    print("- Use shared router with layer weights if you need more flexibility")
    print("- Use multiple routers only if you need completely independent routing")


if __name__ == "__main__":
    compare_router_approaches() 