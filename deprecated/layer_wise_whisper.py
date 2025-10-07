"""
 * author Ruitao Feng
 * created on 16-07-2025
 * github: https://github.com/forfrt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class LayerWiseSteeringWhisperEncoder(nn.Module):
    """
    Modified Whisper encoder with layer-wise steering vectors.
    This implementation adds steering vectors to each layer of the Whisper encoder.
    """
    def __init__(self, original_whisper_encoder, num_experts: int = 8, steering_scale: float = 0.1):
        super().__init__()
        self.original_encoder = original_whisper_encoder
        self.num_experts = num_experts
        self.steering_scale = steering_scale
        
        # Get the feature dimension from the original encoder
        if hasattr(original_whisper_encoder, 'config'):
            self.feature_dim = original_whisper_encoder.config.d_model
        else:
            # Default to 1280 for Whisper large
            self.feature_dim = 1280
        
        # Steering vectors for each layer
        # Shape: (num_layers, num_experts, feature_dim)
        num_layers = len(original_whisper_encoder.layers)
        self.steering_vectors = nn.Parameter(
            torch.randn(num_layers, num_experts, self.feature_dim) * 0.01
        )
        
        # Router for each layer
        # Shape: (num_layers, feature_dim, num_experts)
        self.routers = nn.ModuleList([
            nn.Linear(self.feature_dim, num_experts) 
            for _ in range(num_layers)
        ])
        
        # Layer-specific scaling factors
        self.layer_scales = nn.Parameter(torch.ones(num_layers) * steering_scale)
        
        # Freeze the original encoder
        for param in self.original_encoder.parameters():
            param.requires_grad = False
            
    def forward(self, mel_spec, return_gating=False):
        """
        Forward pass with layer-wise steering.
        
        Args:
            mel_spec: Mel spectrogram input
            return_gating: Whether to return gating scores for analysis
            
        Returns:
            Steered encoder output
        """
        # Store gating scores for analysis
        gating_scores_list = []
        
        # Get the encoder layers
        encoder_layers = self.original_encoder.layers
        
        # Process through each layer with steering
        x = mel_spec
        for layer_idx, layer in enumerate(encoder_layers):
            # 1. Apply original layer
            layer_output = layer(x)
            
            # 2. Compute steering for this layer
            router = self.routers[layer_idx]
            steering_vectors = self.steering_vectors[layer_idx]  # (num_experts, feature_dim)
            layer_scale = self.layer_scales[layer_idx]
            
            # 3. Compute gating scores
            gating_logits = router(layer_output)  # (batch, seq_len, num_experts)
            gating_scores = F.softmax(gating_logits, dim=-1)
            
            # 4. Compute steering adjustment
            # (batch, seq_len, num_experts) @ (num_experts, feature_dim) -> (batch, seq_len, feature_dim)
            steering_adjustment = torch.einsum('bte,ef->btf', gating_scores, steering_vectors)
            
            # 5. Apply steering with layer-specific scaling
            steered_output = layer_output + layer_scale * steering_adjustment
            
            # Store gating scores for analysis
            gating_scores_list.append(gating_scores)
            
            # Update x for next layer
            x = steered_output
        
        # Apply final layer norm if exists
        if hasattr(self.original_encoder, 'layer_norm'):
            final_output = self.original_encoder.layer_norm(x)
        else:
            final_output = x
            
        if return_gating:
            return final_output, gating_scores_list
        return final_output
    
    def tokenize_waveform(self, audio, return_gating=False):
        """
        Tokenize waveform with layer-wise steering.
        This method integrates with the original WhisperEncoder.tokenize_waveform logic.
        """
        # This is a simplified version - in practice, you'd need to integrate with the original
        # tokenize_waveform method from the WhisperEncoder class
        
        # For now, assume audio is already processed to mel spectrogram
        if return_gating:
            return self.forward(audio, return_gating=True)
        else:
            return self.forward(audio)
    
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


def create_layer_wise_steering_model(whisper_encoder_path: str, num_experts: int = 8):
    """
    Create a layer-wise steering Whisper encoder.
    
    Args:
        whisper_encoder_path: Path to the Whisper encoder
        num_experts: Number of experts for steering
        
    Returns:
        LayerWiseSteeringWhisperEncoder instance
    """
    # Load the original Whisper encoder
    from .tokenizer.whisper_Lv3.whisper import WhisperEncoder
    original_encoder = WhisperEncoder(whisper_encoder_path)
    
    # Create the layer-wise steering version
    steering_encoder = LayerWiseSteeringWhisperEncoder(
        original_encoder, 
        num_experts=num_experts
    )
    
    return steering_encoder


# Performance comparison and analysis
def analyze_layer_wise_vs_post_encoder():
    """
    Detailed analysis of layer-wise steering vs post-encoder steering.
    """
    print("=== Detailed Analysis: Layer-wise vs Post-encoder Steering ===\n")
    
    print("1. THEORETICAL ADVANTAGES OF LAYER-WISE STEERING:")
    print("   a) Fine-grained Adaptation:")
    print("      - Each layer can specialize in different audio characteristics")
    print("      - Early layers: phonetic features, acoustic patterns")
    print("      - Middle layers: word-level features, prosody")
    print("      - Late layers: semantic features, context")
    print()
    print("   b) Interpretability:")
    print("      - Can analyze which experts are used at each layer")
    print("      - Understand how different audio types are processed")
    print("      - Debug layer-specific issues")
    print()
    print("   c) Adaptive Processing:")
    print("      - Different audio types (speech, music, noise) can use different experts")
    print("      - Layer-specific scaling allows fine-tuning of steering strength")
    print()
    
    print("2. THEORETICAL DISADVANTAGES:")
    print("   a) Computational Cost:")
    print("      - 32x more routing operations (one per layer)")
    print("      - More parameters: 32 * num_experts * feature_dim steering vectors")
    print("      - 32 routers instead of 1")
    print()
    print("   b) Training Complexity:")
    print("      - More parameters to train")
    print("      - Potential overfitting with limited data")
    print("      - Harder to converge due to increased complexity")
    print()
    print("   c) Implementation Complexity:")
    print("      - Need to modify Whisper encoder internals")
    print("      - More complex debugging and analysis")
    print()
    
    print("3. WHEN LAYER-WISE STEERING WOULD PERFORM BETTER:")
    print("   a) Diverse Audio Types:")
    print("      - When training on varied audio (speech, music, environmental sounds)")
    print("      - Different layers can specialize in different audio characteristics")
    print()
    print("   b) Large Training Datasets:")
    print("      - With sufficient data, the increased capacity can be beneficial")
    print("      - Can learn layer-specific patterns effectively")
    print()
    print("   c) Specialized Tasks:")
    print("      - When you need fine-grained control over audio processing")
    print("      - Research scenarios requiring interpretability")
    print()
    
    print("4. WHEN POST-ENCODER STEERING WOULD PERFORM BETTER:")
    print("   a) Limited Training Data:")
    print("      - Simpler model is less prone to overfitting")
    print("      - Fewer parameters to train")
    print()
    print("   b) Computational Constraints:")
    print("      - When inference speed is critical")
    print("      - When memory usage is limited")
    print()
    print("   c) Simple Audio Tasks:")
    print("      - For straightforward ASR tasks")
    print("      - When layer-specific patterns aren't critical")
    print()
    
    print("5. RECOMMENDATION:")
    print("   - Start with post-encoder steering for initial experiments")
    print("   - Move to layer-wise steering if you need:")
    print("     * Better performance on diverse audio types")
    print("     * More interpretable results")
    print("     * Fine-grained control over audio processing")
    print("   - Consider hybrid approaches (steering at selected layers only)")


if __name__ == "__main__":
    analyze_layer_wise_vs_post_encoder() 