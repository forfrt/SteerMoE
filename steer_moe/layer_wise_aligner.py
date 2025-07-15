import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class LayerWiseSteerMoEAligner(nn.Module):
    """
    Layer-wise steering implementation that adds steering vectors to each layer
    of the Whisper encoder, with a router that adjusts steering weights.
    """
    def __init__(self, feature_dim: int, num_experts: int, num_layers: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.num_layers = num_layers
        
        # Steering vectors for each layer
        # Shape: (num_layers, num_experts, feature_dim)
        self.steering_vectors = nn.Parameter(
            torch.randn(num_layers, num_experts, feature_dim)
        )
        
        # Router for each layer
        # Shape: (num_layers, feature_dim, num_experts)
        self.routers = nn.ModuleList([
            nn.Linear(feature_dim, num_experts) 
            for _ in range(num_layers)
        ])
        
        # Optional: Layer-specific scaling factors
        self.layer_scales = nn.Parameter(torch.ones(num_layers))
        
    def forward(self, whisper_encoder, audio_input, return_gating=False):
        """
        Forward pass with layer-wise steering.
        
        Args:
            whisper_encoder: The Whisper encoder model
            audio_input: Input audio features
            return_gating: Whether to return gating scores for analysis
            
        Returns:
            Steered encoder output
        """
        # Store intermediate outputs and gating scores
        layer_outputs = []
        gating_scores_list = []
        
        # Get the encoder layers
        encoder_layers = whisper_encoder.layers
        
        # Process through each layer with steering
        x = audio_input
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
            
            # Store for analysis
            layer_outputs.append(steered_output)
            gating_scores_list.append(gating_scores)
            
            # Update x for next layer
            x = steered_output
        
        # Apply final layer norm if exists
        if hasattr(whisper_encoder, 'layer_norm'):
            final_output = whisper_encoder.layer_norm(x)
        else:
            final_output = x
            
        if return_gating:
            return final_output, gating_scores_list, layer_outputs
        return final_output
    
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
            'steering_strength': []
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


class WhisperEncoderWithSteering(nn.Module):
    """
    Wrapper that combines Whisper encoder with layer-wise steering.
    """
    def __init__(self, whisper_encoder, aligner: LayerWiseSteerMoEAligner):
        super().__init__()
        self.whisper_encoder = whisper_encoder
        self.aligner = aligner
        
    def forward(self, audio_input, return_gating=False):
        """
        Forward pass with layer-wise steering.
        """
        return self.aligner(self.whisper_encoder, audio_input, return_gating)
    
    def tokenize_waveform(self, audio, return_gating=False):
        """
        Tokenize waveform with layer-wise steering.
        """
        # First, get mel spectrogram (assuming this is done in the original tokenize_waveform)
        # This is a simplified version - in practice, you'd need to integrate with the original
        # WhisperEncoder.tokenize_waveform method
        
        # For now, assume audio_input is already processed mel features
        if return_gating:
            return self.forward(audio, return_gating=True)
        else:
            return self.forward(audio)


# Example usage and comparison
def compare_implementations():
    """
    Compare post-encoder steering vs layer-wise steering.
    """
    print("=== Implementation Comparison ===")
    
    # Current implementation (post-encoder)
    print("\n1. Current Implementation (Post-Encoder Steering):")
    print("   Audio → Whisper Encoder (32 layers) → SteerMoE Aligner → LLM")
    print("   Pros:")
    print("   - Simple implementation")
    print("   - Lower computational cost")
    print("   - Easier to debug")
    print("   Cons:")
    print("   - No fine-grained control over encoder layers")
    print("   - Steering applied only to final output")
    print("   - May miss layer-specific patterns")
    
    # Proposed implementation (layer-wise)
    print("\n2. Proposed Implementation (Layer-Wise Steering):")
    print("   Audio → Whisper Encoder (32 layers + steering at each layer) → LLM")
    print("   Pros:")
    print("   - Fine-grained control over each layer")
    print("   - Can capture layer-specific audio patterns")
    print("   - More interpretable (can analyze per-layer steering)")
    print("   - Potentially better adaptation to different audio characteristics")
    print("   Cons:")
    print("   - Higher computational cost (32x more routing)")
    print("   - More complex implementation")
    print("   - Harder to train (more parameters)")
    print("   - May overfit to training data")


if __name__ == "__main__":
    compare_implementations() 