import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class SharedRouterLayerWiseSteeringWhisperEncoder(nn.Module):
    """
    Corrected implementation: Shared router with layer-specific weight matrices.
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
        
        # SHARED router - same for all layers
        self.shared_router = nn.Linear(self.feature_dim, num_experts)
        
        # Layer-specific weight matrices to transform shared router output
        # Shape: (num_layers, num_experts, num_experts)
        # Initialize as identity matrices (no transformation initially)
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
        KEY DIFFERENCE: Same router output, different weight matrices per layer.
        """
        gating_scores_list = []
        encoder_layers = self.original_encoder.layers
        
        x = mel_spec
        for layer_idx, layer in enumerate(encoder_layers):
            # 1. Apply original layer
            layer_output = layer(x)
            
            # 2. Get SHARED router output (same for all layers)
            shared_gating_logits = self.shared_router(layer_output)  # (batch, seq_len, num_experts)
            
            # 3. Apply LAYER-SPECIFIC weight matrix to transform the shared output
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


class MultipleRoutersLayerWiseSteeringWhisperEncoder(nn.Module):
    """
    Original implementation: Multiple independent routers (one per layer).
    This approach uses separate routers for each layer.
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
        
        # MULTIPLE routers - one per layer
        self.routers = nn.ModuleList([
            nn.Linear(self.feature_dim, num_experts) 
            for _ in range(self.num_layers)
        ])
        
        # Layer-specific scaling factors
        self.layer_scales = nn.Parameter(torch.ones(self.num_layers) * steering_scale)
        
        # Freeze the original encoder
        for param in self.original_encoder.parameters():
            param.requires_grad = False
            
    def forward(self, mel_spec, return_gating=False):
        """
        Forward pass with multiple independent routers.
        KEY DIFFERENCE: Different router per layer, no parameter sharing.
        """
        gating_scores_list = []
        encoder_layers = self.original_encoder.layers
        
        x = mel_spec
        for layer_idx, layer in enumerate(encoder_layers):
            # 1. Apply original layer
            layer_output = layer(x)
            
            # 2. Get LAYER-SPECIFIC router output (different router per layer)
            router = self.routers[layer_idx]  # Different router for each layer
            layer_gating_logits = router(layer_output)  # (batch, seq_len, num_experts)
            
            # 3. Apply softmax
            gating_scores = F.softmax(layer_gating_logits, dim=-1)
            
            # 4. Get steering vectors and apply
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


# Detailed comparison of the three approaches
def compare_router_approaches_detailed():
    """
    Detailed comparison of the three router approaches.
    """
    print("=== Detailed Router Approach Comparison ===\n")
    
    feature_dim = 1280
    num_experts = 8
    num_layers = 32
    
    print("1. MULTIPLE ROUTERS (Original Implementation):")
    print("   - 32 separate routers (one per layer)")
    print("   - Each router: Linear(feature_dim, num_experts)")
    print("   - Parameters: 32 * feature_dim * num_experts")
    print("   - Computation: Each layer uses its own independent router")
    print("   - Pros: Complete independence per layer")
    print("   - Cons: High parameter count, no parameter sharing")
    print()
    
    print("2. SINGLE ROUTER (Your Proposed Approach):")
    print("   - 1 router: Linear(feature_dim, num_experts * num_layers)")
    print("   - Parameters: feature_dim * num_experts * num_layers")
    print("   - Computation: Extract layer-specific scores from single router output")
    print("   - Pros: Efficient, parameter sharing")
    print("   - Cons: Less flexibility, all layers share same router")
    print()
    
    print("3. SHARED ROUTER WITH LAYER WEIGHTS (Alternative):")
    print("   - 1 shared router: Linear(feature_dim, num_experts)")
    print("   - Layer-specific weights: (num_layers, num_experts, num_experts)")
    print("   - Parameters: feature_dim * num_experts + num_layers * num_experts^2")
    print("   - Computation: Same router output, different weight matrices per layer")
    print("   - Pros: Parameter sharing + layer-specific adaptation")
    print("   - Cons: More complex, potential redundancy")
    print()
    
    # Parameter count comparison
    params_multiple = 32 * feature_dim * num_experts
    params_single = feature_dim * num_experts * num_layers
    params_shared = feature_dim * num_experts + num_layers * num_experts * num_experts
    
    print("PARAMETER COUNT COMPARISON:")
    print(f"Multiple routers: {params_multiple:,} parameters")
    print(f"Single router: {params_single:,} parameters")
    print(f"Shared router: {params_shared:,} parameters")
    print()
    
    print("KEY DIFFERENCES IN COMPUTATION:")
    print("Multiple Routers:")
    print("  for layer in layers:")
    print("    gating = layer_router[layer_idx](layer_output)")
    print()
    print("Single Router:")
    print("  router_output = single_router(layer_output)")
    print("  gating = router_output[:, :, layer_idx*num_experts:(layer_idx+1)*num_experts]")
    print()
    print("Shared Router with Weights:")
    print("  shared_output = shared_router(layer_output)")
    print("  gating = torch.einsum('bte,ef->btf', shared_output, layer_weights[layer_idx])")
    print()
    
    print("RECOMMENDATION:")
    print("- Use single router approach for efficiency (your suggestion)")
    print("- Use shared router with weights if you need more flexibility")
    print("- Use multiple routers only if you need completely independent routing")


if __name__ == "__main__":
    compare_router_approaches_detailed() 