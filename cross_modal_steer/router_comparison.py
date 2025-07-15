import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class CurrentRouter(nn.Module):
    """
    Current SteerMoE router implementation for comparison.
    """
    def __init__(self, feature_dim: int, num_experts: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        
        # Simple router and steering vectors
        self.router = nn.Linear(feature_dim, num_experts)
        self.steering_vectors = nn.Parameter(torch.randn(num_experts, feature_dim) * 0.01)
        
    def forward(self, x: torch.Tensor, return_gating: bool = False):
        # Single routing decision
        gating_logits = self.router(x)  # (batch, seq_len, num_experts)
        gating_scores = F.softmax(gating_logits, dim=-1)
        
        # Apply steering
        steering_adjustment = torch.einsum('bte,ef->btf', gating_scores, self.steering_vectors)
        steered_output = x + steering_adjustment
        
        if return_gating:
            return steered_output, gating_scores
        return steered_output


class HierarchicalRouter(nn.Module):
    """
    Hierarchical router with global and local routing decisions.
    """
    def __init__(self, feature_dim: int, num_experts: int = 8, num_global_experts: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.num_global_experts = num_global_experts
        
        # Global router for high-level decisions
        self.global_router = nn.Linear(feature_dim, num_global_experts)
        self.global_steering = nn.Parameter(torch.randn(num_global_experts, feature_dim) * 0.01)
        
        # Local routers for fine-grained decisions
        self.local_routers = nn.ModuleList([
            nn.Linear(feature_dim, num_experts) for _ in range(num_global_experts)
        ])
        self.local_steering = nn.Parameter(torch.randn(num_global_experts, num_experts, feature_dim) * 0.01)
        
        # Layer normalization
        self.global_norm = nn.LayerNorm(feature_dim)
        self.local_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x: torch.Tensor, return_gating: bool = False):
        # Global routing
        global_logits = self.global_router(x)  # (batch, seq_len, num_global_experts)
        global_scores = F.softmax(global_logits, dim=-1)
        
        # Apply global steering
        global_steering = torch.einsum('bte,ef->btf', global_scores, self.global_steering)
        global_output = self.global_norm(x + global_steering)
        
        # Local routing per global expert
        local_outputs = []
        for i in range(self.num_global_experts):
            local_logits = self.local_routers[i](global_output)  # (batch, seq_len, num_experts)
            local_scores = F.softmax(local_logits, dim=-1)
            local_steering = torch.einsum('bte,ef->btf', local_scores, self.local_steering[i])
            local_outputs.append(local_steering)
        
        # Combine local steering with global weights
        local_steering = torch.stack(local_outputs, dim=-1)  # (batch, seq_len, feature_dim, num_global_experts)
        local_steering = torch.einsum('btfe,bte->btf', local_steering, global_scores)
        
        # Final output
        final_output = self.local_norm(global_output + local_steering)
        
        if return_gating:
            return final_output, (global_scores, local_outputs)
        return final_output


class ProgressiveRouter(nn.Module):
    """
    Progressive router that builds routing decisions step by step.
    """
    def __init__(self, feature_dim: int, num_experts: int = 8, num_stages: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.num_stages = num_stages
        
        # Progressive routers that build on previous decisions
        self.stage_routers = nn.ModuleList([
            nn.Linear(feature_dim + i * num_experts, num_experts) 
            for i in range(num_stages)
        ])
        
        # Stage-specific steering vectors
        self.stage_steering = nn.Parameter(
            torch.randn(num_stages, num_experts, feature_dim) * 0.01
        )
        
        # Layer normalization for each stage
        self.stage_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_stages)
        ])
        
    def forward(self, x: torch.Tensor, return_gating: bool = False):
        current_input = x
        accumulated_steering = torch.zeros_like(x)
        stage_gating_scores = []
        
        for stage in range(self.num_stages):
            # Route based on current input (includes previous steering info)
            stage_logits = self.stage_routers[stage](current_input)
            stage_scores = F.softmax(stage_logits, dim=-1)
            stage_gating_scores.append(stage_scores)
            
            # Apply stage-specific steering
            stage_steering = torch.einsum('bte,ef->btf', stage_scores, self.stage_steering[stage])
            accumulated_steering += stage_steering
            
            # Update input for next stage
            current_input = torch.cat([x, accumulated_steering], dim=-1)
        
        # Apply final normalization
        final_output = self.stage_norms[-1](x + accumulated_steering)
        
        if return_gating:
            return final_output, stage_gating_scores
        return final_output


class AttentionBasedRouter(nn.Module):
    """
    Attention-based router using expert embeddings and multihead attention.
    """
    def __init__(self, feature_dim: int, num_experts: int = 8, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.num_heads = num_heads
        
        # Expert embeddings
        self.expert_embeddings = nn.Parameter(torch.randn(num_experts, feature_dim) * 0.01)
        
        # Multihead attention for routing
        self.attention = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )
        
        # Steering vectors
        self.steering_vectors = nn.Parameter(torch.randn(num_experts, feature_dim) * 0.01)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
    def forward(self, x: torch.Tensor, return_gating: bool = False):
        batch_size, seq_len, _ = x.shape
        
        # Expand expert embeddings to match sequence length
        expert_embeddings = self.expert_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Attention between input and expert embeddings
        attended_experts, attention_weights = self.attention(
            query=x,
            key=expert_embeddings,
            value=expert_embeddings
        )
        
        # Apply residual connection and normalization
        attended_experts = self.norm1(x + attended_experts)
        
        # Compute gating scores from attention weights
        # attention_weights: (batch_size, seq_len, num_experts)
        gating_scores = F.softmax(attention_weights, dim=-1)
        
        # Apply steering
        steering_adjustment = torch.einsum('bte,ef->btf', gating_scores, self.steering_vectors)
        final_output = self.norm2(attended_experts + steering_adjustment)
        
        if return_gating:
            return final_output, gating_scores
        return final_output


def compare_router_implementations():
    """
    Comprehensive comparison of router implementations.
    """
    print("=== Router Implementation Comparison ===\n")
    
    # Test parameters
    batch_size = 2
    seq_len = 100
    feature_dim = 1280
    num_experts = 8
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, feature_dim)
    
    # Test Current Router
    current_router = CurrentRouter(feature_dim, num_experts)
    current_output, current_gating = current_router(x, return_gating=True)
    current_params = sum(p.numel() for p in current_router.parameters())
    
    print("1. CURRENT ROUTER (SteerMoE Aligner):")
    print(f"   Output shape: {current_output.shape}")
    print(f"   Parameters: {current_params:,}")
    print(f"   Gating scores shape: {current_gating.shape}")
    print("   Pros: Simple, efficient, easy to train")
    print("   Cons: Single routing decision, limited expressiveness")
    print()
    
    # Test Hierarchical Router
    hierarchical_router = HierarchicalRouter(feature_dim, num_experts, num_global_experts=4)
    hierarchical_output, hierarchical_gating = hierarchical_router(x, return_gating=True)
    hierarchical_params = sum(p.numel() for p in hierarchical_router.parameters())
    
    print("2. HIERARCHICAL ROUTER:")
    print(f"   Output shape: {hierarchical_output.shape}")
    print(f"   Parameters: {hierarchical_params:,}")
    print(f"   Global gating shape: {hierarchical_gating[0].shape}")
    print(f"   Local gating shapes: {[g.shape for g in hierarchical_gating[1]]}")
    print("   Pros: Hierarchical decisions, better modeling of complex patterns")
    print("   Cons: Higher computational cost, more complex training")
    print()
    
    # Test Progressive Router
    progressive_router = ProgressiveRouter(feature_dim, num_experts, num_stages=3)
    progressive_output, progressive_gating = progressive_router(x, return_gating=True)
    progressive_params = sum(p.numel() for p in progressive_router.parameters())
    
    print("3. PROGRESSIVE ROUTER:")
    print(f"   Output shape: {progressive_output.shape}")
    print(f"   Parameters: {progressive_params:,}")
    print(f"   Stage gating shapes: {[g.shape for g in progressive_gating]}")
    print("   Pros: Progressive refinement, can correct mistakes")
    print("   Cons: Sequential computation, higher memory usage")
    print()
    
    # Test Attention-Based Router
    attention_router = AttentionBasedRouter(feature_dim, num_experts, num_heads=8)
    attention_output, attention_gating = attention_router(x, return_gating=True)
    attention_params = sum(p.numel() for p in attention_router.parameters())
    
    print("4. ATTENTION-BASED ROUTER:")
    print(f"   Output shape: {attention_output.shape}")
    print(f"   Parameters: {attention_params:,}")
    print(f"   Gating scores shape: {attention_gating.shape}")
    print("   Pros: Attention-based routing, interpretable")
    print("   Cons: Higher computational cost, potential overfitting")
    print()
    
    # Parameter efficiency comparison
    print("=== PARAMETER EFFICIENCY COMPARISON ===")
    print(f"Current Router: {current_params:,} parameters")
    print(f"Hierarchical Router: {hierarchical_params:,} parameters")
    print(f"Progressive Router: {progressive_params:,} parameters")
    print(f"Attention Router: {attention_params:,} parameters")
    print()
    
    # Computational complexity comparison
    print("=== COMPUTATIONAL COMPLEXITY ===")
    print("Current Router: O(seq_len * feature_dim * num_experts)")
    print("Hierarchical Router: O(seq_len * feature_dim * (num_global + num_local))")
    print("Progressive Router: O(seq_len * feature_dim * num_experts * num_stages)")
    print("Attention Router: O(seq_len^2 * feature_dim + seq_len * feature_dim * num_experts)")
    print()
    
    # Use case recommendations
    print("=== USE CASE RECOMMENDATIONS ===")
    print("Current Router:")
    print("  - Use for: Simple alignment tasks, efficiency-focused applications")
    print("  - Avoid for: Complex multimodal alignment, hierarchical patterns")
    print()
    print("Hierarchical Router:")
    print("  - Use for: Complex alignment patterns, interpretable routing")
    print("  - Avoid for: Resource-constrained environments, simple tasks")
    print()
    print("Progressive Router:")
    print("  - Use for: Complex alignment tasks, error correction")
    print("  - Avoid for: Real-time applications, simple tasks")
    print()
    print("Attention Router:")
    print("  - Use for: Interpretable routing, attention-based alignment")
    print("  - Avoid for: Long sequences, resource-constrained environments")
    print()
    
    # Performance metrics (simulated)
    print("=== SIMULATED PERFORMANCE METRICS ===")
    print("Metric: Average routing confidence (higher is better)")
    print(f"Current Router: {current_gating.max(dim=-1)[0].mean().item():.3f}")
    print(f"Hierarchical Router: {hierarchical_gating[0].max(dim=-1)[0].mean().item():.3f}")
    print(f"Progressive Router: {progressive_gating[-1].max(dim=-1)[0].mean().item():.3f}")
    print(f"Attention Router: {attention_gating.max(dim=-1)[0].mean().item():.3f}")
    print()
    
    print("=== FINAL RECOMMENDATIONS ===")
    print("1. Start with Current Router for baseline performance")
    print("2. Use Progressive Router for complex alignment tasks")
    print("3. Use Hierarchical Router when interpretability is important")
    print("4. Use Attention Router for attention-based alignment patterns")
    print("5. Consider computational constraints when choosing")


def analyze_routing_patterns():
    """
    Analyze the routing patterns of different router implementations.
    """
    print("\n=== ROUTING PATTERN ANALYSIS ===\n")
    
    batch_size = 1
    seq_len = 10
    feature_dim = 1280
    num_experts = 8
    
    # Create input with clear patterns
    x = torch.randn(batch_size, seq_len, feature_dim)
    
    routers = {
        "Current": CurrentRouter(feature_dim, num_experts),
        "Hierarchical": HierarchicalRouter(feature_dim, num_experts, 4),
        "Progressive": ProgressiveRouter(feature_dim, num_experts, 3),
        "Attention": AttentionBasedRouter(feature_dim, num_experts, 8)
    }
    
    for name, router in routers.items():
        print(f"{name} Router Analysis:")
        
        if name == "Current":
            _, gating = router(x, return_gating=True)
            print(f"  Gating scores shape: {gating.shape}")
            print(f"  Expert utilization: {gating.mean(dim=(0,1)).tolist()}")
            
        elif name == "Hierarchical":
            _, (global_gating, local_gating) = router(x, return_gating=True)
            print(f"  Global gating shape: {global_gating.shape}")
            print(f"  Local gating shapes: {[g.shape for g in local_gating]}")
            print(f"  Global expert utilization: {global_gating.mean(dim=(0,1)).tolist()}")
            
        elif name == "Progressive":
            _, stage_gating = router(x, return_gating=True)
            print(f"  Stage gating shapes: {[g.shape for g in stage_gating]}")
            for i, stage_g in enumerate(stage_gating):
                print(f"  Stage {i} expert utilization: {stage_g.mean(dim=(0,1)).tolist()}")
                
        elif name == "Attention":
            _, gating = router(x, return_gating=True)
            print(f"  Gating scores shape: {gating.shape}")
            print(f"  Expert utilization: {gating.mean(dim=(0,1)).tolist()}")
        
        print()


if __name__ == "__main__":
    compare_router_implementations()
    analyze_routing_patterns() 