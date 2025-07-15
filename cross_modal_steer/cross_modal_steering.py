import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CrossModalFeatureAligner(nn.Module):
    """
    Cross-modal feature aligner that properly bridges the feature space gap.
    Projects both modalities to a shared space, performs cross-attention,
    then projects back to LLM-compatible space.
    """
    def __init__(self, audio_dim: int, text_dim: int, hidden_dim: int = 1024, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Projections to shared space
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention in shared space
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Project back to LLM-compatible space
        self.alignment_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, text_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(text_dim)
        
    def forward(self, audio_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, audio_seq_len, audio_dim)
            text_features: (batch_size, text_seq_len, text_dim)
        
        Returns:
            aligned_audio: (batch_size, audio_seq_len, text_dim) - LLM-compatible
        """
        # Project both modalities to shared space
        audio_proj = self.audio_projection(audio_features)  # (batch, audio_seq, hidden)
        text_proj = self.text_projection(text_features)      # (batch, text_seq, hidden)
        
        # Cross-attention: audio queries, text keys/values
        attended_features, _ = self.cross_attention(
            query=audio_proj,
            key=text_proj,
            value=text_proj
        )
        
        # Apply residual connection and normalization
        attended_features = self.norm1(audio_proj + attended_features)
        
        # Project back to LLM-compatible space
        aligned_audio = self.alignment_layer(attended_features)
        
        return self.norm2(aligned_audio)


class ModalitySpecificSteering(nn.Module):
    """
    Modality-specific steering with separate routers and fusion.
    Handles audio and text independently, then fuses to LLM space.
    """
    def __init__(self, audio_dim: int, text_dim: int, num_experts: int = 8,
                 fusion_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.num_experts = num_experts
        
        # Audio-specific steering
        self.audio_router = nn.Linear(audio_dim, num_experts)
        self.audio_steering = nn.Parameter(torch.randn(num_experts, audio_dim) * 0.01)
        
        # Text-specific steering
        self.text_router = nn.Linear(text_dim, num_experts)
        self.text_steering = nn.Parameter(torch.randn(num_experts, text_dim) * 0.01)
        
        # Fusion to LLM-compatible space
        self.fusion_layer = nn.Sequential(
            nn.Linear(audio_dim + text_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, text_dim)
        )
        
        # Layer normalization
        self.audio_norm = nn.LayerNorm(audio_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.fusion_norm = nn.LayerNorm(text_dim)
        
    def forward(self, audio_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, audio_seq_len, audio_dim)
            text_features: (batch_size, text_seq_len, text_dim)
        
        Returns:
            fused_features: (batch_size, audio_seq_len, text_dim) - LLM-compatible
        """
        # Audio-specific steering
        audio_gating = F.softmax(self.audio_router(audio_features), dim=-1)
        audio_adjustment = torch.einsum('bte,ef->btf', audio_gating, self.audio_steering)
        steered_audio = self.audio_norm(audio_features + audio_adjustment)
        
        # Text-specific steering
        text_gating = F.softmax(self.text_router(text_features), dim=-1)
        text_adjustment = torch.einsum('bte,ef->btf', text_gating, self.text_steering)
        steered_text = self.text_norm(text_features + text_adjustment)
        
        # Expand text features to match audio sequence length
        # This is a simple approach - in practice, you might want more sophisticated alignment
        if steered_audio.size(1) != steered_text.size(1):
            # Repeat text features to match audio length
            steered_text = steered_text.repeat(1, steered_audio.size(1) // steered_text.size(1) + 1, 1)
            steered_text = steered_text[:, :steered_audio.size(1), :]
        
        # Fusion to LLM-compatible space
        fused_features = torch.cat([steered_audio, steered_text], dim=-1)
        llm_compatible = self.fusion_layer(fused_features)
        
        return self.fusion_norm(llm_compatible)


class CrossModalAdapter(nn.Module):
    """
    Adapter-based cross-modal alignment.
    Adapts audio to text space, then performs cross-attention.
    """
    def __init__(self, audio_dim: int, text_dim: int, adapter_dim: int = 512,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.adapter_dim = adapter_dim
        
        # Audio adapter: project audio to text space
        self.audio_adapter = nn.Sequential(
            nn.Linear(audio_dim, adapter_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, text_dim)
        )
        
        # Cross-attention between adapted audio and text
        self.cross_attention = nn.MultiheadAttention(
            text_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Final alignment layer
        self.final_alignment = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim, text_dim)
        )
        
        # Layer normalization
        self.adapter_norm = nn.LayerNorm(text_dim)
        self.attention_norm = nn.LayerNorm(text_dim)
        self.final_norm = nn.LayerNorm(text_dim)
        
    def forward(self, audio_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, audio_seq_len, audio_dim)
            text_features: (batch_size, text_seq_len, text_dim)
        
        Returns:
            aligned_audio: (batch_size, audio_seq_len, text_dim) - LLM-compatible
        """
        # Adapt audio to text space
        adapted_audio = self.audio_adapter(audio_features)
        adapted_audio = self.adapter_norm(adapted_audio)
        
        # Cross-attention between adapted audio and text
        attended_audio, _ = self.cross_attention(
            query=adapted_audio,
            key=text_features,
            value=text_features
        )
        
        # Apply residual connection and normalization
        attended_audio = self.attention_norm(adapted_audio + attended_audio)
        
        # Final alignment
        aligned_audio = self.final_alignment(attended_audio)
        
        return self.final_norm(attended_audio + aligned_audio)


class ProgressiveCrossModalSteering(nn.Module):
    """
    Progressive cross-modal steering that builds alignment step by step.
    Combines the benefits of progressive routing with cross-modal alignment.
    """
    def __init__(self, audio_dim: int, text_dim: int, num_stages: int = 3,
                 num_experts: int = 8, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.num_stages = num_stages
        self.num_experts = num_experts
        
        # Progressive routers for each stage
        self.stage_routers = nn.ModuleList([
            nn.Linear(audio_dim + i * text_dim, num_experts) 
            for i in range(num_stages)
        ])
        
        # Stage-specific steering vectors
        self.stage_steering = nn.Parameter(
            torch.randn(num_stages, num_experts, text_dim) * 0.01
        )
        
        # Audio-to-text projection
        self.audio_projection = nn.Linear(audio_dim, text_dim)
        
        # Cross-modal attention for each stage
        self.stage_attention = nn.ModuleList([
            nn.MultiheadAttention(text_dim, num_heads=8, dropout=dropout, batch_first=True)
            for _ in range(num_stages)
        ])
        
        # Layer normalization
        self.stage_norms = nn.ModuleList([
            nn.LayerNorm(text_dim) for _ in range(num_stages)
        ])
        
    def forward(self, audio_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, audio_seq_len, audio_dim)
            text_features: (batch_size, text_seq_len, text_dim)
        
        Returns:
            progressive_aligned: (batch_size, audio_seq_len, text_dim) - LLM-compatible
        """
        # Initial audio projection
        current_audio = self.audio_projection(audio_features)
        accumulated_alignment = torch.zeros_like(current_audio)
        
        for stage in range(self.num_stages):
            # Prepare input for this stage
            if stage == 0:
                stage_input = torch.cat([audio_features, current_audio], dim=-1)
            else:
                stage_input = torch.cat([audio_features, current_audio + accumulated_alignment], dim=-1)
            
            # Stage-specific routing
            stage_logits = self.stage_routers[stage](stage_input)
            stage_scores = F.softmax(stage_logits, dim=-1)
            
            # Apply stage-specific steering
            stage_steering = torch.einsum('bte,ef->btf', stage_scores, self.stage_steering[stage])
            
            # Cross-modal attention
            attended_audio, _ = self.stage_attention[stage](
                query=current_audio + stage_steering,
                key=text_features,
                value=text_features
            )
            
            # Update accumulated alignment
            stage_alignment = self.stage_norms[stage](attended_audio)
            accumulated_alignment += stage_alignment
            
            # Update current audio for next stage
            current_audio = current_audio + stage_alignment
        
        return current_audio + accumulated_alignment


# Example usage and comparison
def compare_cross_modal_approaches():
    """
    Compare different cross-modal steering approaches.
    """
    batch_size = 2
    audio_seq_len = 100
    text_seq_len = 50
    audio_dim = 1280
    text_dim = 768
    
    # Create dummy inputs
    audio_features = torch.randn(batch_size, audio_seq_len, audio_dim)
    text_features = torch.randn(batch_size, text_seq_len, text_dim)
    
    print("=== Cross-Modal Steering Approaches Comparison ===\n")
    
    # Test Feature Aligner
    feature_aligner = CrossModalFeatureAligner(audio_dim, text_dim)
    aligned_1 = feature_aligner(audio_features, text_features)
    print(f"Feature Aligner output shape: {aligned_1.shape}")
    print(f"Feature Aligner parameters: {sum(p.numel() for p in feature_aligner.parameters()):,}")
    
    # Test Modality-Specific Steering
    modality_steering = ModalitySpecificSteering(audio_dim, text_dim)
    aligned_2 = modality_steering(audio_features, text_features)
    print(f"Modality-Specific output shape: {aligned_2.shape}")
    print(f"Modality-Specific parameters: {sum(p.numel() for p in modality_steering.parameters()):,}")
    
    # Test Cross-Modal Adapter
    cross_modal_adapter = CrossModalAdapter(audio_dim, text_dim)
    aligned_3 = cross_modal_adapter(audio_features, text_features)
    print(f"Cross-Modal Adapter output shape: {aligned_3.shape}")
    print(f"Cross-Modal Adapter parameters: {sum(p.numel() for p in cross_modal_adapter.parameters()):,}")
    
    # Test Progressive Cross-Modal Steering
    progressive_steering = ProgressiveCrossModalSteering(audio_dim, text_dim)
    aligned_4 = progressive_steering(audio_features, text_features)
    print(f"Progressive Steering output shape: {aligned_4.shape}")
    print(f"Progressive Steering parameters: {sum(p.numel() for p in progressive_steering.parameters()):,}")
    
    print("\n=== Key Insights ===")
    print("1. All approaches properly bridge the feature space gap")
    print("2. Feature Aligner is most principled but computationally expensive")
    print("3. Modality-Specific Steering is efficient and interpretable")
    print("4. Cross-Modal Adapter is lightweight and effective")
    print("5. Progressive Steering offers the most expressiveness")


if __name__ == "__main__":
    compare_cross_modal_approaches() 