"""
 * author Ruitao Feng
 * created on 16-07-2025
 * github: https://github.com/forfrt
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class SteerMoEAligner(nn.Module):
    def __init__(self, feature_dim, num_experts):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        # Bank of learnable steering vectors (experts)
        self.steering_vectors = nn.Parameter(torch.randn(num_experts, feature_dim))
        # Lightweight router: a single linear layer
        self.router = nn.Linear(feature_dim, num_experts)

    def forward(self, h_audio, return_gating=False):
        # h_audio: (batch, seq_len, feature_dim)
        batch, seq_len, dim = h_audio.shape
        # 1. Compute gating scores for each token
        gating_logits = self.router(h_audio)  # (batch, seq_len, num_experts)
        gating_scores = F.softmax(gating_logits, dim=-1)  # (batch, seq_len, num_experts)
        # 2. Compute adjustment vector as weighted sum of steering vectors
        # (batch, seq_len, feature_dim)
        delta_h = torch.einsum('bte,ef->btf', gating_scores, self.steering_vectors)
        # 3. Add adjustment to original audio representation
        h_aligned = h_audio + delta_h
        if return_gating:
            return h_aligned, gating_scores
        return h_aligned
