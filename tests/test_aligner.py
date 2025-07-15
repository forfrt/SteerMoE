import torch
from steer_moe.aligner import SteerMoEAligner
from steer_moe.utils import load_balancing_loss

def test_aligner_output_and_lb_loss():
    batch, seq_len, feature_dim, num_experts = 2, 5, 16, 4
    aligner = SteerMoEAligner(feature_dim, num_experts)
    x = torch.randn(batch, seq_len, feature_dim)
    h_aligned, gating_scores = aligner(x, return_gating=True)
    assert h_aligned.shape == (batch, seq_len, feature_dim)
    assert gating_scores.shape == (batch, seq_len, num_experts)
    lb_loss = load_balancing_loss(gating_scores)
    assert lb_loss.item() >= 0
    print("Test passed: output shape and load balancing loss.")

if __name__ == "__main__":
    test_aligner_output_and_lb_loss()
