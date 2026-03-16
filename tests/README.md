# Tests Directory

This folder contains test files for validating SteerMoE components.

## 📁 Files

### Test Scripts

**`test_aligner.py`**
- Tests the `SteerMoEAligner` component and load balancing loss calculation
- Single comprehensive test function: `test_aligner_output_and_lb_loss()`

**What it tests:**
1. Creates a SteerMoEAligner instance with feature dimension of 16 and 4 experts
2. Generates random input tensor with shape (batch=2, seq_len=5, feature_dim=16)
3. Passes input through the aligner with `return_gating=True` to get both aligned hidden states and gating scores
4. Validates output shapes:
   - `h_aligned` should be (2, 5, 16) - same as input
   - `gating_scores` should be (2, 5, 4) - batch × sequence × num_experts
5. Computes load balancing loss from gating scores
6. Asserts that load balancing loss is non-negative (>= 0)

**Dependencies:**
- `torch` - PyTorch tensors
- `steer_moe.aligner.SteerMoEAligner` - The aligner component being tested
- `steer_moe.utils.load_balancing_loss` - Utility function for computing load balancing loss

**Usage:**
```bash
python tests/test_aligner.py
```

**Expected Output:**
```
Test passed: Aligner output and load balancing loss are correct!
```

### Audio Files

**`output_audio.wav`**
- Test output audio file (likely generated during testing)

**`小鹏汽车2024年四季报业绩交流会.mp3`**
- Chinese audio file (appears to be a recording of a quarterly earnings call)
- Used for testing Chinese ASR functionality

## 🔧 Running Tests

To run all tests:
```bash
# Run specific test
python tests/test_aligner.py

# Run with pytest (if available)
pytest tests/
```

## 📝 Adding New Tests

When adding new test files, follow this structure:

```python
import torch
from steer_moe.your_module import YourComponent

def test_your_component():
    """Test description"""
    # Setup
    component = YourComponent(...)
    input_data = torch.randn(...)

    # Execute
    output = component(input_data)

    # Validate
    assert output.shape == expected_shape
    assert output.dtype == expected_dtype

    print("Test passed!")

if __name__ == "__main__":
    test_your_component()
```

## 🔗 Related Documentation

- Core implementations: [`steer_moe/`](../steer_moe/)
- Training scripts: [`scripts/`](../scripts/)
