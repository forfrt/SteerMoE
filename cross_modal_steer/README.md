# Cross-Modal Steering Directory

This folder contains exploratory implementations and analysis of advanced routing mechanisms and cross-modal alignment strategies for multimodal mixture-of-experts models.

## 📁 Purpose

This directory explores alternative architectures for:
- **Router mechanisms**: Different ways to route tokens to experts
- **Cross-modal alignment**: Strategies for bridging audio and text modalities
- **Architectural trade-offs**: Analysis of complexity vs. expressiveness

⚠️ **Note**: These are **research explorations** and may not be used in the main training pipeline. They serve as reference implementations for future improvements.

## 📝 Files

### `router_comparison.py`

**Purpose**: Implements and compares multiple router architectures for expert routing in mixture-of-experts models.

**Key Classes**:

1. **CurrentRouter** - Simple baseline router
   - Single routing decision per token
   - Uses `nn.Linear` to compute gating logits for each expert
   - Applies steering vectors via einsum operation
   - Returns steered output as residual addition to input

2. **HierarchicalRouter** - Two-level hierarchical routing
   - Global router makes high-level decisions (4 global experts by default)
   - Local routers (one per global expert) make fine-grained decisions
   - Combines global and local steering with layer normalization
   - More expressive but computationally heavier

3. **ProgressiveRouter** - Multi-stage progressive routing
   - Routes through multiple stages sequentially
   - Each stage refines routing decisions from previous stage
   - Includes layer normalization between stages
   - Tracks gating scores at each stage

4. **AttentionBasedRouter** - Attention-driven routing
   - Uses multi-head attention to compute routing weights
   - Attention heads act as different routing perspectives
   - Combines attention outputs for final routing decision
   - More sophisticated routing mechanism

**Main Functions**:
- `compare_router_implementations()` - Creates dummy input and tests all routers, comparing output shapes and parameter counts
- `analyze_routing_patterns()` - Analyzes expert utilization and routing patterns for each router type

**Usage**:
```python
from cross_modal_steer.router_comparison import compare_router_implementations

# Compare all router types
compare_router_implementations()
```

---

### `cross_modal_steering.py`

**Purpose**: Implements cross-modal feature alignment and steering approaches for bridging audio and text modalities.

**Key Classes**:

1. **CrossModalFeatureAligner** - Principled feature space alignment
   - Projects audio and text to shared hidden space
   - Performs cross-attention (audio queries, text keys/values)
   - Projects back to LLM-compatible text dimension
   - Uses residual connections and layer normalization
   - Most theoretically sound but computationally expensive

2. **ModalitySpecificSteering** - Independent modality handling with fusion
   - Separate routers for audio and text modalities
   - Each modality has its own steering vectors
   - Fuses modality-specific outputs in shared fusion space
   - Projects fused output to LLM space
   - Efficient and interpretable

3. **CrossModalAdapter** - Lightweight adapter approach
   - Simple sequential adapter for audio to text space
   - Cross-attention between adapted audio and text
   - Minimal parameter overhead
   - Effective for lightweight integration

4. **ProgressiveCrossModalSteering** - Multi-stage cross-modal alignment
   - Progressive refinement through multiple stages
   - Each stage performs cross-modal attention
   - Builds up alignment gradually
   - Most expressive approach

**Main Functions**:
- `compare_cross_modal_approaches()` - Tests all four approaches with dummy inputs, compares output shapes, parameter counts, and provides insights

**Usage**:
```python
from cross_modal_steer.cross_modal_steering import compare_cross_modal_approaches

# Compare all cross-modal approaches
compare_cross_modal_approaches()
```

---

### `router_analysis_and_cross_modal_steering.md`

**Purpose**: Documentation and analysis of router implementations and cross-modal steering strategies.

**Content Sections**:

1. **Router Implementation Comparison**
   - Analyzes Current Implementation (simple, efficient, limited expressiveness)
   - Hierarchical Router (better for complex patterns, higher cost)
   - Progressive Router (multi-stage refinement, good balance)
   - Attention-Based Router (sophisticated routing via attention)

2. **Cross-Modal Alignment Approaches**
   - Feature Space Alignment (principled but expensive)
   - Modality-Specific Steering (efficient and interpretable)
   - Cross-Modal Adapter (lightweight)
   - Progressive Cross-Modal Steering (most expressive)

3. **Recommendations**
   - For layer-wise steering: Start with Progressive Router
   - For cross-modal alignment: Use Feature Space Alignment
   - Key insight: Frozen LLM decoder constraint is critical
   - Cross-modal attention requires proper feature space bridging

**Key Insights**:
- Progressive routers offer better expressiveness for complex alignment patterns
- Feature space alignment is crucial for multimodal integration
- Direct cross-attention without alignment is problematic
- Trade-offs between computational cost and expressiveness

## 🔬 Research Directions

These implementations explore:

1. **Router Architectures**:
   - Single-stage vs. multi-stage routing
   - Flat vs. hierarchical expert organization
   - Attention-based routing mechanisms

2. **Cross-Modal Alignment**:
   - Feature space projection strategies
   - Modality-specific vs. shared routing
   - Progressive refinement approaches

3. **Trade-offs**:
   - Parameter efficiency vs. expressiveness
   - Computational cost vs. alignment quality
   - Simplicity vs. flexibility

## 📊 Comparison Summary

| Approach | Parameters | Complexity | Expressiveness | Use Case |
|----------|-----------|------------|----------------|----------|
| **Current Router** | Low | Low | Medium | Production baseline |
| **Hierarchical Router** | Medium | Medium | High | Complex patterns |
| **Progressive Router** | Medium | Medium | High | Balanced approach |
| **Attention Router** | High | High | Very High | Research exploration |
| **Feature Aligner** | High | High | Very High | Principled alignment |
| **Modality-Specific** | Medium | Medium | High | Efficient multimodal |
| **Cross-Modal Adapter** | Low | Low | Medium | Lightweight integration |
| **Progressive Cross-Modal** | High | High | Very High | Maximum expressiveness |

## 🔗 Related Documentation

- Current implementations: [`steer_moe/`](../steer_moe/)
- Architecture analysis: [`docs/SteerMoE_Architecture_Analysis.md`](../docs/SteerMoE_Architecture_Analysis.md)
- Training scripts: [`scripts/`](../scripts/)
