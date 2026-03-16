# Documentation Directory

This folder contains technical documentation and architectural analysis for the SteerMoE project.

## 📁 Files

### `SteerMoE_Architecture_Analysis.md`

**Purpose**: A comprehensive technical analysis document of the SteerMoE architecture (created January 2026).

**Key Sections Covered**:

1. **Introduction**
   - Introduces SteerMoE as an efficient layer-wise steering mechanism
   - For adapting frozen audio encoders (like Whisper) for cross-modal alignment with LLMs

2. **Understanding SteerMoE Architecture**
   - Details the three main trainable components:
     - `steering_vectors`: Learned direction vectors in representation space (shape: num_layers × num_experts × feature_dim)
     - `router`: Linear layer for input-dependent expert selection
     - `layer_scales`: Per-layer magnitude control parameters
   - Explains the core steering formula: `x = layer_output + layer_scale × Σ(gating_score × steering_vector)`

3. **SteerMoE vs Residual Connections**
   - Compares SteerMoE with classical residual connections
   - Analyzes how SteerMoE differs through its use of learned steering vectors and dynamic routing rather than simple skip connections

4. **Is SteerMoE a Multi-Modal Residual Connection?**
   - Discusses whether SteerMoE qualifies as a residual connection variant
   - Concludes it's better understood as a "Routed Residual Adapter"

5. **Comparison with mHC (Manifold Hyper-Connections)**
   - Analyzes DeepSeek's manifold Hyper-Connections approach and how it relates to SteerMoE's design
   - Discusses manifold constraints and stability considerations

6. **Normalization and Manifold Constraint Methods**
   - Explores various normalization techniques and manifold constraint approaches
   - That could improve SteerMoE's stability and performance

7. **Practical Recommendations**
   - Provides actionable guidance for implementation and optimization

8. **Conclusion**
   - Summarizes SteerMoE's identity as a routed residual adapter
   - Discusses connections to mHC principles

**References**:
- Includes citations to ResNet, Mixture of Experts, LoRA, Adapters, and DeepSeek's Hyper-Connections work

**Usage**:
```bash
# Read the documentation
cat docs/SteerMoE_Architecture_Analysis.md

# Or open in a Markdown viewer
```

## 📚 Documentation Topics

### Architectural Design

The document explores in depth:
- **Steering Mechanism**: How steering vectors modify encoder outputs
- **Routing Strategy**: How the MoE router selects experts
- **Parameter Efficiency**: Why SteerMoE needs only ~1.8M trainable parameters
- **Layer Scaling**: The role of per-layer magnitude control

### Theoretical Foundations

Analysis includes:
- **Residual Learning**: How SteerMoE relates to ResNet-style residuals
- **Manifold Constraints**: Keeping representations on valid manifolds
- **Expert Specialization**: How different experts learn different transformations
- **Cross-Modal Alignment**: Bridging audio and text representations

### Implementation Details

Covers:
- **Single Router Design**: Why use a shared router vs. per-layer routers
- **Gradient Flow**: How gradients flow during training
- **Stability Techniques**: Methods to prevent training instability
- **Normalization Strategies**: When and how to apply normalization

## 🔍 Key Insights

From the documentation:

1. **SteerMoE as Routed Residual Adapter**
   - Not a traditional residual connection
   - Combines MoE routing with residual learning
   - Provides input-dependent adaptation

2. **Connection to Manifold Hyper-Connections**
   - Shares concepts with DeepSeek's mHC approach
   - Both focus on preserving representation manifolds
   - SteerMoE achieves this through steering vectors

3. **Design Trade-offs**
   - Parameter efficiency vs. expressiveness
   - Single router vs. multiple routers
   - Layer-wise vs. post-encoder steering

## 📖 Recommended Reading Order

For new users:
1. Start with **Introduction** to understand motivation
2. Read **Understanding SteerMoE Architecture** for core concepts
3. Check **Practical Recommendations** for implementation guidance
4. Explore **Comparison sections** for design choices

For researchers:
1. Focus on **Theoretical sections** (Residual Connections, mHC)
2. Study **Normalization and Manifold Constraints**
3. Analyze **Design Trade-offs** and alternatives

## 🔗 Related Documentation

- Core implementations: [`steer_moe/`](../steer_moe/)
- Training scripts: [`scripts/`](../scripts/)
- Configuration files: [`configs/`](../configs/)
- Exploratory research: [`cross_modal_steer/`](../cross_modal_steer/)
- Deprecated implementations: [`deprecated/`](../deprecated/)
