# 文档目录

本文件夹包含 SteerMoE 项目的技术文档和架构分析。

## 📁 文件

### `SteerMoE_Architecture_Analysis.md`

**用途**：SteerMoE 架构的全面技术分析文档（创建于 2026 年 1 月）。

**涵盖的关键部分**：

1. **引言**
   - 介绍 SteerMoE 作为一种高效的逐层引导机制
   - 用于适配冻结的音频编码器（如 Whisper）以实现与 LLM 的跨模态对齐

2. **理解 SteerMoE 架构**
   - 详细说明三个主要可训练组件：
     - `steering_vectors`：表示空间中的学习方向向量（形状：num_layers × num_experts × feature_dim）
     - `router`：用于输入依赖专家选择的线性层
     - `layer_scales`：每层幅度控制参数
   - 解释核心引导公式：`x = layer_output + layer_scale × Σ(gating_score × steering_vector)`

3. **SteerMoE vs 残差连接**
   - 将 SteerMoE 与经典残差连接进行比较
   - 分析 SteerMoE 如何通过使用学习的引导向量和动态路由而非简单的跳跃连接来区别

4. **SteerMoE 是多模态残差连接吗？**
   - 讨论 SteerMoE 是否符合残差连接变体的条件
   - 得出结论，它更适合理解为"路由残差适配器"

5. **与 mHC（流形超连接）的比较**
   - 分析 DeepSeek 的流形超连接方法及其与 SteerMoE 设计的关系
   - 讨论流形约束和稳定性考虑

6. **归一化和流形约束方法**
   - 探索各种归一化技术和流形约束方法
   - 可以改善 SteerMoE 的稳定性和性能

7. **实用建议**
   - 提供实现和优化的可操作指导

8. **结论**
   - 总结 SteerMoE 作为路由残差适配器的身份
   - 讨论与 mHC 原则的联系

**参考文献**：
- 包括对 ResNet、混合专家、LoRA、适配器和 DeepSeek 超连接工作的引用

**使用方法**：
```bash
# 阅读文档
cat docs/SteerMoE_Architecture_Analysis.md

# 或在 Markdown 查看器中打开
```

## 📚 文档主题

### 架构设计

文档深入探讨：
- **引导机制**：引导向量如何修改编码器输出
- **路由策略**：MoE 路由器如何选择专家
- **参数效率**：为什么 SteerMoE 只需要约 180 万可训练参数
- **层缩放**：每层幅度控制的作用

### 理论基础

分析包括：
- **残差学习**：SteerMoE 与 ResNet 风格残差的关系
- **流形约束**：保持表示在有效流形上
- **专家专业化**：不同专家如何学习不同的转换
- **跨模态对齐**：桥接音频和文本表示

### 实现细节

涵盖：
- **单路由器设计**：为什么使用共享路由器而非每层路由器
- **梯度流**：训练期间梯度如何流动
- **稳定性技术**：防止训练不稳定的方法
- **归一化策略**：何时以及如何应用归一化

## 🔍 关键见解

从文档中：

1. **SteerMoE 作为路由残差适配器**
   - 不是传统的残差连接
   - 结合了 MoE 路由和残差学习
   - 提供输入依赖的适配

2. **与流形超连接的联系**
   - 与 DeepSeek 的 mHC 方法共享概念
   - 两者都关注保持表示流形
   - SteerMoE 通过引导向量实现这一点

3. **设计权衡**
   - 参数效率 vs. 表达能力
   - 单路由器 vs. 多路由器
   - 逐层 vs. 编码器后引导

## 📖 推荐阅读顺序

对于新用户：
1. 从**引言**开始理解动机
2. 阅读**理解 SteerMoE 架构**了解核心概念
3. 查看**实用建议**了解实现指导
4. 探索**比较部分**了解设计选择

对于研究人员：
1. 专注于**理论部分**（残差连接、mHC）
2. 研究**归一化和流形约束**
3. 分析**设计权衡**和替代方法

## 🔗 相关文档

- 核心实现：[`steer_moe/`](../steer_moe/) | [`steer_moe/README_CN.md`](../steer_moe/README_CN.md)
- 训练脚本：[`scripts/`](../scripts/) | [`scripts/README_CN.md`](../scripts/README_CN.md)
- 配置文件：[`configs/`](../configs/) | [`configs/README_CN.md`](../configs/README_CN.md)
- 探索性研究：[`cross_modal_steer/`](../cross_modal_steer/) | [`cross_modal_steer/README_CN.md`](../cross_modal_steer/README_CN.md)
- 已弃用实现：[`deprecated/`](../deprecated/) | [`deprecated/README_CN.md`](../deprecated/README_CN.md)
