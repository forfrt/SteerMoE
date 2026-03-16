# 测试目录

本文件夹包含用于验证 SteerMoE 组件的测试文件。

## 📁 文件

### 测试脚本

**`test_aligner.py`**
- 测试 `SteerMoEAligner` 组件和负载均衡损失计算
- 单个综合测试函数：`test_aligner_output_and_lb_loss()`

**测试内容：**
1. 创建一个 SteerMoEAligner 实例，特征维度为 16，4 个专家
2. 生成随机输入张量，形状为 (batch=2, seq_len=5, feature_dim=16)
3. 通过对齐器传递输入，设置 `return_gating=True` 以获取对齐的隐藏状态和门控分数
4. 验证输出形状：
   - `h_aligned` 应为 (2, 5, 16) - 与输入相同
   - `gating_scores` 应为 (2, 5, 4) - batch × 序列 × 专家数
5. 从门控分数计算负载均衡损失
6. 断言负载均衡损失非负 (>= 0)

**依赖项：**
- `torch` - PyTorch 张量
- `steer_moe.aligner.SteerMoEAligner` - 被测试的对齐器组件
- `steer_moe.utils.load_balancing_loss` - 计算负载均衡损失的工具函数

**使用方法：**
```bash
python tests/test_aligner.py
```

**预期输出：**
```
Test passed: Aligner output and load balancing loss are correct!
```

### 音频文件

**`output_audio.wav`**
- 测试输出音频文件（可能在测试期间生成）

**`小鹏汽车2024年四季报业绩交流会.mp3`**
- 中文音频文件（似乎是季度财报电话会议的录音）
- 用于测试中文 ASR 功能

## 🔧 运行测试

运行所有测试：
```bash
# 运行特定测试
python tests/test_aligner.py

# 使用 pytest 运行（如果可用）
pytest tests/
```

## 📝 添加新测试

添加新测试文件时，遵循以下结构：

```python
import torch
from steer_moe.your_module import YourComponent

def test_your_component():
    """测试描述"""
    # 设置
    component = YourComponent(...)
    input_data = torch.randn(...)

    # 执行
    output = component(input_data)

    # 验证
    assert output.shape == expected_shape
    assert output.dtype == expected_dtype

    print("测试通过！")

if __name__ == "__main__":
    test_your_component()
```

## 🔗 相关文档

- 核心实现：[`steer_moe/`](../steer_moe/) | [`steer_moe/README_CN.md`](../steer_moe/README_CN.md)
- 训练脚本：[`scripts/`](../scripts/) | [`scripts/README_CN.md`](../scripts/README_CN.md)
