# 已处理数据集目录

本文件夹用于存储 HuggingFace Dataset 格式的预处理数据集文件。

## 📁 状态

**当前为空** - 此目录作为占位符存在，用于存放预处理流程中生成的已处理数据集文件。

## 🎯 用途

此目录用于存储 [`pre_process/`](../pre_process/) 文件夹中预处理脚本的输出。运行预处理脚本后，数据集将以 HuggingFace Arrow 格式保存在此处。

## 📦 预期结构

预处理后，此目录将包含每个已处理数据集的子目录：

```
processed_datasets/
├── librispeech_train_clean_100/
│   ├── dataset_info.json
│   └── data-00000-of-00001.arrow
├── librispeech_train_clean_360/
│   ├── dataset_info.json
│   └── data-00000-of-00001.arrow
├── aishell_train/
│   ├── dataset_info.json
│   └── data-00000-of-00001.arrow
└── clothoaqa_train/
    ├── dataset_info.json
    └── data-00000-of-00001.arrow
```

## 🚀 生成已处理数据集

要填充此目录，运行预处理脚本：

```bash
# 处理 LibriSpeech
python pre_process/pre_process_librispeech.py \
  --output_dir processed_datasets/librispeech_train_clean_100

# 处理 AISHELL
python pre_process/pre_process_aishell.py \
  --output_dir processed_datasets/aishell_train

# 处理 ClothoAQA
python pre_process/pre_process_clothoaqa.py \
  --output_dir processed_datasets/clothoaqa_train
```

## 📊 数据集格式

每个已处理数据集将包含：
- `input_features`：音频特征（mel 频谱图或 FBANK）
- `labels`：分词后的文本（LLM token ID）
- `input_length`：音频时长（秒）
- `attention_mask`：文本 token 的注意力掩码
- `text`：原始转录（供参考）

## 💾 存储注意事项

已处理数据集可能很大：
- **LibriSpeech train-clean-100**：约 10GB
- **LibriSpeech train-clean-360**：约 35GB
- **AISHELL train**：约 15GB
- **ClothoAQA**：约 5GB

预处理前请确保有足够的磁盘空间。

## 🔗 相关文档

- 预处理脚本：[`pre_process/`](../pre_process/) | [`pre_process/README_CN.md`](../pre_process/README_CN.md)
- 训练配置：[`configs/`](../configs/) | [`configs/README_CN.md`](../configs/README_CN.md)
- 训练中的使用：[`scripts/`](../scripts/) | [`scripts/README_CN.md`](../scripts/README_CN.md)
