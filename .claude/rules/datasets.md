# SteerMoE 数据集与模型路径

> 切换服务器后，数据集存放路径为 `~/data`，模型权重存放路径为 `~/models`。

## 模型权重路径 (`~/models/`)

| 模型 | 路径 | 用途 |
|------|------|------|
| Whisper Large V3 | `~/models/whisper-large-v3/` | 音频编码器 (Whisper 路径) |
| Qwen2.5-7B-Instruct | `~/models/Qwen2.5-7B-Instruct/` | LLM 解码器 |
| Qwen2.5-3B-Instruct | `~/models/Qwen2.5-3B-Instruct/` | LLM 解码器 (轻量) |
| Qwen3-4B-Instruct | `~/models/Qwen3-4B-Instruct-2507/` | 实验用 LLM |
| Qwen3.5-9B | `~/models/Qwen3.5-9B/` | 实验用 LLM |
| Qwen3-ASR-1.7B | `~/models/Qwen3-ASR-1.7B/` | ASR 专用模型 |
| VibeVoice-ASR | `~/models/VibeVoice-ASR/` | ASR 对比模型 |

注意：FireRedASR-AED-L (Conformer 编码器) 此前位于 `/mnt/models/FireRedTeam/FireRedASR-AED-L`，切换服务器后需确认新位置。

## 原始数据集路径 (`~/data/`)

### LibriSpeech ASR

| 路径 | 说明 |
|------|------|
| `~/data/librispeech_asr/all/train.clean.100/` | 训练集 100h (parquet) |
| `~/data/librispeech_asr/all/train.clean.360/` | 训练集 360h (parquet) |
| `~/data/librispeech_asr/all/train.other.500/` | 训练集 500h noisy (parquet) |
| `~/data/librispeech_asr/all/test.clean/` | 测试集 clean (parquet) |
| `~/data/librispeech_asr/all/test.other/` | 测试集 other (parquet) |
| `~/data/librispeech_asr/all/validation.clean/` | 验证集 clean (parquet) |
| `~/data/librispeech_asr/all/validation.other/` | 验证集 other (parquet) |

### AISHELL (中文 ASR)

| 路径 | 说明 |
|------|------|
| `~/data/AISHELL-DEV-TEST-SET/iOS/test/wav/` | 测试集音频 |
| `~/data/AISHELL-DEV-TEST-SET/iOS/test/trans.txt` | 测试集转录 |

### ClothoAQA (音频问答)

| 路径 | 说明 |
|------|------|
| `~/data/ClothoAQA/clotho_aqa/` | 原始 CSV + 音频文件 |
| `~/data/ClothoAQA/clotho_asqa_test_v2/` | 测试集 v2 |

### MMAU-Pro (多任务音频理解)

| 路径 | 说明 |
|------|------|
| `~/data/MMAU-Pro/test.parquet` | 测试集 (parquet) |

### OpenAudioBench (评估基准)

| 路径 | 说明 |
|------|------|
| `~/data/OpenAudioBench/eval_datas/` | 评估数据子目录 |

## 预处理后数据集路径 (`~/data/processed_datasets/`)

预处理脚本将原始数据转换为 HuggingFace Dataset 格式，保存在此目录。

| 路径 | 编码器 | 来源数据 |
|------|--------|---------|
| `~/data/processed_datasets/librispeech_asr/train.clean.100/` | Whisper | LibriSpeech 100h |
| `~/data/processed_datasets/librispeech_asr/train.clean.360/` | Whisper | LibriSpeech 360h |
| `~/data/processed_datasets/librispeech_asr/test.clean/` | Whisper | LibriSpeech test |
| `~/data/processed_datasets/librispeech_asr_for_conformer/` | Conformer | LibriSpeech (int16 raw) |
| `~/data/processed_datasets/ClothoAQA/` | Whisper | ClothoAQA QA |
| `~/data/processed_datasets/ClothoAQA_Conformer/` | Conformer | ClothoAQA QA (int16 raw) |
| `~/data/processed_datasets/mmau-pro/` | Whisper | MMAU 基准 |
| `~/data/processed_datasets/processed_aishell/` | Whisper | AISHELL 训练集 |
| `~/data/processed_datasets/processed_aishell_test/` | Whisper | AISHELL 测试集 |
| `~/data/processed_datasets/OpenAudioBench/` | Whisper | OpenAudioBench |

## 预处理后的数据 Schema

所有预处理脚本输出相同的 schema (保存为 HuggingFace Arrow Dataset):

```python
{
    'input_features': np.ndarray,   # Whisper: float32 (128, T), Conformer: int16 1D
    'labels': List[int],            # LLM tokenizer 输出的 token IDs
    'input_length': float,          # 音频时长 (秒)
    'attention_mask': List[int],    # tokenizer 的 attention mask
    'text': str,                    # 原始文本
    # 以下字段仅 QA 任务:
    'text_prompt': str,             # 问题提示文本
    'sample_rate': int,             # 仅 Conformer: 采样率
}
```

## 预处理命令

```bash
# LibriSpeech (Whisper)
python pre_process/pre_process_librispeech.py

# LibriSpeech (Conformer)
python pre_process/pre_process_librispeech_for_conformer.py

# AISHELL 训练集
python pre_process/pre_process_aishell.py

# AISHELL 测试集
python pre_process/pre_process_aishell_test.py

# ClothoAQA (带去重和时长过滤)
python pre_process/pre_process_clothoaqa_rm_dup.py          # Whisper
python pre_process/pre_process_clothoaqa_rm_dup_conformer.py # Conformer

# MMAU-Pro
python pre_process/pre_process_mmau.py

# OpenAudioBench
python pre_process/pre_process_openaudiobench.py
```

注意：预处理脚本中的硬编码路径需要更新为 `~/data/` 和 `~/models/` 前缀。大多数脚本使用 `num_proc=200` 进行并行处理。

## 配置文件中的路径映射

YAML 配置文件中的旧路径应映射到新路径：

| 旧路径 | 新路径 |
|--------|--------|
| `/mnt/models/...` | `~/models/...` |
| `/mnt/processed_datasets/...` | `~/data/processed_datasets/...` |
| `/mnt/datasets/...` | `~/data/...` |
| `/home/fengruitao/models/...` | `~/models/...` |
| `/home/fengruitao/rt_nas/data/...` | `~/data/...` |
| `/root/autodl-tmp/model/...` | `~/models/...` |
| `/root/autodl-nas/ruitao/data/...` | `~/data/...` |
