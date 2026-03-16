# 数据预处理

本文件夹包含将音频数据集预处理为 SteerMoE 训练所需格式的脚本。

## 📋 概览

预处理流程：
1. **加载音频文件** — 从各种数据集格式加载
2. **提取音频特征** — 使用 Whisper 或 Conformer 特征提取器
3. **分词转录文本** — 使用 LLM 分词器
4. **保存预处理数据** — 以 HuggingFace Dataset 格式（Parquet）保存

## 🎯 主要预处理脚本

### 活跃脚本（当前实验使用）

| 脚本 | 数据集 | 语言 | 说明 |
|------|--------|------|------|
| `pre_process_librispeech.py` | LibriSpeech | 英文 | 英文 ASR 数据集 |
| `pre_process_librispeech_for_conformer.py` | LibriSpeech | 英文 | Conformer 编码器版本 |
| `pre_process_aishell.py` | AISHELL-1 | 中文 | 普通话中文 ASR |
| `pre_process_aishell_test.py` | AISHELL-1 | 中文 | 测试集预处理 |
| `pre_process_clothoaqa.py` | ClothoAQA | 英文 | 音频问答 |
| `pre_process_clothoaqa_rm_dup.py` | ClothoAQA | 英文 | 带去重的 ClothoAQA |
| `pre_process_clothoaqa_rm_dup_conformer.py` | ClothoAQA | 英文 | Conformer 编码器版本 |
| `pre_process_openaudiobench.py` | OpenAudioBench | 多语言 | 多任务音频基准 |
| `pre_process_mmau.py` | MMAU | 多语言 | 多模态音频理解 |
| `pre_process_word_correct_data.py` | 词纠正数据 | 中文 | 词纠正训练数据 |

### 遗留脚本（仅供参考）

⚠️ **注意**：这些脚本用于原始 Whisper 模型训练和专门的词纠正数据。它们**不用于当前 SteerMoE 实验**，但可作为其他预处理任务的参考。

| 脚本 | 原始用途 |
|------|----------|
| `main_word_correct_clips.py` | 使用词纠正数据训练原始 Whisper |
| `generate_raw_parquet_multi_processing_word.py` | 词纠正数据的多进程流程 |
| `generate_raw_parquet_multi_processing_word.sh` | 并行处理的 Shell 脚本 |
| `main_word_correct_clips.sh` | 词纠正训练的 Shell 脚本 |

这些脚本展示了高级技术，如：
- 大规模数据的多进程处理
- 自定义数据增强
- 词级对齐
- 批处理流程

可根据需要改编用于自定义预处理！

## 🚀 快速开始

### 1. LibriSpeech（英文）

```bash
python pre_process/pre_process_librispeech.py \
  --audio_dir /path/to/LibriSpeech/train-clean-100 \
  --output_dir /path/to/processed_librispeech/train.clean.100 \
  --whisper_model /path/to/whisper-large-v3 \
  --llm_tokenizer /path/to/Qwen2.5-7B-Instruct
```

**预期数据集结构**：
```
LibriSpeech/
└── train-clean-100/
    ├── 19/
    │   ├── 198/
    │   │   ├── 19-198-0000.flac
    │   │   ├── 19-198-0001.flac
    │   │   └── 19-198.trans.txt
    │   └── 227/
    └── 26/
```

### 2. AISHELL（中文）

```bash
python pre_process/pre_process_aishell.py \
  --audio_dir /path/to/aishell/data/wav \
  --trans_file /path/to/aishell/data/trans.txt \
  --output_dir /path/to/processed_aishell/train \
  --whisper_model /path/to/whisper-large-v3 \
  --llm_tokenizer /path/to/Qwen2.5-7B-Instruct
```

**预期数据集结构**：
```
aishell/
└── data/
    ├── wav/
    │   └── train/
    │       ├── S0002/
    │       │   ├── BAC009S0002W0122.wav
    │       │   └── BAC009S0002W0123.wav
    │       └── S0003/
    └── trans.txt  # 格式：audio_id transcription
```

### 3. ClothoAQA（音频问答）

```bash
python pre_process/pre_process_clothoaqa.py \
  --data_dir /path/to/clotho_aqa \
  --output_dir /path/to/processed_clothoaqa/train \
  --whisper_model /path/to/whisper-large-v3 \
  --llm_tokenizer /path/to/Qwen2.5-7B-Instruct
```

## 📦 输出格式

所有预处理脚本生成以下模式的数据集：

```python
{
    'input_features': np.ndarray,  # 形状：(128, T) for Whisper
                                    # 128 个 mel 频带，T 个时间帧
    'labels': List[int],           # 分词后的文本（LLM token ID）
    'input_length': float,         # 音频时长（秒）
    'attention_mask': List[int],   # 文本 token 的注意力掩码
    'text': str                    # 原始转录（用于调试）
}
```

### 特征维度

| 编码器 | 特征类型 | 维度 | 帧率 |
|--------|---------|------|------|
| Whisper-large-v3 | Mel 频谱图 | 128 × T | 50 Hz（50 帧/秒） |
| Conformer | FBANK + CMVN | 80 × T | 100 Hz |

**示例**：30 秒音频
- Whisper：`(128, 3000)` — 50 Hz 下 3000 帧
- Conformer：`(80, 3000)` — 100 Hz 下 3000 帧（通常）

## 🛠️ 核心工具

### `preprocess_utils.py`

包含可重用的预处理函数：

#### `prepare_dataset()`

处理音频特征提取和文本分词的主预处理函数。

```python
from preprocess_utils import prepare_dataset

def prepare_dataset(
    batch,                    # 单个数据样本
    audio_column,             # 音频列名
    text_column,              # 文本列名
    feature_extractor,        # Whisper 或 Conformer 特征提取器
    tokenizer,                # LLM 分词器
    sample_rate=16000         # 音频采样率
):
    """
    处理单批数据。

    返回：
        dict: {
            'input_features': 音频特征,
            'labels': 分词后的文本,
            'input_length': 音频时长,
            'attention_mask': 文本注意力掩码,
            'text': 原始文本
        }
    """
```

**Whisper 版本**：
```python
from transformers import WhisperFeatureExtractor
from preprocess_utils import prepare_dataset

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "/path/to/whisper-large-v3"
)
processed = prepare_dataset(
    batch, 'audio', 'text',
    feature_extractor, tokenizer, 16000
)
```

**Conformer 版本**：
```python
from preprocess_utils import prepare_dataset_for_conformer
from steer_moe.conformer_module.asr_feat import ASRFeatExtractor

feature_extractor = ASRFeatExtractor(cmvn_path)
processed = prepare_dataset_for_conformer(
    batch, 'audio', 'text',
    tokenizer, 16000
)
```

### `modules.py`

包含数据处理的辅助类和函数。

#### `DataCollatorSpeechSeq2SeqWithPadding`

数据整理器类，分别填充音频输入和文本标签，用 -100 替换填充 token 以用于损失计算，处理 BOS token 移除。

#### `ASREvaluator`

使用自定义 WER 指标加载器和批量解码计算 ASR 评估的 WER（词错误率）指标。

### `data_process.py`

包含数据集加载和处理类：

- `DatasetCommonVoiceHindi`：加载 Common Voice 印地语数据集
- `DatasetCommonVoiceCN`：加载 Common Voice 中文数据集
- `DatasetFinAudioCN`：从 parquet 文件加载 FinAudio 中文数据集
- `HFDataProcessorPipeline`：预处理 HuggingFace 数据集的主流程

## 📝 详细脚本文档

### `pre_process_librispeech.py`

处理 LibriSpeech ASR 语料库。

**特性**：
- 处理 `.flac` 音频格式
- 从 `.trans.txt` 文件读取转录
- 支持 train-clean-100、train-clean-360、train-other-500
- 多进程支持（默认 200 个工作进程）
- 批处理（每批 10,000 个样本）

### `pre_process_aishell.py`

处理 AISHELL-1 普通话中文语料库。

**特性**：
- 处理 `.wav` 音频格式
- 从集中的 `trans.txt` 读取转录
- 带错误处理的批处理
- 记录失败批次以便调试

**转录格式**：
```
# trans.txt 格式
BAC009S0002W0122 绿 是 阳春 烟 景 大块 文章 的 底色 四月 的 林 峦 更是 绿 得 鲜活 秀媚 诗意 盎然
BAC009S0002W0123 党 军 愤怒 地 挥舞 着 小 拳头 走 向 了 他
```

### `pre_process_clothoaqa.py`

处理 ClothoAQA 音频问答数据集。

**特性**：
- 问答任务的音频-文本对
- 支持问题和答案
- 处理音频字幕元数据
- 可选去重（`pre_process_clothoaqa_rm_dup.py`）

### Conformer 专用脚本

带 `_for_conformer` 后缀的脚本使用 Conformer 特征提取而非 Whisper：

```python
# 不同的特征提取器
from steer_moe.conformer_module.asr_feat import ASRFeatExtractor
cmvn_path = os.path.join(conformer_model_dir, "cmvn.ark")
feature_extractor = ASRFeatExtractor(cmvn_path)
```

**输出特征**：
- 带 CMVN 归一化的 FBANK 特征
- 与 Whisper 不同的维度
- 针对流式 ASR 优化

## 🔧 创建自定义预处理脚本

### 模板脚本

```python
import os
import tqdm
from datasets import Dataset, Audio
from transformers import WhisperFeatureExtractor, AutoTokenizer
from preprocess_utils import prepare_dataset

def create_custom_dataset(
    audio_dir, trans_file, output_dir,
    whisper_model_path, llm_tokenizer_path
):
    # 1. 初始化特征提取器和分词器
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        whisper_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_path)

    # 2. 加载数据
    audio_files = []
    transcriptions = []

    # 在此处添加自定义数据加载逻辑
    # ...

    # 3. 创建 HuggingFace Dataset
    dataset_dict = {
        "audio": audio_files,
        "sentence": transcriptions,
    }
    raw_dataset = Dataset.from_dict(dataset_dict)

    # 4. 转换音频列
    hf_dataset = raw_dataset.cast_column(
        "audio",
        Audio(sampling_rate=16000)
    )

    # 5. 特征提取预处理
    def _prepare(batch):
        return prepare_dataset(
            batch, 'audio', 'sentence',
            feature_extractor, tokenizer, 16000
        )

    processed_dataset = hf_dataset.map(
        _prepare,
        remove_columns=hf_dataset.column_names,
        num_proc=200  # 并行处理
    )

    # 6. 保存到磁盘
    processed_dataset.save_to_disk(output_dir)
    print(f"数据集已保存到 {output_dir}")
    print(f"总样本数：{len(processed_dataset)}")

if __name__ == '__main__':
    create_custom_dataset(
        audio_dir='/path/to/your/audio',
        trans_file='/path/to/transcriptions.txt',
        output_dir='/path/to/output',
        whisper_model_path='/path/to/whisper-large-v3',
        llm_tokenizer_path='/path/to/Qwen2.5-7B-Instruct'
    )
```

### 关键注意事项

1. **音频格式**：使用 `datasets.Audio` 自动处理格式
2. **采样率**：确保所有音频重采样到 16kHz
3. **批大小**：根据可用 RAM 调整（10k 是安全值）
4. **错误处理**：使用 try-except 块提高鲁棒性
5. **日志记录**：记录进度和错误以便调试

## 🐛 故障排除

### 预处理时内存不足

**问题**：脚本因内存错误崩溃。

**解决方案**：
1. 减小 `batch_size`：尝试 `batch_size=5000` 或 `1000`
2. 减少 `num_proc`：尝试 `num_proc=50` 或 `20`
3. 分块处理：
   ```python
   for i in range(0, len(data), chunk_size):
       chunk = data[i:i+chunk_size]
       process_chunk(chunk)
   ```

### 音频加载错误

**问题**：`soundfile` 或 `librosa` 无法加载音频。

**解决方案**：
1. 安装音频后端：
   ```bash
   pip install soundfile librosa
   # MP3 支持
   pip install ffmpeg-python
   ```
2. 验证音频文件完整性：
   ```bash
   ffmpeg -v error -i audio.wav -f null -
   ```
3. 转换有问题的格式：
   ```bash
   ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
   ```

### 预处理速度慢

**问题**：预处理耗时过长。

**解决方案**：
1. **增加并行工作进程**：`num_proc=400`（如果 RAM 允许）
2. **使用 SSD**：在 SSD 而非 HDD 上处理数据
3. **预转换音频**：先批量转换为 WAV/16kHz
   ```bash
   # 使用 GNU parallel 并行转换
   find audio_dir -name "*.flac" | \
     parallel -j 32 ffmpeg -i {} -ar 16000 {.}.wav
   ```

### 分词问题

**问题**：文本被截断或分词不正确。

**解决方案**：
1. 检查分词器设置：
   ```python
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   # 如需要添加特殊 token
   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
   ```
2. 验证文本编码：
   ```python
   # 中文文本
   text = text.encode('utf-8').decode('utf-8')
   ```
3. 处理特殊字符：
   ```python
   import unicodedata
   text = unicodedata.normalize('NFKC', text)
   ```

## 📊 预处理统计

预处理后，验证数据：

```python
from datasets import load_from_disk

dataset = load_from_disk('/path/to/processed_data')
print(f"总样本数：{len(dataset)}")
print(f"列：{dataset.column_names}")
print(f"样本：{dataset[0]}")

# 检查特征形状
import numpy as np
features = [len(x['input_features']) for x in dataset]
print(f"特征长度 - 最小：{min(features)}，最大：{max(features)}，平均：{np.mean(features):.1f}")

# 检查文本长度
labels = [len(x['labels']) for x in dataset]
print(f"标签长度 - 最小：{min(labels)}，最大：{max(labels)}，平均：{np.mean(labels):.1f}")
```

**预期范围**：
- **音频特征**：500-3000 帧（50 Hz 下 5-30 秒）
- **文本标签**：10-500 个 token（典型 ASR 语句）

## 🔗 相关文档

- 特征提取器：[`steer_moe/tokenizer/`](../steer_moe/tokenizer/)
- 数据整理器：[`steer_moe/utils.py`](../steer_moe/utils.py)
- 训练配置：[`configs/README.md`](../configs/README.md) | [`configs/README_CN.md`](../configs/README_CN.md)
