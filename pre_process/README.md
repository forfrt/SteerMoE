# Data Preprocessing

This folder contains scripts for preprocessing audio datasets into the format required by SteerMoE training.

## 📋 Overview

The preprocessing pipeline:
1. **Loads audio files** from various dataset formats
2. **Extracts audio features** using Whisper or Conformer feature extractors
3. **Tokenizes transcriptions** using the LLM tokenizer
4. **Saves preprocessed data** in HuggingFace Dataset format (Parquet)

## 🎯 Main Preprocessing Scripts

### Active Scripts (Used in Current Experiments)

| Script | Dataset | Language | Description |
|--------|---------|----------|-------------|
| `pre_process_librispeech.py` | LibriSpeech | English | English ASR dataset |
| `pre_process_aishell.py` | AISHELL-1 | Chinese | Mandarin Chinese ASR |
| `pre_process_aishell_test.py` | AISHELL-1 | Chinese | Test set preprocessing |
| `pre_process_clothoaqa.py` | ClothoAQA | English | Audio question answering |
| `pre_process_clothoaqa_rm_dup.py` | ClothoAQA | English | ClothoAQA with deduplication |
| `pre_process_clothoaqa_rm_dup_conformer.py` | ClothoAQA | English | ClothoAQA for Conformer encoder |
| `pre_process_librispeech_for_conformer.py` | LibriSpeech | English | LibriSpeech for Conformer |
| `pre_process_openaudiobench.py` | OpenAudioBench | Multi | Multi-task audio benchmark |
| `pre_process_mmau.py` | MMAU | Multi | Multi-modal audio understanding |

### Legacy Scripts (Inspirational Reference Only)

⚠️ **Note**: These scripts were used for original Whisper model training with specialized word correction data. They are **not used in current SteerMoE experiments** but may be useful as reference for other preprocessing tasks.

| Script | Original Purpose |
|--------|-----------------|
| `main_word_correct_clips.py` | Training original Whisper with word correction data |
| `generate_raw_parquet_multi_processing_word.py` | Multi-processing pipeline for word correction data |
| `generate_raw_parquet_multi_processing_word.sh` | Shell script for parallel processing |
| `pre_process_word_correct_data.py` | Word correction data preprocessing |
| `pre_process_word_correct_data.sh` | Batch word correction preprocessing |

These scripts demonstrate advanced techniques like:
- Multi-processing for large-scale data
- Custom data augmentation
- Word-level alignment
- Batch processing pipelines

Feel free to adapt them for your custom preprocessing needs!

## 🚀 Quick Start

### 1. LibriSpeech (English)

```bash
python pre_process/pre_process_librispeech.py \
  --audio_dir /path/to/LibriSpeech/train-clean-100 \
  --output_dir /path/to/processed_librispeech/train.clean.100 \
  --whisper_model /path/to/whisper-large-v3 \
  --llm_tokenizer /path/to/Qwen2.5-7B-Instruct
```

**Dataset Structure Expected**:
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

### 2. AISHELL (Chinese)

```bash
python pre_process/pre_process_aishell.py \
  --audio_dir /path/to/aishell/data/wav \
  --trans_file /path/to/aishell/data/trans.txt \
  --output_dir /path/to/processed_aishell/train \
  --whisper_model /path/to/whisper-large-v3 \
  --llm_tokenizer /path/to/Qwen2.5-7B-Instruct
```

**Dataset Structure Expected**:
```
aishell/
└── data/
    ├── wav/
    │   └── train/
    │       ├── S0002/
    │       │   ├── BAC009S0002W0122.wav
    │       │   └── BAC009S0002W0123.wav
    │       └── S0003/
    └── trans.txt  # Format: audio_id transcription
```

### 3. ClothoAQA (Audio QA)

```bash
python pre_process/pre_process_clothoaqa.py \
  --data_dir /path/to/clotho_aqa \
  --output_dir /path/to/processed_clothoaqa/train \
  --whisper_model /path/to/whisper-large-v3 \
  --llm_tokenizer /path/to/Qwen2.5-7B-Instruct
```

## 📦 Output Format

All preprocessing scripts generate datasets with the following schema:

```python
{
    'input_features': np.ndarray,  # Shape: (128, T) for Whisper
                                    # 128 mel bins, T time frames
    'labels': List[int],           # Tokenized text (LLM token IDs)
    'input_length': float,         # Audio duration in seconds
    'attention_mask': List[int],   # Attention mask for text tokens
    'text': str                    # Original transcription (for debugging)
}
```

### Feature Dimensions

| Encoder | Feature Type | Dimensions | Frame Rate |
|---------|-------------|------------|------------|
| Whisper-large-v3 | Mel-spectrogram | 128 × T | 50 Hz (50 frames/sec) |
| Conformer | FBANK + CMVN | 80 × T | 100 Hz |

**Example**: 30-second audio
- Whisper: `(128, 3000)` - 3000 frames at 50 Hz
- Conformer: `(80, 3000)` - 3000 frames at 100 Hz (typically)

## 🛠️ Core Utilities

### `preprocess_utils.py`

Contains reusable preprocessing functions:

#### `prepare_dataset()`

Main preprocessing function that handles audio feature extraction and text tokenization.

```python
from preprocess_utils import prepare_dataset

def prepare_dataset(
    batch,                    # Single data sample
    audio_column,             # Name of audio column
    text_column,              # Name of text column
    feature_extractor,        # Whisper or Conformer feature extractor
    tokenizer,                # LLM tokenizer
    sample_rate=16000         # Audio sample rate
):
    """
    Process a single batch of data.
    
    Returns:
        dict: {
            'input_features': audio features,
            'labels': tokenized text,
            'input_length': audio duration,
            'attention_mask': text attention mask,
            'text': original text
        }
    """
```

**For Whisper**:
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

**For Conformer**:
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

Contains helper classes and functions for data processing.

## 📝 Detailed Script Documentation

### `pre_process_librispeech.py`

Processes LibriSpeech ASR corpus.

**Features**:
- Handles `.flac` audio format
- Reads transcription from `.trans.txt` files
- Supports train-clean-100, train-clean-360, train-other-500
- Multi-processing support (200 workers by default)
- Batch processing (10,000 samples per batch)

**Configuration**:
```python
# In script
audio_directory = '/path/to/LibriSpeech/train-clean-100'
output_dataset_path = '/path/to/processed_librispeech/train.clean.100'
whisper_model_path = '/path/to/whisper-large-v3'
llm_tokenizer_path = '/path/to/Qwen2.5-7B-Instruct'
```

**Output Structure**:
```
processed_librispeech/
└── train.clean.100/
    ├── batch_0_10000/
    │   ├── dataset_info.json
    │   └── data-00000-of-00001.arrow
    ├── batch_10000_20000/
    └── ...
```

### `pre_process_aishell.py`

Processes AISHELL-1 Mandarin Chinese corpus.

**Features**:
- Handles `.wav` audio format
- Reads transcription from centralized `trans.txt`
- Batch processing with error handling
- Logs failed batches for debugging

**Transcription Format**:
```
# trans.txt format
BAC009S0002W0122 绿 是 阳春 烟 景 大块 文章 的 底色 四月 的 林 峦 更是 绿 得 鲜活 秀媚 诗意 盎然
BAC009S0002W0123 党 军 愤怒 地 挥舞 着 小 拳头 走 向 了 他
```

**Key Parameters**:
```python
batch_size = 10000          # Process 10k samples at a time
num_proc = 200              # 200 parallel workers
```

### `pre_process_clothoaqa.py`

Processes ClothoAQA audio question answering dataset.

**Features**:
- Audio-text pairs for QA tasks
- Supports both questions and answers
- Handles audio captioning metadata
- Optional deduplication (`pre_process_clothoaqa_rm_dup.py`)

**ClothoAQA Format**:
```json
{
    "audio_id": "audio_00123",
    "audio_path": "/path/to/audio.wav",
    "question": "What sound do you hear?",
    "answer": "A dog barking in the distance"
}
```

### Conformer-Specific Scripts

Scripts with `_for_conformer` suffix use Conformer feature extraction instead of Whisper:

```python
# Different feature extractor
from steer_moe.conformer_module.asr_feat import ASRFeatExtractor
cmvn_path = os.path.join(conformer_model_dir, "cmvn.ark")
feature_extractor = ASRFeatExtractor(cmvn_path)
```

**Output Features**:
- FBANK features with CMVN normalization
- Different dimensionality than Whisper
- Optimized for streaming ASR

## 🔧 Creating Custom Preprocessing Scripts

### Template Script

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
    # 1. Initialize feature extractor and tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        whisper_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_path)
    
    # 2. Load your data
    audio_files = []
    transcriptions = []
    
    # Your custom data loading logic here
    # ...
    
    # 3. Create HuggingFace Dataset
    dataset_dict = {
        "audio": audio_files,
        "sentence": transcriptions,
    }
    raw_dataset = Dataset.from_dict(dataset_dict)
    
    # 4. Cast audio column
    hf_dataset = raw_dataset.cast_column(
        "audio", 
        Audio(sampling_rate=16000)
    )
    
    # 5. Preprocess with feature extraction
    def _prepare(batch):
        return prepare_dataset(
            batch, 'audio', 'sentence',
            feature_extractor, tokenizer, 16000
        )
    
    processed_dataset = hf_dataset.map(
        _prepare,
        remove_columns=hf_dataset.column_names,
        num_proc=200  # Parallel processing
    )
    
    # 6. Save to disk
    processed_dataset.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")
    print(f"Total samples: {len(processed_dataset)}")

if __name__ == '__main__':
    create_custom_dataset(
        audio_dir='/path/to/your/audio',
        trans_file='/path/to/transcriptions.txt',
        output_dir='/path/to/output',
        whisper_model_path='/path/to/whisper-large-v3',
        llm_tokenizer_path='/path/to/Qwen2.5-7B-Instruct'
    )
```

### Key Considerations

1. **Audio Format**: Use `datasets.Audio` for automatic format handling
2. **Sample Rate**: Ensure all audio is resampled to 16kHz
3. **Batch Size**: Adjust based on available RAM (10k is safe)
4. **Error Handling**: Use try-except blocks for robustness
5. **Logging**: Log progress and errors for debugging

## 🐛 Troubleshooting

### Out of Memory During Preprocessing

**Problem**: Script crashes with memory error.

**Solutions**:
1. Reduce `batch_size`: Try `batch_size=5000` or `1000`
2. Reduce `num_proc`: Try `num_proc=50` or `20`
3. Process in smaller chunks:
   ```python
   for i in range(0, len(data), chunk_size):
       chunk = data[i:i+chunk_size]
       process_chunk(chunk)
   ```

### Audio Loading Errors

**Problem**: `soundfile` or `librosa` cannot load audio.

**Solutions**:
1. Install audio backends:
   ```bash
   pip install soundfile librosa
   # For MP3 support
   pip install ffmpeg-python
   ```
2. Verify audio file integrity:
   ```bash
   ffmpeg -v error -i audio.wav -f null -
   ```
3. Convert problematic formats:
   ```bash
   ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
   ```

### Slow Preprocessing

**Problem**: Preprocessing takes too long.

**Solutions**:
1. **Increase parallel workers**: `num_proc=400` (if RAM allows)
2. **Use SSD**: Process data on SSD rather than HDD
3. **Pre-convert audio**: Batch convert to WAV/16kHz first
   ```bash
   # Parallel conversion with GNU parallel
   find audio_dir -name "*.flac" | \
     parallel -j 32 ffmpeg -i {} -ar 16000 {.}.wav
   ```

### Tokenization Issues

**Problem**: Text truncated or incorrectly tokenized.

**Solutions**:
1. Check tokenizer settings:
   ```python
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   # Add special tokens if needed
   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
   ```
2. Verify text encoding:
   ```python
   # For Chinese text
   text = text.encode('utf-8').decode('utf-8')
   ```
3. Handle special characters:
   ```python
   import unicodedata
   text = unicodedata.normalize('NFKC', text)
   ```

## 📊 Preprocessing Statistics

After preprocessing, verify your data:

```python
from datasets import load_from_disk

dataset = load_from_disk('/path/to/processed_data')
print(f"Total samples: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
print(f"Sample: {dataset[0]}")

# Check feature shapes
import numpy as np
features = [len(x['input_features']) for x in dataset]
print(f"Feature length - Min: {min(features)}, Max: {max(features)}, Mean: {np.mean(features):.1f}")

# Check text lengths
labels = [len(x['labels']) for x in dataset]
print(f"Label length - Min: {min(labels)}, Max: {max(labels)}, Mean: {np.mean(labels):.1f}")
```

**Expected Ranges**:
- **Audio features**: 500-3000 frames (5-30 seconds at 50 Hz)
- **Text labels**: 10-500 tokens (typical ASR utterance)

## 🔗 Related Documentation

- Feature extractors: [`steer_moe/tokenizer/`](../steer_moe/tokenizer/)
- Data collators: [`steer_moe/utils.py`](../steer_moe/utils.py)
- Training configs: [`configs/README.md`](../configs/README.md)
- Main README: [`README.md`](../README.md)

---

**Need help?** Open an issue on GitHub or check the main documentation.

