# Processed Datasets Directory

This folder is designated for storing preprocessed dataset artifacts in HuggingFace Dataset format.

## 📁 Status

**Currently Empty** - This directory exists as a placeholder for processed dataset files that will be generated during the preprocessing pipeline.

## 🎯 Purpose

This directory is intended to store the output of preprocessing scripts from the [`pre_process/`](../pre_process/) folder. After running preprocessing scripts, datasets will be saved here in HuggingFace Arrow format.

## 📦 Expected Structure

After preprocessing, this directory will contain subdirectories for each processed dataset:

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

## 🚀 Generating Processed Datasets

To populate this directory, run preprocessing scripts:

```bash
# Process LibriSpeech
python pre_process/pre_process_librispeech.py \
  --output_dir processed_datasets/librispeech_train_clean_100

# Process AISHELL
python pre_process/pre_process_aishell.py \
  --output_dir processed_datasets/aishell_train

# Process ClothoAQA
python pre_process/pre_process_clothoaqa.py \
  --output_dir processed_datasets/clothoaqa_train
```

## 📊 Dataset Format

Each processed dataset will contain:
- `input_features`: Audio features (mel-spectrogram or FBANK)
- `labels`: Tokenized text (LLM token IDs)
- `input_length`: Audio duration in seconds
- `attention_mask`: Attention mask for text tokens
- `text`: Original transcription (for reference)

## 💾 Storage Considerations

Processed datasets can be large:
- **LibriSpeech train-clean-100**: ~10GB
- **LibriSpeech train-clean-360**: ~35GB
- **AISHELL train**: ~15GB
- **ClothoAQA**: ~5GB

Ensure sufficient disk space before preprocessing.

## 🔗 Related Documentation

- Preprocessing scripts: [`pre_process/`](../pre_process/)
- Training configurations: [`configs/`](../configs/)
- Usage in training: [`scripts/`](../scripts/)
