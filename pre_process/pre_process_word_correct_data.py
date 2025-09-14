"""
 * author Ruitao Feng
 * created on 21-03-2025
 * github: https://github.com/forfrt
"""

import os
import pandas as pd
import tqdm
# 加载数据集
from datasets import load_dataset
from datasets import DatasetDict, load_from_disk
from datasets import Audio
from datasets import load_from_disk

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

# Init feature extractor and tokenizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, AutoTokenizer

processed_data = []

# # 构建特征 - Use WhisperEncoder for proper audio feature extraction
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from steer_moe.tokenizer.whisper_Lv3.whisper import WhisperEncoder

# Load WhisperEncoder for audio feature extraction
whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("/root/autodl-tmp/model/whisper-large-v3")
# whisper_encoder = WhisperEncoder("/root/autodl-tmp/model/whisper-large-v3")
llm_tokenizer=AutoTokenizer.from_pretrained("/root/autodl-tmp/model/Qwen2.5-7B-Ins-1M/")
# dataset = load_dataset("parquet", data_files="/root/autodl-tmp/wys/whisper/tts_test/train_terms_20231224.parquet", split='train', cache_dir='.cache')
# dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# dataset = dataset.map(data_pipe.prepare_dataset, remove_columns=dataset.column_names)
# dataset.save_to_disk("/root/autodl-tmp/wys/whisper/tts_test/train_terms_20231224_processed")
# exit()

def prepare_dataset(batch, audio_column, text_column, feature_extractor, tokenizer, sample_rate=16000):
    """
    Map function to process a batch: loads audio, extracts features, tokenizes text.
    Args:
        batch: dict with audio and text columns
        audio_column: str, name of audio file path column
        text_column: str, name of text column
        whisper_processor: callable, e.g. WhisperEncoder.tokenize_waveform
        tokenizer: HuggingFace tokenizer for text
        sample_rate: int, target sample rate
    Returns:
        dict with processed audio features and tokenized text
    """
    # Load audio
    audio_path = batch[audio_column]

    # compute log-Mel input features from input audio array
    # (128, 3000), float32 for a 30s audio
    input_features = feature_extractor(audio_path["array"], sampling_rate=audio_path["sampling_rate"]).input_features[0]
    input_length=len(audio_path['array'])/audio_path['sampling_rate']
    
    # Tokenize text
    text = batch[text_column]
    # could also pad here, but the input batch contains only 1 data, so pointless
    # text_tokens = tokenizer(text, return_tensors='pt', padding='longest', truncation=True)
    text_tokens = tokenizer(text)

    logging.info(f"input_features: {input_features.shape}, dtype: {input_features.dtype}")
    logging.info(f"text: {text}, input_ids: {len(text_tokens['input_ids'])}")
    logging.info(f"attention_mask: {len(text_tokens['attention_mask'])}")
    
    return {
        'input_features': input_features,  # Use 'input_features' to match training pipeline
        'labels': text_tokens['input_ids'],  # Use 'labels' to match training pipeline  

        'input_length': input_length,
        'attention_mask': text_tokens['attention_mask'],
        'text': text
    }


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_tag', required=True, help='Batch ID')
    args = parser.parse_args()

    # BATCH_TAG=1
    # SOURCE_TAG='WORD_CORRECT'
    if args.batch_tag:
        logging.info(f"batch_tag from arugments: {args.batch_tag}")
        batch_tag=args.batch_tag
    else:
        batch_tag = f'{setting.SOURCE_TAG}_{setting.BATCH_TAG}'
        logging.info(f"batch_tag from setting.json: {batch_tag}")
    # # features extraction
    # ori_parquet_dir = f'/root/autodl-tmp/ruitao/whisper_test/data/train/origin_parquet/{batch_tag}'
    ori_parquet_dir = f'/root/autodl-nas/ruitao/data/train/origin_parquet/{batch_tag}'
    processed_parquet_dir = f'/root/autodl-nas/ruitao/data/train/processed_parquet/{batch_tag}_processed_in_batch'
    processed_status_dir = f'/root/autodl-nas/ruitao/data/train/'
    processed_parquet_prefix = f'/root/autodl-nas/ruitao/data/train/processed_parquet/'

    logging.info(f"processed_parquet_dir: {processed_parquet_dir}")

    if not os.path.exists(processed_parquet_dir):
        os.makedirs(processed_parquet_dir)

    train_files = os.listdir(ori_parquet_dir)
    logging.info(f"len(train_files): {len(train_files)}")
    # 获取已处理过的文件
    processed_batchs_df = os.listdir(processed_status_dir)
    processed_batchs = []

    for f in processed_batchs_df:
        if f.startswith(f'processed_batch_{batch_tag}_'):
            df = pd.read_csv(f'{processed_status_dir}{f}')
            processed_batchs.extend(df['file'].tolist())

    processed_batchs = [file.split('/')[-1] for file in processed_batchs]
    train_files = [file for file in train_files if file not in processed_batchs]
    logging.info(f"train_files_len: {len(train_files)}")

    # exit()
    batch_size=10
    error_batchs = []
    import traceback
    for i in tqdm.tqdm(range(0, len(train_files), batch_size)):
        offset=(len(processed_batchs)//batch_size*batch_size)
        batch = train_files[i:i+batch_size]
        i+=offset

        logging.info(f"processing batch id: {i}, offset: {offset}")

        batch = [os.path.join(ori_parquet_dir,file) for file in batch]
        logging.info(batch)

        # simplex style error handling, adapt or spend more time on it
        try:
            tmp_dataset = load_dataset("parquet", data_files=batch, split='train', cache_dir='.cache')
        except:
            logging.error(f"batch: {batch}, i: {i}, offset: {offset}")
            logging.error(traceback.format_exc())
            error_batchs.append(batch)
            continue
        tmp_dataset = tmp_dataset.cast_column("audio", Audio(sampling_rate=16000))


        def _prepare(batch):
            # Use whisper_encoder for proper audio feature extraction
            return prepare_dataset(batch, 'audio', 'sentence', whisper_feature_extractor, llm_tokenizer, 16000)

        try:
            tmp_dataset = tmp_dataset.map(_prepare, remove_columns=tmp_dataset.column_names)
            # tmp_dataset = tmp_dataset.map(data_pipe.prepare_dataset, remove_columns=tmp_dataset.column_names, num_proc=100)
        except:
            logging.error(f"batch: {batch}, i: {i}, offset: {offset}")
            logging.error(traceback.format_exc())
            error_batchs.append(batch)
            continue
        folder_path = f"{processed_parquet_prefix}/{batch_tag}_processed_in_batch/batch_{i}_{i + batch_size}"

        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # 然后保存数据集
        tmp_dataset.save_to_disk(folder_path)
        processed_batch = [{'file': file} for file in batch]
        tmp_batch_df = pd.DataFrame(processed_batch)
        tmp_batch_df.to_csv(f"{processed_status_dir}processed_batch_{batch_tag}_{i}.csv", index=False)

    logging.error(error_batchs)
