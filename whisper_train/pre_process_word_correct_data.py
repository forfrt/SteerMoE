"""
 * author Ruitao Feng
 * created on 21-03-2025
 * github: https://github.com/forfrt
"""

from utils import CosFileServer
import setting
import os
import pandas as pd
import tqdm
# 加载数据集
from datasets import load_dataset
from datasets import DatasetDict, load_from_disk
from datasets import Audio
from data_process import HFDataProcessorPipeline
from datasets import load_from_disk

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

# Init feature extractor and tokenizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from setting import model_path, tokenizer_path
def download_from_cos(path,tag='train'):
    audio_path = path
    # 将audio_path的最后一个点替换为下划线，然后添加后缀_train.parquet
    base, extension = audio_path.rsplit('.', 1)
    parquet_path = f"{base}_{extension}_{tag}.parquet"
    cos = CosFileServer(setting.COS_BUCKET, setting.COS_SECRET_ID, setting.COS_SECRET_KEY)
    file_is_exits = cos.exists_object(parquet_path)
    if not file_is_exits:
        return False
    cos.download_object(parquet_path, os.path.join(f'./tmp_scripts/tmp_data_{tag}',parquet_path.split('/')[-1]))
    return True

processed_data = []

# # 构建特征
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path["whisper_large-v2"])
tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path=model_path["whisper_large-v2"],
                                             language="chinese", task="transcribe")
data_pipe = HFDataProcessorPipeline(feature_extractor=feature_extractor, tokenizer=tokenizer)
# dataset = load_dataset("parquet", data_files="/root/autodl-tmp/wys/whisper/tts_test/train_terms_20231224.parquet", split='train', cache_dir='.cache')
# dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# dataset = dataset.map(data_pipe.prepare_dataset, remove_columns=dataset.column_names)
# dataset.save_to_disk("/root/autodl-tmp/wys/whisper/tts_test/train_terms_20231224_processed")
# exit()

def prepare_dataset(batch, audio_column, text_column, whisper_processor, tokenizer, sample_rate=16000):
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
    # If using datasets.Audio, batch[audio_column] may be a dict with 'array' and 'sampling_rate'
    if isinstance(audio_path, dict) and 'array' in audio_path:
        waveform = torch.tensor(audio_path['array'], dtype=torch.float32)
        if audio_path['sampling_rate'] != sample_rate:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, audio_path['sampling_rate'], sample_rate)
    else:
        import soundfile as sf
        waveform, sr = sf.read(audio_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if sr != sample_rate:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)  # convert to mono
    waveform = waveform.unsqueeze(0)  # (1, T)
    # Extract audio features
    audio_features = whisper_processor.tokenize_waveform(waveform)
    # Tokenize text
    text = batch[text_column]
    text_tokens = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)
    
    return {
        'audio_features': audio_features.squeeze(0),  # Remove batch dimension
        'input_ids': text_tokens['input_ids'].squeeze(0),
        'attention_mask': text_tokens['attention_mask'].squeeze(0),
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
            return prepare_dataset(batch, 'audio', 'text', tokenizer, llm_tokenizer, 'sample_rate')

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

    # 删除.cache/parquet文件夹
    import shutil

    try:
        if os.path.exists('.cache/parquet'):
            # 删除文件夹及其内容
            shutil.rmtree('.cache/parquet')
    except Exception as e:
        logging.error(f"eeeeeeeeeeeeeeeeeeee: {e}")
    # 标记完成
    with open(f"../status_folder/feature_extraction_{batch_tag}.done", "w") as f:
        f.write("done")
# dataset = dataset.map(data_pipe.prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=20)
# # print(common_voice)
# # print(common_voice["input_features"][0])
#
# # save data to disk
# dataset.save_to_disk("./tmp_scripts/aopeng_999")
