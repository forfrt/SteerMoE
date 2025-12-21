import os
import tqdm
import datasets
import traceback
from datasets import Dataset, Audio, load_dataset
from transformers import WhisperFeatureExtractor, AutoTokenizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

from preprocess_utils import prepare_dataset

def create_librispeech_hf_dataset(whisper_feature_extractor, llm_tokenizer, parquets_path, output_path):
    """
    Generates a Hugging Face audio dataset from the AISHELL dataset.

    Args:
        audio_base_path (str): The root directory where the audio wav files are stored.
        trans_file_path (str): The path to the transcription file (trans.txt).
        output_path (str): The path where the generated Hugging Face dataset will be saved.
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_files = os.listdir(parquets_path)
    logging.info(f"len(train_files): {len(train_files)}")

    # 3. Create a Hugging Face Dataset
    batch_size=10
    error_batchs=[]
    for i in tqdm.tqdm(range(0, len(train_files), batch_size)):
        batch = train_files[i:i+batch_size]
        batch = [os.path.join(parquets_path,file) for file in batch]

        logging.info(f"processing batch id: {i}")
        logging.info(f"processing batch: {batch}")

        try:
            tmp_dataset = load_dataset("parquet", data_files=batch, split='train', cache_dir='/home/fengruitao/.cache/')
        except:
            logging.error(f"batch: {batch}, i: {i}")
            logging.error(traceback.format_exc())
            error_batchs.append(batch)
            continue
        tmp_dataset = tmp_dataset.cast_column("audio", Audio(sampling_rate=16000))

        logging.info(f"tmp_dataset.column_names: {tmp_dataset.column_names}")

        def _prepare(batch):
            # Use whisper_encoder for proper audio feature extraction
            # https://huggingface.co/datasets/openslr/librispeech_asr
            return prepare_dataset(batch, 'audio', 'text', whisper_feature_extractor, llm_tokenizer, 16000)

        try:
            tmp_dataset = tmp_dataset.map(_prepare, remove_columns=tmp_dataset.column_names, num_proc=200)
            # tmp_dataset = tmp_dataset.map(data_pipe.prepare_dataset, remove_columns=tmp_dataset.column_names, num_proc=100)
        except:
            logging.error(f"batch_size: {batch_size}, i: {i}")
            logging.error(traceback.format_exc())
            error_batchs.append(i)
            continue

        folder_path = f"{output_path}/batch_{i}_{i + batch_size}"

        hf_dataset=tmp_dataset

        # 5. Save the dataset to disk
        hf_dataset.save_to_disk(folder_path)
        logging.info(f"Dataset successfully created and saved to {folder_path}")
        logging.info(f"Number of examples: {len(hf_dataset)}")
        logging.info(f"First example: {hf_dataset[0]}")

    logging.info(f"error_batchs: {error_batchs}")

if __name__ == '__main__':
    # PLEASE UPDATE THESE PATHS ACCORDING TO YOUR SETUP

    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("/home/fengruitao/models/openai/whisper-large-v3")
    llm_tokenizer=AutoTokenizer.from_pretrained("/home/fengruitao/models/Qwen2.5-7B-Instruct/")

    parquets_path='/home/fengruitao/rt_nas/data/librispeech_asr/all/train.clean.100'
    output_dataset_path = '/home/fengruitao/SteerMoE/processed_datasets/librispeech_asr/train.clean.100'

    create_librispeech_hf_dataset(whisper_feature_extractor, llm_tokenizer, parquets_path, output_dataset_path)

    parquets_path='/home/fengruitao/rt_nas/data/librispeech_asr/all/train.clean.360'
    output_dataset_path = '/home/fengruitao/SteerMoE/processed_datasets/librispeech_asr/train.clean.360'

    create_librispeech_hf_dataset(whisper_feature_extractor, llm_tokenizer, parquets_path, output_dataset_path)
    
    parquets_path='/home/fengruitao/rt_nas/data/librispeech_asr/all/test.clean'
    output_dataset_path = '/home/fengruitao/SteerMoE/processed_datasets/librispeech_asr/test.clean'

    create_librispeech_hf_dataset(whisper_feature_extractor, llm_tokenizer, parquets_path, output_dataset_path)

    # You can later load the dataset from the saved path like this:
    # from datasets import load_from_disk
    # loaded_dataset = load_from_disk(output_dataset_path)
    # print(loaded_dataset)