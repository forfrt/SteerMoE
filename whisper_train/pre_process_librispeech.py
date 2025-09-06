import os
import tqdm
import datasets
import traceback
from datasets import Dataset, Audio
from transformers import WhisperFeatureExtractor, AutoTokenizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

from preprocess_utils import prepare_dataset

def create_aishell_hf_dataset(whisper_feature_extractor, llm_tokenizer, parquets_path, output_path):
    """
    Generates a Hugging Face audio dataset from the AISHELL dataset.

    Args:
        audio_base_path (str): The root directory where the audio wav files are stored.
        trans_file_path (str): The path to the transcription file (trans.txt).
        output_path (str): The path where the generated Hugging Face dataset will be saved.
    """



    # 3. Create a Hugging Face Dataset
    batch_size=10000
    error_batchs=[]
    for i in tqdm.tqdm(range(0, len(audio_files), batch_size)):
        batch_audio_files = audio_files[i:i+batch_size]
        batch_transcription_list = transcription_list[i:i+batch_size]

        if not audio_files:
            raise ValueError("No audio files found in the specified directory.")

        dataset_dict = {
            "audio": batch_audio_files,
            "sentence": batch_transcription_list,
        }

        logging.info(f"batch_audio_files[:10]: {batch_audio_files[:10]}")
        logging.info(f"batch_transcription_list[:10]: {batch_transcription_list[:10]}")

        # Using from_dict is memory-efficient for this step
        raw_dataset = Dataset.from_dict(dataset_dict)

        # 4. Cast the "audio" column to the Audio feature type
        # This will load and resample audio on the fly when you access it.
        hf_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))

        tmp_dataset=hf_dataset

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

        folder_path = f"{output_path}/train/batch_{i}_{i + batch_size}"

        hf_dataset=tmp_dataset

        # 5. Save the dataset to disk
        hf_dataset.save_to_disk(folder_path)
        logging.info(f"Dataset successfully created and saved to {folder_path}")
        logging.info(f"Number of examples: {len(hf_dataset)}")
        logging.info(f"First example: {hf_dataset[0]}")

    logging.info(f"error_batchs: {error_batchs}")

if __name__ == '__main__':
    # PLEASE UPDATE THESE PATHS ACCORDING TO YOUR SETUP
    parquets_path='/root/autodl-nas/ruitao/data/librispeech_asr/all/train.clean.100'
    output_dataset_path = '/root/autodl-nas/ruitao/data/processed_librispeech/train.clean.100'

    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("/root/autodl-tmp/model/whisper-large-v3")
    # whisper_encoder = WhisperEncoder("/root/autodl-tmp/model/whisper-large-v3")
    llm_tokenizer=AutoTokenizer.from_pretrained("/root/autodl-tmp/model/Qwen2.5-7B-Instruct/")

    create_aishell_hf_dataset(whisper_feature_extractor, llm_tokenizer, parquets_path, output_dataset_path)

    # You can later load the dataset from the saved path like this:
    # from datasets import load_from_disk
    # loaded_dataset = load_from_disk(output_dataset_path)
    # print(loaded_dataset)