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

def create_aishell_hf_dataset(whisper_feature_extractor, llm_tokenizer, audio_base_path, trans_file_path, output_path):
    """
    Generates a Hugging Face audio dataset from the AISHELL dataset.

    Args:
        audio_base_path (str): The root directory where the audio wav files are stored.
        trans_file_path (str): The path to the transcription file (trans.txt).
        output_path (str): The path where the generated Hugging Face dataset will be saved.
    """
    # 1. Read the transcription file and create a mapping
    transcriptions = {}
    with open(trans_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                audio_id, text = parts
                transcriptions[audio_id] = text

    logging.info(f"transcriptions: {len(transcriptions)}")

    # 2. Find all audio files and their corresponding transcriptions
    audio_files = []
    transcription_list = []
    for dirpath, _, filenames in os.walk(audio_base_path):
        for filename in filenames:
            if filename.endswith(".wav"):
                audio_id = os.path.splitext(filename)[0]
                if audio_id in transcriptions:
                    full_audio_path = os.path.join(dirpath, filename)
                    audio_files.append(full_audio_path)
                    transcription_list.append(transcriptions[audio_id])

    logging.info(f"audio_files: {len(audio_files)}")
    logging.info(f"transcription_list: {len(transcription_list)}")

    # 3. Create a Hugging Face Dataset
    batch_size=100
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

        # Using from_dict is memory-efficient for this step
        raw_dataset = Dataset.from_dict(dataset_dict)

        # 4. Cast the "audio" column to the Audio feature type
        # This will load and resample audio on the fly when you access it.
        hf_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))

        tmp_dataset=hf_dataset

        def _prepare(batch):
            # Use whisper_encoder for proper audio feature extraction
            return prepare_dataset(batch, 'audio', 'sentence', whisper_feature_extractor, llm_tokenizer, 16000)

        try:
            tmp_dataset = tmp_dataset.map(_prepare, remove_columns=tmp_dataset.column_names, num_proc=100)
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
    audio_directory = '/root/autodl-nas/ruitao/data/aishell/data/wav'
    transcription_file = '/root/autodl-nas/ruitao/data/aishell/data/trans.txt'
    output_dataset_path = '/root/autodl-nas/ruitao/data/processed_aishell/'

    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("/root/autodl-tmp/model/whisper-large-v3")
    # whisper_encoder = WhisperEncoder("/root/autodl-tmp/model/whisper-large-v3")
    llm_tokenizer=AutoTokenizer.from_pretrained("/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct/")

    create_aishell_hf_dataset(whisper_feature_extractor, llm_tokenizer, audio_directory, transcription_file, output_dataset_path)

    # You can later load the dataset from the saved path like this:
    # from datasets import load_from_disk
    # loaded_dataset = load_from_disk(output_dataset_path)
    # print(loaded_dataset)