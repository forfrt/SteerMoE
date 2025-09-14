import os
import tqdm
import datasets
import traceback
from datasets import Dataset, Audio, load_dataset
from transformers import WhisperFeatureExtractor, AutoTokenizer
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')


def filter_single_audio_path(example):
    return len(example['audio_path']) == 1

def update_audio_paths(example):
    """
    Function to update the 'audio_path' for each example.
    'example' is a dictionary representing a single row in the dataset.
    """
    example['audio_path'] = ["/mnt/datasets/mmau-pro/" + audio_path for audio_path in example['audio_path']]
    return example

def flatten_list_to_string(example):
    # We already filtered and mapped, so each 'audio_path' is guaranteed to be ['/prefix/path.wav']
    example['audio_path'] = example['audio_path'][0]
    return example

def prepare_dataset(batch,  feature_extractor, tokenizer, sampling_rate):
    # logging.info(f"batch: {batch}")
    audio = batch['audio_path']

    # compute log-Mel input features from input audio array
    # (128, 3000), float32 for a 30s audio
    input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    input_length=len(audio['array'])/audio['sampling_rate']

    text_prompt="please answer the following question by choosing the correct answer from the choices; The question is : " \
        +batch["question"]+" The choices are "+', '.join(batch["choices"])

    # Tokenize text
    text = batch['answer']
    # could also pad here, but the input batch contains only 1 data, so pointless
    # text_tokens = tokenizer(text, return_tensors='pt', padding='longest', truncation=True)
    text_tokens = tokenizer(text)

    logging.debug(f"input_features: {input_features.shape}, dtype: {input_features.dtype}")
    logging.debug(f"text: {text}, input_ids: {len(text_tokens['input_ids'])}")
    logging.debug(f"text_prompt: {text_prompt}")
    logging.debug(f"attention_mask: {len(text_tokens['attention_mask'])}")

    return {
        'input_features': input_features,  # Use 'input_features' to match training pipeline
        'labels': text_tokens['input_ids'],  # Use 'labels' to match training pipeline  

        'input_length': input_length,
        'attention_mask': text_tokens['attention_mask'],
        'text': text,
        'text_prompt': text_prompt
    }


if __name__ == '__main__':

    logging.info("loading feature extractor")
    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("/mnt/models/whisper-large-v3")
    logging.info("loading tokenizer")
    llm_tokenizer=AutoTokenizer.from_pretrained("/mnt/models/Qwen2.5-7B-Instruct/")

    parquets_path='/mnt/datasets/mmau-pro/'
    output_dataset_path = '/mnt/processed_datasets/mmau-pro/'
    
    # Load evaluation dataset
    logging.info(f"loading datase from {parquets_path}")
    tmp_dataset=load_dataset(parquets_path)
    logging.info(f"dataset loaded from {parquets_path}")

    test_ds=tmp_dataset['test']

    filtered_ds=test_ds.filter(filter_single_audio_path)
    prefixed_ds=filtered_ds.map(update_audio_paths)
    flattened_ds=prefixed_ds.map(flatten_list_to_string)

    final_ds = flattened_ds.cast_column("audio_path", Audio(sampling_rate=16000))

    logging.info(f"dataset filtered and mapped")
    logging.info(f"final_ds[0]: {final_ds[0]}")

    def _prepare(batch):
        # Use whisper_encoder for proper audio feature extraction
        # https://huggingface.co/datasets/openslr/librispeech_asr
        return prepare_dataset(batch, whisper_feature_extractor, llm_tokenizer, 16000)

    final_ds = final_ds.map(_prepare, remove_columns=final_ds.column_names, num_proc=200)

    final_ds.save_to_disk(output_dataset_path)
    logging.info(f"Dataset successfully created and saved to {output_dataset_path}")
    logging.info(f"Number of examples: {len(final_ds)}")
    logging.info(f"First example: {final_ds[0]}")