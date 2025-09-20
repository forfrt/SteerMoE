import os
import tqdm
import datasets
import traceback
from datasets import Dataset, Audio, load_dataset, Features, Value, ClassLabel
from transformers import WhisperFeatureExtractor, AutoTokenizer
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

AUDIO_BASE_PATH = '/mnt/datasets/ClothoAQA/audio_files/'
SAMPLING_RATE = 16000

def prepare_dataset(batch,  feature_extractor, tokenizer, sampling_rate):
    # logging.info(f"batch: {batch}")
    audio = batch['audio_path']

    # compute log-Mel input features from input audio array
    # (128, 3000), float32 for a 30s audio
    input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    input_length=len(audio['array'])/audio['sampling_rate']

    text_prompt="please answer the following question. The question is : " +batch["QuestionText"]
    logging.debug(f"list text_prompt: {text_prompt}")

    # Tokenize text
    text = batch['answer']
    # could also pad here, but the input batch contains only 1 data, so pointless
    # text_tokens = tokenizer(text, return_tensors='pt', padding='longest', truncation=True)
    text_tokens = tokenizer(text)

    logging.debug(f"input_features: {input_features.shape}, dtype: {input_features.dtype}")
    logging.debug(f"text: {text}, input_ids: {len(text_tokens['input_ids'])}")
    logging.debug(f"attention_mask: {len(text_tokens['attention_mask'])}")

    return {
        'input_features': input_features,  # Use 'input_features' to match training pipeline
        'labels': text_tokens['input_ids'],  # Use 'labels' to match training pipeline  

        'input_length': input_length,
        'attention_mask': text_tokens['attention_mask'],
        'text': text,
        'text_prompt': text_prompt
    }


def create_clothoaqa_dataset(csv_path, output_dataset_path, whisper_feature_extractor, llm_tokenizer):
    # Load evaluation dataset
    logging.info(f"loading datase from {csv_path}")
    initial_features = Features({
        'file_name': Value('string'),
        'QuestionText': Value('string'),
        'answer': Value('string'),
        'confidence': Value('string') # Could be ClassLabel if fixed categories
    })

    raw_ds = Dataset.from_csv(csv_path, features=initial_features)
    logging.info(f"dataset loaded from {csv_path}")
    logging.info("\n--- Original ClothoAQA Dataset (from CSV) ---")
    logging.info(raw_ds[0])

    renamed_ds = raw_ds.rename_column('file_name', 'audio_path')
    logging.info("\n--- Dataset after renaming 'file_name' to 'audio_path' ---")
    logging.info(renamed_ds[0])

    def prefix_audio_path(example):
        example['audio_path'] = os.path.join(AUDIO_BASE_PATH, example['audio_path'])
        return example

    prefixed_ds = renamed_ds.map(prefix_audio_path)

    logging.info("\n--- Dataset after prefixing 'audio_path' ---")
    logging.info(prefixed_ds[0])

    final_ds = prefixed_ds.cast_column("audio_path", Audio(sampling_rate=SAMPLING_RATE))
    logging.info("\n--- Final dataset after casting 'audio_path' to Audio feature ---")
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

if __name__ == '__main__':

    logging.info("loading feature extractor")
    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("/mnt/models/whisper-large-v3")
    logging.info("loading tokenizer")
    llm_tokenizer=AutoTokenizer.from_pretrained("/mnt/models/Qwen2.5-7B-Instruct/")

    train_csv_path='/mnt/datasets/ClothoAQA/clotho_aqa_train.csv'
    test_csv_path='/mnt/datasets/ClothoAQA/clotho_aqa_test.csv'
    val_csv_path='/mnt/datasets/ClothoAQA/clotho_aqa_val.csv'

    output_train_ds = '/mnt/processed_datasets/ClothoAQA/train/'
    output_test_ds = '/mnt/processed_datasets/ClothoAQA/test/'
    output_val_ds = '/mnt/processed_datasets/ClothoAQA/val/'

    create_clothoaqa_dataset(train_csv_path, output_train_ds, whisper_feature_extractor, llm_tokenizer)
    # create_clothoaqa_dataset(test_csv_path, output_test_ds, whisper_feature_extractor, llm_tokenizer)
    # create_clothoaqa_dataset(val_csv_path, output_val_ds, whisper_feature_extractor, llm_tokenizer)
