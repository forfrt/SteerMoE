
import os
import tqdm
import datasets
import traceback
from datasets import Dataset, Audio, load_dataset, Features, Value, ClassLabel
from transformers import WhisperFeatureExtractor, AutoTokenizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

SAMPLING_RATE = 16000

# This function remains largely the same, as the fix is in create_clothoaqa_dataset
def prepare_dataset(batch,  feature_extractor, tokenizer, sampling_rate):
    # The 'audio_path' column will now contain the loaded audio data (a dict with 'array', 'sampling_rate', 'path')
    logging.info(f"batch: {batch}")
    audio = batch['audio_path']

    # compute log-Mel input features from input audio array
    # (128, 3000), float32 for a 30s audio
    # Ensure WhisperFeatureExtractor expects a numpy array as input, not a list of paths
    input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    input_length=len(audio['array'])/audio['sampling_rate']
    logging.info(f"input_length: {input_length}") # Too verbose for DEBUG in map

    # text_prompt="please answer the question in the previous audio. The question is : " + batch["QuestionText"]
    text_prompt="please answer the question in the previous audio: "
    logging.info(f"text_prompt: {text_prompt}") # Too verbose for DEBUG in map

    # Tokenize text
    text = batch['Answer'] # 'answer' is a string or list of strings if batched
    # If `batch` is truly a single example, `text` is a string. If `batch` contains multiple examples (batched=True, default for num_proc > 1)
    # then `batch['answer']` would be a list of strings. AutoTokenizer can handle both.
    text_tokens = tokenizer(text) # Added truncation/padding for consistency

    logging.info(f"input_features shape: {input_features.shape}, dtype: {input_features.dtype}")
    logging.info(f"text: {text}, input_ids len: {len(text_tokens['input_ids']) if isinstance(text_tokens['input_ids'], list) else text_tokens['input_ids'].shape}")
    logging.info(f"attention_mask len: {len(text_tokens['attention_mask']) if isinstance(text_tokens['attention_mask'], list) else text_tokens['attention_mask'].shape}")

    return {
        'input_features': input_features,
        'labels': text_tokens['input_ids'],
        'input_length': input_length,
        'attention_mask': text_tokens['attention_mask'],
        'text': text,
        'text_prompt': text_prompt
    }


def create_openaudiobench_dataset(csv_path, output_dataset_path, audio_base_path, whisper_feature_extractor, llm_tokenizer):
    logging.info(f"Loading dataset from {csv_path}")
    initial_features = Features({
        'Questions': Value('string'),
        'Answer': Value('string'),
        'audio_filename': Value('string'),
    })

    raw_ds = Dataset.from_csv(csv_path, features=initial_features)
    logging.info(f"Dataset loaded from {csv_path}. First example: {raw_ds[0]}")

    renamed_ds = raw_ds.rename_column('audio_filename', 'audio_path')
    logging.info(f"Dataset after renaming 'file_name' to 'audio_path'. First example: {renamed_ds[0]}")

    def prefix_audio_path(example):
        example['audio_path'] = os.path.join(audio_base_path, example['audio_path'])
        return example

    prefixed_ds = renamed_ds.map(prefix_audio_path)
    logging.info(f"Dataset after prefixing 'audio_path'. First example: {prefixed_ds[0]}")

    # --- NEW STEP: Filter out rows with non-existent audio files ---
    logging.info("Filtering dataset to keep only rows with existing audio files...")
    initial_num_rows = len(prefixed_ds)

    def check_audio_file_exists(example):
        return os.path.exists(example['audio_path'])

    # Use `num_proc` for parallel checking if you have many files
    # Make sure your AUDIO_BASE_PATH is accessible by all worker processes
    ds_with_existing_audio = prefixed_ds.filter(check_audio_file_exists, num_proc=os.cpu_count())

    num_rows_after_filter = len(ds_with_existing_audio)
    logging.info(f"Filtered out {initial_num_rows - num_rows_after_filter} rows due to missing audio files.")
    logging.info(f"Dataset size after filtering for existing audio files: {num_rows_after_filter}")
    if num_rows_after_filter == 0:
        logging.error("No audio files found after filtering. Aborting processing.")
        return # Or raise an error

    # First, let's create a temporary dataset with the Audio feature to get durations
    # This might still raise FileNotFoundError if the paths are truly wrong or permissions are an issue
    # But we've filtered for existence, so it should be fine.
    try:
        temp_audio_ds = ds_with_existing_audio.cast_column("audio_path", Audio(sampling_rate=SAMPLING_RATE))
        logging.info("Temporarily casted 'audio_path' to Audio feature to get durations.")
        logging.info(f"Dataset after casting 'audio_path' to Audio feature. First example: {temp_audio_ds[0]}")

        def filter_by_duration(example):
            audio_array = example['audio_path']['array']
            sr = example['audio_path']['sampling_rate']
            duration = len(audio_array) / sr
            return duration <= MAX_AUDIO_DURATION_SECONDS # MAX_AUDIO_DURATION_SECONDS should be defined globally

        initial_num_rows_duration = len(temp_audio_ds)
        ds_filtered_by_duration = temp_audio_ds.filter(filter_by_duration, num_proc=os.cpu_count())
        num_rows_after_duration_filter = len(ds_filtered_by_duration)
        logging.info(f"Filtered out {initial_num_rows_duration - num_rows_after_duration_filter} rows due to audio > {MAX_AUDIO_DURATION_SECONDS}s.")
        logging.info(f"Dataset size after filtering by duration: {num_rows_after_duration_filter}")
        logging.info(f"Dataset after filtering by duration. First example: {ds_filtered_by_duration[0]}")

        # Now, revert 'audio_path' column to just strings for final_ds
        # We need to explicitly put the original string paths back so that the
        # subsequent _prepare map function, which expects the Audio feature, can
        # re-load them or the datasets library will keep the Audio feature.
        # It's cleaner to re-cast after filtering by duration.
        # Or, just continue with ds_filtered_by_duration and let _prepare use its Audio feature.
        # Let's directly proceed with the Audio feature in `ds_filtered_by_duration`.
        final_ds_for_map = ds_filtered_by_duration

    except FileNotFoundError as e:
        logging.error(f"FATAL ERROR during temporary audio loading for duration check: {e}")
        logging.error("Ensure all audio files are present and accessible. Aborting processing.")
        return
    except Exception as e: # Catch other potential loading errors
        logging.error(f"An unexpected error occurred during temporary audio loading for duration check: {e}")
        logging.error("Aborting processing.")
        return


    # Use the _prepare wrapper for the map function
    def _prepare_wrapper(batch):
        # The 'audio_path' in 'batch' here will already be the Audio feature (dict with 'array', 'sampling_rate', 'path')
        # due to the previous `cast_column` and `filter` operations.
        return prepare_dataset(batch, whisper_feature_extractor, llm_tokenizer, SAMPLING_RATE)

    logging.info(f"Starting final mapping with _prepare function. Number of processes: 200")


    final_processed_ds = final_ds_for_map.map(
        _prepare_wrapper,
        remove_columns=final_ds_for_map.column_names, # Remove original columns
        num_proc=200,
    )

    final_processed_ds.save_to_disk(output_dataset_path)
    logging.info(f"Dataset successfully created and saved to {output_dataset_path}")
    logging.info(f"Number of examples: {len(final_processed_ds)}")
    logging.info(f"First example: {final_processed_ds[0]}")

if __name__ == '__main__':
    # Define MAX_AUDIO_DURATION_SECONDS here as it's used globally
    MAX_AUDIO_DURATION_SECONDS = 30

    logging.info("Loading feature extractor")
    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("/mnt/models/whisper-large-v3")
    logging.info("Loading tokenizer")
    llm_tokenizer=AutoTokenizer.from_pretrained("/mnt/models/Qwen2.5-7B-Instruct/")

    llama_csv_path='/mnt/datasets/OpenAudioBench/eval_datas/llama_questions/llama_questions.csv'
    llama_audio_base_path = '/mnt/datasets/OpenAudioBench/eval_datas/llama_questions/audios/'
    output_llama_ds = '/mnt/processed_datasets/OpenAudioBench/llama_questions/'

    create_openaudiobench_dataset(llama_csv_path, output_llama_ds, llama_audio_base_path, whisper_feature_extractor, llm_tokenizer)