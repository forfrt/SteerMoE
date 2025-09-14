import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')


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

    logging.debug(f"input_features: {input_features.shape}, dtype: {input_features.dtype}")
    logging.debug(f"text: {text}, input_ids: {len(text_tokens['input_ids'])}")
    logging.debug(f"attention_mask: {len(text_tokens['attention_mask'])}")
    
    return {
        'input_features': input_features,  # Use 'input_features' to match training pipeline
        'labels': text_tokens['input_ids'],  # Use 'labels' to match training pipeline  

        'input_length': input_length,
        'attention_mask': text_tokens['attention_mask'],
        'text': text
    }
    
    
def prepare_dataset_for_conformer(batch, audio_column, text_column, tokenizer, sample_rate):
    import numpy as np
    """
    Map function to process a batch: loads audio, extracts features, tokenizes text.
    Args:
        batch: dict with audio and text columns
        audio_column: str, name of audio file path column
        text_column: str, name of text column
        tokenizer: HuggingFace tokenizer for text
    Returns:
        dict with processed audio features and tokenized text
    """
    # Load audio
    audio_path = batch[audio_column]

    input_features = np.array([i * 32768 for i in audio_path["array"]], dtype=np.int16)
    input_length=len(audio_path['array'])/audio_path['sampling_rate']
    
    # Tokenize text
    text = batch[text_column]
    text_tokens = tokenizer(text)
    
    logging.debug(f"input_features: {input_features.shape}, dtype: {input_features.dtype}")
    logging.debug(f"text: {text}, input_ids: {len(text_tokens['input_ids'])}")
    logging.debug(f"attention_mask: {len(text_tokens['attention_mask'])}")

    return {
        'input_length': input_length,
        'input_features': input_features,  # Use 'input_features' to match training pipeline
        'labels': text_tokens['input_ids'],  # Use 'labels' to match training pipeline  
        'attention_mask': text_tokens['attention_mask'],
        'sample_rate':sample_rate,
        'text': text
    }