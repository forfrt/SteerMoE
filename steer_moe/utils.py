import torch
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Audio
import tqdm


def load_balancing_loss(gating_scores):
    """
    Compute the auxiliary load-balancing loss for MoE router.
    Encourages uniform usage of all experts.
    Args:
        gating_scores: (batch, seq_len, num_experts) softmax outputs from router
    Returns:
        Scalar load balancing loss (torch.Tensor)
    """
    # Average over batch and sequence
    expert_usage = gating_scores.mean(dim=(0, 1))  # (num_experts,)
    # Target is uniform distribution
    num_experts = gating_scores.size(-1)
    target = torch.full_like(expert_usage, 1.0 / num_experts)
    # KL divergence (can use other metrics)
    loss = torch.nn.functional.kl_div(
        expert_usage.log(), target, reduction="batchmean"
    )

    return loss


def load_parquet_datasets(parquet_dirs):
    """
    Load and concatenate datasets from a list of parquet directories.
    Returns a DatasetDict with 'train' split.
    """
    print('start loading dataset')
    dataset = DatasetDict()
    train_datasets = []
    for folder in tqdm.tqdm(parquet_dirs):
        tmp = load_from_disk(folder)
        train_datasets.append(tmp)
    concat = train_datasets[0]
    for tmp_dataset in tqdm.tqdm(train_datasets[1:]):
        concat = concatenate_datasets([concat, tmp_dataset])
    dataset['train'] = concat
    return dataset


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
        'audio_features': audio_features.squeeze(0),  # (seq_len, feature_dim)
        'labels': text_tokens['input_ids'].squeeze(0),
        'attention_mask': text_tokens['attention_mask'].squeeze(0)
    }
