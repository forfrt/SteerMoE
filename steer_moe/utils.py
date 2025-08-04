"""
 * author Ruitao Feng
 * created on 16-07-2025
 * github: https://github.com/forfrt
"""

import torch
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Audio
import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s>%(funcName)s>%(lineno)d - %(message)s')

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
    input_features = feature_extractor(audio_path["array"], sampling_rate=audio_path["sampling_rate"]).input_features[0]
    input_length=len(audio_path['array'])/audio_path['sampling_rate']
    
    # Tokenize text
    text = batch[text_column]
    text_tokens = tokenizer(text, return_tensors='pt', padding='longest', truncation=True)

    return {
        'input_length': input_length,
        'input_features': input_features,  # Use 'input_features' to match training pipeline
        'labels': text_tokens['input_ids'].squeeze(0),  # Use 'labels' to match training pipeline  
        'attention_mask': text_tokens['attention_mask'].squeeze(0),
        'text': text
    }


def prepare_dataset_from_scatch(batch, audio_column, text_column, whisper_processor, tokenizer, sample_rate=16000):
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


@dataclass
class DataCollatorSpeechSeqSeqWithPadding:
    """
    Enhanced data collator for SteerMoE model that handles preprocessed audio features and labels.
    Works with preprocessed datasets that have 'input_features' and 'labels' columns.
    """
    feature_extractor: Any
    tokenizer: Any
    textual_prompt: Optional[str] = None
    max_length: int = 512
    audio_column: str = "input_features"  # Preprocessed audio features
    text_column: str = "labels"  # Preprocessed tokenized labels
    return_attention_mask: bool = False
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of preprocessed features for SteerMoE training.
        
        Args:
            features: List of dictionaries with input_features, labels, etc.
            
        Returns:
            Batch dictionary with padded tensors
        """
        batch_size = len(features)
        
        # Handle preprocessed audio features
        audio_features = []
        for feature in features:
            if self.audio_column in feature:
                audio_feat = feature[self.audio_column]
                if isinstance(audio_feat, torch.Tensor):
                    audio_features.append(audio_feat)
                else:
                    audio_features.append(torch.tensor(audio_feat, dtype=torch.float32))
            else:
                raise KeyError(f"Expected '{self.audio_column}' in features")
        
        # Pad audio features to same length (these are already processed Whisper features)
        audio_features=[{"input_features": audio_feat} for audio_feat in audio_features]
        if audio_features:
            batch_audio=self.feature_extractor.pad(audio_features, return_tensors="pt").input_features
        else:
            batch_audio = torch.empty(batch_size, 128, 3000, dtype=torch.float32)

        logging.info(f"batch_audio: {batch_audio.shape} {batch_audio.dtype} {batch_audio}")
        
        # Handle preprocessed labels (already tokenized)
        labels = []
        decoder_input_ids = []
        
        for feature in features:
            if self.text_column in feature:
                label_ids = feature[self.text_column]
                if isinstance(label_ids, torch.Tensor):
                    label_ids = label_ids.squeeze()
                else:
                    label_ids = torch.tensor(label_ids, dtype=torch.long).squeeze()

                logging.debug(f"label_ids: {label_ids.shape}, {label_ids.dtype}, {label_ids}")
                
                # Remove padding tokens and special tokens for processing
                if isinstance(label_ids, torch.Tensor):
                    # Remove padding tokens (-100 and pad_token_id)
                    valid_mask = (label_ids != -100) & (label_ids != self.tokenizer.pad_token_id)
                    if valid_mask.any():
                        clean_labels = label_ids[valid_mask]
                    else:
                        clean_labels = torch.tensor([], dtype=torch.long)
                else:
                    clean_labels = torch.tensor(label_ids, dtype=torch.long)

                logging.debug(f"clean_labels: {clean_labels.shape}, {clean_labels.dtype}, {clean_labels}")
                
                # Create decoder input with optional textual prompt
                if self.textual_prompt is not None and len(clean_labels) > 0:
                    # Tokenize the textual prompt
                    prompt_tokens = self.tokenizer.encode(
                        self.textual_prompt, 
                        add_special_tokens=False,
                        return_tensors="pt"
                    ).squeeze(0)

                    logging.debug(f"prompt_tokens: {prompt_tokens.shape}, {prompt_tokens.dtype}, {prompt_tokens}")
                    
                    # Combine prompt + clean labels for decoder input
                    decoder_input = torch.cat([prompt_tokens, clean_labels])
                    # decoder_input = prompt_tokens 
                    
                    # Labels should be the original clean labels only (prompt gets masked in model)
                    empty_prompt=torch.full_like(prompt_tokens, fill_value=-100)
                    label=torch.cat([empty_prompt, clean_labels])

                    # label = clean_labels.clone()
                else:
                    # No prompt, use clean labels directly
                    decoder_input = clean_labels.clone()
                    label = clean_labels.clone()
                
                logging.debug(f"decoder_input: {decoder_input.shape}, {decoder_input.dtype}, {decoder_input}")
                logging.debug(f"label: {label.shape}, {label.dtype}, {label}")
                decoder_input_ids.append(decoder_input)
                labels.append(label)
            else:
                # No labels provided - create empty tensors
                decoder_input_ids.append(torch.tensor([], dtype=torch.long))
                labels.append(torch.tensor([], dtype=torch.long))

        # Pad lables 
        label_features=[{"input_ids": input_ids} for input_ids in labels]
        if label_features:
            labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
            batch_labels = labels_batch["input_ids"]
            # batch_labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            # if (batch_labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            #     batch_labels = batch_labels[:, 1:]
        else:
            batch_labels = torch.empty(batch_size, 0, dtype=torch.long)

        logging.info(f"batch_labels: {batch_labels.shape}, {batch_labels.dtype}, {batch_labels}")

        # Pad prompt_tokens+labels
        input_features=[{"input_ids": input_ids} for input_ids in decoder_input_ids]
        if label_features:
            input_ids_batch = self.tokenizer.pad(input_features, return_tensors="pt")
            batch_input_ids = input_ids_batch["input_ids"]
            # batch_input_ids = labels_batch["input_ids"].masked_fill(input_ids_batch.attention_mask.ne(1), -100)
            # if (batch_input_ids[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            #     batch_input_ids = batch_input_ids[:, 1:]
        else:
            batch_input_ids = torch.empty(batch_size, 0, dtype=torch.long)

        logging.info(f"batch_input_ids: {batch_input_ids.shape}, {batch_input_ids.dtype}, {batch_input_ids}")

        
        # Create final batch - use different key name for audio to match model expectations
        batch = {
            "input_features": batch_audio,  # Model expects audio_waveform parameter
            "decoder_input_ids": batch_input_ids,
            "labels": batch_labels,
        }
        
        # if self.return_attention_mask:
        #     batch["attention_mask"] = batch_attention_mask
            
        return batch


@dataclass  
class DataCollatorSpeechSeqSeqWithPaddingLegacy:
    """
    Legacy data collator for compatibility with existing whisper training code.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
