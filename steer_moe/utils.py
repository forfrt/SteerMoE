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
        'audio_features': audio_features.squeeze(0),  # Remove batch dimension
        'input_ids': text_tokens['input_ids'].squeeze(0),
        'attention_mask': text_tokens['attention_mask'].squeeze(0),
        'text': text
    }


@dataclass
class DataCollatorSpeechSeqSeqWithPadding:
    """
    Enhanced data collator for SteerMoE model that handles audio and text inputs with optional textual prompts.
    """
    tokenizer: Any
    textual_prompt: Optional[str] = None
    max_length: int = 512
    audio_column: str = "audio_features"
    text_column: str = "input_ids"
    return_attention_mask: bool = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features for SteerMoE training.
        
        Args:
            features: List of dictionaries with audio_features, input_ids, etc.
            
        Returns:
            Batch dictionary with padded tensors
        """
        batch_size = len(features)
        
        # Handle audio features (waveforms)
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
        
        # Pad audio features to same length
        if audio_features:
            max_audio_len = max(feat.size(-1) for feat in audio_features if feat.numel() > 0)
            padded_audio = []
            for feat in audio_features:
                if feat.numel() == 0:
                    # Handle empty audio
                    padded_feat = torch.zeros(max_audio_len, dtype=torch.float32)
                else:
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(0)  # Add batch dimension if needed
                    current_len = feat.size(-1)
                    if current_len < max_audio_len:
                        # Pad
                        pad_length = max_audio_len - current_len
                        padded_feat = torch.nn.functional.pad(feat, (0, pad_length))
                    else:
                        # Truncate
                        padded_feat = feat[..., :max_audio_len]
                padded_audio.append(padded_feat)
            batch_audio = torch.stack(padded_audio)
        else:
            batch_audio = torch.empty(batch_size, 0, dtype=torch.float32)
        
        # Handle text input_ids and create decoder_input_ids
        input_ids = []
        labels = []
        
        for feature in features:
            if self.text_column in feature:
                text_ids = feature[self.text_column]
                if isinstance(text_ids, torch.Tensor):
                    text_ids = text_ids.squeeze()
                else:
                    text_ids = torch.tensor(text_ids, dtype=torch.long).squeeze()
                
                # Create decoder input and labels
                if self.textual_prompt is not None:
                    # Tokenize the textual prompt
                    prompt_tokens = self.tokenizer.encode(
                        self.textual_prompt, 
                        add_special_tokens=False,
                        return_tensors="pt"
                    ).squeeze(0)
                    
                    # Combine prompt + text for decoder input
                    decoder_input = torch.cat([prompt_tokens, text_ids])
                    
                    # Labels should be the text only (prompt gets -100 in the model)
                    label = text_ids.clone()
                else:
                    # No prompt, use text directly
                    decoder_input = text_ids.clone()
                    label = text_ids.clone()
                
                input_ids.append(decoder_input)
                labels.append(label)
            else:
                # No text provided - create empty tensors
                input_ids.append(torch.tensor([], dtype=torch.long))
                labels.append(torch.tensor([], dtype=torch.long))
        
        # Pad text sequences
        if input_ids and any(len(ids) > 0 for ids in input_ids):
            # Pad decoder_input_ids
            max_text_len = min(max(len(ids) for ids in input_ids if len(ids) > 0), self.max_length)
            padded_input_ids = []
            padded_labels = []
            attention_masks = []
            
            for i, (input_seq, label_seq) in enumerate(zip(input_ids, labels)):
                if len(input_seq) == 0:
                    # Empty sequence
                    padded_input = torch.full((max_text_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                    padded_label = torch.full((max_text_len,), -100, dtype=torch.long)
                    attention_mask = torch.zeros(max_text_len, dtype=torch.long)
                else:
                    # Truncate if too long
                    if len(input_seq) > max_text_len:
                        input_seq = input_seq[:max_text_len]
                        label_seq = label_seq[:max_text_len]
                    
                    # Pad if too short
                    pad_length = max_text_len - len(input_seq)
                    padded_input = torch.cat([
                        input_seq,
                        torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                    ])
                    
                    label_pad_length = max_text_len - len(label_seq)
                    padded_label = torch.cat([
                        label_seq,
                        torch.full((label_pad_length,), -100, dtype=torch.long)
                    ])
                    
                    # Create attention mask
                    attention_mask = torch.cat([
                        torch.ones(len(input_seq), dtype=torch.long),
                        torch.zeros(pad_length, dtype=torch.long)
                    ])
                
                padded_input_ids.append(padded_input)
                padded_labels.append(padded_label)
                attention_masks.append(attention_mask)
            
            batch_input_ids = torch.stack(padded_input_ids)
            batch_labels = torch.stack(padded_labels)
            batch_attention_mask = torch.stack(attention_masks)
        else:
            # No text sequences
            batch_input_ids = torch.empty(batch_size, 0, dtype=torch.long)
            batch_labels = torch.empty(batch_size, 0, dtype=torch.long)
            batch_attention_mask = torch.empty(batch_size, 0, dtype=torch.long)
        
        # Create final batch
        batch = {
            "audio_waveform": batch_audio,
            "decoder_input_ids": batch_input_ids,
            "labels": batch_labels,
        }
        
        if self.return_attention_mask:
            batch["attention_mask"] = batch_attention_mask
            
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
