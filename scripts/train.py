import yaml
import torch
from steer_moe.models import SteerMoEModel, SteerMoEHybridModel
from steer_moe.aligner import SteerMoEAligner
from steer_moe.utils import load_balancing_loss, load_parquet_datasets, prepare_dataset
from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import DataLoader
from datasets import Audio, load_dataset, concatenate_datasets, DatasetDict, load_metric, load_from_disk
import tqdm
import os


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_asr_dataset_by_name(dataset_name, split='test', sample_rate=16000):
    """
    Load a single ASR dataset by name and split.
    """
    # Map friendly names to HuggingFace datasets
    dataset_map = {
        'aishell1': ('speech_asr/aishell', 'default'),
        'aishell2': ('speech_asr/aishell2', 'default'),
        'ws_net': ('ws_net', None),
        'ws_meeting': ('ws_meeting', None),
        'kespeech': ('kespeech', None),
        'librispeech_test_clean': ('librispeech_asr', 'clean'),
        'librispeech_test_other': ('librispeech_asr', 'other'),
    }
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    ds_name, ds_config = dataset_map[dataset_name]
    if ds_config:
        dataset = load_dataset(ds_name, ds_config, split=split)
    else:
        dataset = load_dataset(ds_name, split=split)
    return dataset

def load_parquet_datasets_for_steermoe(parquet_dirs):
    """
    Load and concatenate datasets from parquet directories, similar to main_word_correct_clips.py.
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
    print(f"Total samples: {len(concat)}")
    return dataset

def filter_dataset_by_length(dataset, max_audio_length=30.0, max_text_length=448):
    """
    Filter dataset by audio length and text length, similar to main_word_correct_clips.py.
    """
    max_input_length = max_audio_length * 16000
    
    def filter_inputs(input_length):
        """Filter inputs with zero input length or longer than max_audio_length""" 
        return 0 < input_length < max_input_length

    def filter_labels(labels):
        """Filter label sequences longer than max length""" 
        return len(labels) < max_text_length

    print(f"Dataset before filter: {dataset}")
    # filter by audio length
    if "input_length" in dataset['train'].column_names:
        dataset = dataset.filter(filter_inputs, input_columns=["input_length"])
    # filter by label length
    if "labels" in dataset['train'].column_names:
        dataset = dataset.filter(filter_labels, input_columns=["labels"])
    print(f"Dataset after filter: {dataset}")
    return dataset

def compute_metrics(pred):
    """
    Compute CER and WER for evaluation.
    """
    cer_metric = load_metric('cer')
    wer_metric = load_metric('wer')
    pred_str = pred.predictions
    label_str = pred.label_ids
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer, "wer": wer}

def prepare_asr_dataset(dataset, audio_column, text_column, whisper_encoder, tokenizer, sample_rate=16000):
    def _prepare(batch):
        return prepare_dataset(batch, audio_column, text_column, whisper_encoder, tokenizer, sample_rate)
    # Cast audio column to datasets.Audio if needed
    if not isinstance(dataset.features[audio_column], Audio):
        dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sample_rate))
    processed = dataset.map(_prepare, remove_columns=dataset.column_names)
    return processed

# Advanced HuggingFace save/load utilities

def save_trainer_model(trainer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    if trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(output_dir)
    # Save Trainer state (including optimizer, scheduler, RNG)
    trainer.state.save_to_json(os.path.join(output_dir, 'trainer_state.json'))
    print(f"Trainer model, tokenizer, and state saved to {output_dir}")

def load_trainer_model(model_class, model_dir, tokenizer_class=None):
    model = model_class.from_pretrained(model_dir)
    tokenizer = None
    if tokenizer_class is not None:
        tokenizer = tokenizer_class.from_pretrained(model_dir)
    print(f"Model loaded from {model_dir}")
    return model, tokenizer

def save_optimizer_state(trainer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(trainer.optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
    print(f"Optimizer state saved to {output_dir}/optimizer.pt")

def load_optimizer_state(trainer, optimizer_path):
    trainer.optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
    print(f"Optimizer state loaded from {optimizer_path}")

# HuggingFace Hub integration
from huggingface_hub import HfApi, create_repo, upload_folder

def push_to_hub(output_dir, repo_name, token=None, private=True):
    api = HfApi()
    create_repo(repo_id=repo_name, token=token, private=private, exist_ok=True)
    upload_folder(repo_id=repo_name, folder_path=output_dir, token=token)
    print(f"Model pushed to HuggingFace Hub: {repo_name}")

def load_from_hub(model_class, repo_name, subfolder=None, revision=None, tokenizer_class=None):
    model = model_class.from_pretrained(repo_name, subfolder=subfolder, revision=revision)
    tokenizer = None
    if tokenizer_class is not None:
        tokenizer = tokenizer_class.from_pretrained(repo_name, subfolder=subfolder, revision=revision)
    print(f"Model loaded from HuggingFace Hub: {repo_name}")
    return model, tokenizer

def train_with_deepspeed(config_path='configs/default.yaml', deepspeed_config_path='configs/deepspeed_config.json',
                         eval_dataset_name=None, custom_test_set_path=None, resume_from_checkpoint=None,
                         model_type='steermoe', use_hybrid=False):
    """
    Train SteerMoE model with DeepSpeed support.
    
    Args:
        model_type: 'steermoe' or 'hybrid'
        use_hybrid: If True, use SteerMoEHybridModel instead of SteerMoEModel
    """
    config = load_config(config_path)
    whisper_encoder = WhisperEncoder(config['whisper_encoder']['model_path'])
    llm_decoder = AutoModelForCausalLM.from_pretrained(config['llm_decoder']['model_name'])
    llm_decoder.eval()
    for p in llm_decoder.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(config['llm_decoder']['model_name'])
    aligner = SteerMoEAligner(
        feature_dim=config['aligner']['feature_dim'],
        num_experts=config['aligner']['num_experts']
    )

    # Choose model type
    if use_hybrid:
        model = SteerMoEHybridModel(
            whisper_encoder=whisper_encoder,
            aligner=aligner,
            llm_decoder=llm_decoder,
            max_prompt_tokens=config.get('max_prompt_tokens', 512),
            use_adapter=config.get('use_adapter', True)
        )
    else:
        model = SteerMoEModel(whisper_encoder, aligner, llm_decoder)

    model.train()
    parquet_dirs = config.get('parquet_dirs', [])
    audio_column = config.get('audio_column', 'audio')
    text_column = config.get('text_column', 'text')
    sample_rate = config.get('sample_rate', 16000)
    batch_size = config['training']['batch_size']
    
    # Load dataset using the new function
    dataset = load_parquet_datasets_for_steermoe(parquet_dirs)
    
    # Filter dataset if needed
    if config.get('filter_dataset', True):
        dataset = filter_dataset_by_length(
            dataset, 
            max_audio_length=config.get('max_audio_length', 30.0),
            max_text_length=config.get('max_text_length', 448)
        )
    
    if not isinstance(dataset['train'].features[audio_column], Audio):
        dataset['train'] = dataset['train'].cast_column(audio_column, Audio(sampling_rate=sample_rate))
    
    def _prepare(batch):
        return prepare_dataset(batch, audio_column, text_column, whisper_encoder, tokenizer, sample_rate)
    
    processed_dataset = dataset['train'].map(_prepare, remove_columns=dataset['train'].column_names)
    
    # Validation split for early stopping
    if 'validation' in dataset:
        processed_val = dataset['validation'].map(_prepare, remove_columns=dataset['validation'].column_names)
    else:
        # Optionally split train into train/val
        processed_dataset = processed_dataset.train_test_split(test_size=0.05, seed=42)
        processed_val = processed_dataset['test']
        processed_dataset = processed_dataset['train']
    
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=batch_size,
        num_train_epochs=config['training']['epochs'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        deepspeed=deepspeed_config_path,
        fp16=True,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=2,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    
    def compute_metrics_trainer(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        cer_metric = load_metric('cer')
        wer_metric = load_metric('wer')
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer, "wer": wer}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        eval_dataset=processed_val,
        compute_metrics=compute_metrics_trainer,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save best model using HuggingFace utilities
    save_trainer_model(trainer, './results/best_model_hf')
    save_optimizer_state(trainer, './results/best_model_hf')
    
    # Evaluate on specified ASR dataset
    if eval_dataset_name:
        print(f"Evaluating on {eval_dataset_name}")
        ds = get_asr_dataset_by_name(eval_dataset_name, split='test', sample_rate=sample_rate)
        processed = prepare_asr_dataset(ds, audio_column, text_column, whisper_encoder, tokenizer, sample_rate)
        results = trainer.evaluate(processed)
        print(f"Results for {eval_dataset_name}: {results}")
    
    # Evaluate on custom test set if provided
    if custom_test_set_path:
        print(f"Evaluating on custom test set: {custom_test_set_path}")
        custom_ds = load_from_disk(custom_test_set_path)
        processed_custom = custom_ds.map(_prepare, remove_columns=custom_ds.column_names)
        results = trainer.evaluate(processed_custom)
        print(f"Results for custom test set: {results}")

# Example: Inference using a loaded model and tokenizer

def example_inference(model_dir, audio_path, whisper_encoder_path, tokenizer_class=AutoTokenizer, max_length=64):
    # Load model and tokenizer from HuggingFace format
    model, tokenizer = load_trainer_model(SteerMoEModel, model_dir, tokenizer_class)
    model.eval()
    # Load Whisper encoder
    whisper_encoder = WhisperEncoder(whisper_encoder_path)
    # Load and process audio
    import soundfile as sf
    waveform, sr = sf.read(audio_path)
    waveform = torch.tensor(waveform, dtype=torch.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)
    waveform = waveform.unsqueeze(0)
    audio_features = whisper_encoder.tokenize_waveform(waveform)
    with torch.no_grad():
        h_aligned = model.aligner(audio_features)
        generated = model.llm_decoder.generate(inputs_embeds=h_aligned, max_length=max_length)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    print('Inference output:', decoded)
    return decoded

def main():
    # Example usage for different model types
    # For original SteerMoE:
    # train_with_deepspeed(use_hybrid=False)
    
    # For hybrid model:
    # train_with_deepspeed(use_hybrid=True)
    
    # For evaluation:
    # train_with_deepspeed(eval_dataset_name='librispeech_test_clean', use_hybrid=True)
    pass

if __name__ == '__main__':
    # To use distributed/deepspeed training:
    # train_with_deepspeed(eval_dataset_name='librispeech_test_clean', custom_test_set_path=None, resume_from_checkpoint=None, use_hybrid=True)
    # To push to HuggingFace Hub:
    # push_to_hub('./results/best_model_hf', 'your-hf-username/steermoe-demo', token='YOUR_HF_TOKEN')
    # To run example inference:
    # example_inference('./results/best_model_hf', 'path/to/audio.wav', 'path/to/whisper_encoder_dir')
    main()
