import argparse
import os
import torch
import json
import csv
from tqdm import tqdm
from steer_moe.models import SteerMoEModel
from ..steer_moe.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, Audio, load_metric
from .train import load_trainer_model

def batch_infer(
    model_dir,
    whisper_encoder_path,
    input_list=None,
    input_dataset=None,
    audio_column='audio',
    text_column=None,
    tokenizer_class=AutoTokenizer,
    output_file='outputs.csv',
    max_length=64,
    compute_metrics=True
):
    # Load model and tokenizer
    model, tokenizer = load_trainer_model(SteerMoEModel, model_dir, tokenizer_class)
    model.eval()
    whisper_encoder = WhisperEncoder(whisper_encoder_path)
    # Prepare input data
    if input_list:
        audio_files = input_list
        references = None
    elif input_dataset:
        if input_dataset.endswith('.parquet') or os.path.isdir(input_dataset):
            dataset = load_from_disk(input_dataset)
        else:
            dataset = load_dataset(input_dataset)
        audio_files = [x[audio_column] for x in dataset]
        references = [x[text_column] for x in dataset] if text_column else None
    else:
        raise ValueError('Must provide input_list or input_dataset')
    # Inference loop
    results = []
    for idx, audio_path in enumerate(tqdm(audio_files)):
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
        result = {'audio': audio_path, 'prediction': decoded}
        if references:
            result['reference'] = references[idx]
        results.append(result)
    # Save outputs
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'Saved outputs to {output_file}')
    # Compute metrics if references are available
    if compute_metrics and references:
        cer_metric = load_metric('cer')
        wer_metric = load_metric('wer')
        preds = [r['prediction'] for r in results]
        refs = [r['reference'] for r in results]
        cer = cer_metric.compute(predictions=preds, references=refs)
        wer = wer_metric.compute(predictions=preds, references=refs)
        print(f'CER: {cer:.4f}, WER: {wer:.4f}')
    else:
        print('No references provided, skipping metric computation.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch inference/evaluation for SteerMoE')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to trained model directory')
    parser.add_argument('--whisper_encoder_path', type=str, required=True, help='Path to Whisper encoder directory')
    parser.add_argument('--input_list', type=str, nargs='*', help='List of audio file paths')
    parser.add_argument('--input_dataset', type=str, help='Path to dataset (parquet or HuggingFace format)')
    parser.add_argument('--audio_column', type=str, default='audio', help='Audio column name in dataset')
    parser.add_argument('--text_column', type=str, default=None, help='Text/reference column name in dataset')
    parser.add_argument('--output_file', type=str, default='outputs.csv', help='Output CSV file')
    parser.add_argument('--max_length', type=int, default=64, help='Max generation length')
    parser.add_argument('--no_metrics', action='store_true', help='Disable CER/WER computation')
    args = parser.parse_args()
    batch_infer(
        model_dir=args.model_dir,
        whisper_encoder_path=args.whisper_encoder_path,
        input_list=args.input_list,
        input_dataset=args.input_dataset,
        audio_column=args.audio_column,
        text_column=args.text_column,
        output_file=args.output_file,
        max_length=args.max_length,
        compute_metrics=not args.no_metrics
    ) 


"""
    python scripts/batch_infer.py \
  --model_dir ./results/best_model_hf \
  --whisper_encoder_path /path/to/whisper_encoder \
  --input_dataset /path/to/test_dataset \
  --audio_column audio \
  --text_column text \
  --output_file outputs.csv
"""

"""
python scripts/batch_infer.py \
  --model_dir ./results/best_model_hf \
  --whisper_encoder_path /path/to/whisper_encoder \
  --input_list file1.wav file2.wav file3.wav \
  --output_file outputs.csv \
  --no_metrics
"""