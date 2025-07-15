"""
 * author Ruitao Feng
 * created on 10-07-2025
 * github: https://github.com/forfrt
"""

import torch
import torch.nn as nn
import librosa # For audio loading
from transformers import WhisperModel, WhisperProcessor, GPT2LMHeadModel, GPT2Tokenizer

# Audio → Encoder → Projection → [Audio_Prompt + Text_Tokens] → LLM → Output
class AudioToText(nn.Module):
    def __init__(self, encoder, decoder, encoder_dim, decoder_dim, use_adapter=True):
        super(AudioToText, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_adapter = use_adapter

        if use_adapter:
            self.prompt_proj = nn.Linear(encoder_dim, decoder_dim)
        else:
            assert encoder_dim == decoder_dim, "Dims must match if no adapter used"

    def forward(self, input_features, decoder_input_ids, labels=None):
        # (B, T_enc_raw, D_enc) -> (B, T_enc, D_enc)
        # Whisper encoder downsamples time, e.g., T_enc_raw=3000 -> T_enc=1500
        encoder_hidden_states = self.encoder(input_features).last_hidden_state
        
        if self.use_adapter:
            prompts = self.prompt_proj(encoder_hidden_states)
        else:
            prompts = encoder_hidden_states

        # Get token embeddings from Decoder (B, T_dec, D_dec)
        # Assuming decoder is like LLaMA or GPT2 where .model access internal layers
        if hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'embed_tokens'): # LLaMA-like
            input_embeds = self.decoder.model.embed_tokens(decoder_input_ids)
        elif hasattr(self.decoder, 'transformer') and hasattr(self.decoder.transformer, 'wte'): # GPT-2 like
            input_embeds = self.decoder.transformer.wte(decoder_input_ids)
        else:
            raise ValueError("Decoder embedding method not found. Adapt this part.")

        inputs_embeds = torch.cat([prompts, input_embeds], dim=1)

        if labels is not None:
            prompt_len = prompts.size(1)
            # Adjust labels to align with token positions only
            # We mask the prompt part with -100 (ignored in loss)
            labels = torch.cat([
                labels.new_full((labels.size(0), prompt_len), -100),
                labels
            ], dim=1)

        # When labels are provided, decoder typically computes loss internally
        # When labels are None (inference), it returns logits
        output = self.decoder(
            inputs_embeds=inputs_embeds,
            labels=labels
        )
        return output
    
    def get_device(self):
        return next(self.encoder.parameters()).device

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            # This can happen if the module has no parameters, e.g., an empty nn.Sequential
            # Fallback to checking encoder or decoder if top-level has no params
            if hasattr(self, 'encoder') and self.encoder:
                return next(self.encoder.parameters()).device
            if hasattr(self, 'decoder') and self.decoder:
                return next(self.decoder.parameters()).device
            return torch.device("cpu") # Default fallback


# --- Main ASR Example Script ---
def main_asr_example():
    # 1. Setup device
    device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else \
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # 2. Load pre-trained models and processor/tokenizer
    # Using whisper-large-v2 for encoder_dim=1280
    whisper_model_name = "openai/whisper-large-v2"
    whisper_model = WhisperModel.from_pretrained(whisper_model_name)
    processor = WhisperProcessor.from_pretrained(whisper_model_name) # For audio processing

    # Using gpt2 for decoder_dim=768
    gpt2_model_name = "gpt2"
    gpt2_decoder = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

    # GPT-2 needs a pad token for batching, use EOS token
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_decoder.config.pad_token_id = gpt2_decoder.config.eos_token_id

    # 3. Instantiate your AudioToText model
    # Ensure dimensions match what you specified.
    # whisper-large-v2 has d_model = 1280
    # gpt2 (base) has d_model = 768
    encoder_dim = whisper_model.config.d_model
    decoder_dim = gpt2_decoder.config.n_embd

    audio2text = AudioToText(
        encoder=whisper_model.encoder,
        decoder=gpt2_decoder,
        encoder_dim=encoder_dim,    # 1280 for whisper-large-v2
        decoder_dim=decoder_dim,    # 768 for gpt2 base
        use_adapter=True            # Must be True if dims don't match
    )
    
    # The prompt had audio2text.config = whisper.config, let's replicate roughly
    # This isn't strictly used by AudioToText forward, but good practice if underlying models need it
    audio2text.config = whisper_model.config 

    audio2text.to(device)
    audio2text.eval() # Set to evaluation mode

    # 4. Load and preprocess an example audio file
    # You can use any WAV file. For this example, let's generate a dummy one if none is found.
    try:
        # Replace with a path to your audio file
        audio_path = "audio_sample.wav" # e.g., a short speech recording
        if not os.path.exists(audio_path): # If file doesn't exist, create a dummy one
             print(f"Audio file {audio_path} not found. Creating a dummy audio signal.")
             sample_rate = 16000
             duration = 5 # seconds
             frequency = 440 # Hz (A4 note)
             t = np.linspace(0, duration, int(sample_rate * duration), False)
             audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
             # Save it using soundfile (you might need to install it: pip install soundfile)
             import soundfile as sf
             sf.write(audio_path, audio_data, sample_rate)
             print(f"Dummy audio file created: {audio_path}")
        else:
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Could not load or create audio file: {e}")
        print("Using a simple sine wave as a fallback.")
        sample_rate = 16000
        audio_data = 0.5 * torch.sin(2 * torch.pi * 440 * torch.arange(0, 5 * sample_rate, dtype=torch.float32) / sample_rate)
        audio_data = audio_data.numpy()


    # Process audio to get input_features (log-Mel spectrogram)
    inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device) # This is what your model expects

    print(f"Shape of input_features: {input_features.shape}") # e.g., (1, 80, 3000) for Whisper

    # 5. Autoregressive Generation for ASR
    # The decoder needs to start with a Begin-Of-Sequence (BOS) token
    # For GPT-2, this is often gpt2_tokenizer.bos_token_id
    # Ensure the decoder_input_ids starts with the BOS token ID for GPT-2.
    # GPT2's actual BOS token is <|endoftext|> which is also its EOS token.
    decoder_start_token_id = gpt2_tokenizer.bos_token_id if gpt2_tokenizer.bos_token_id is not None else gpt2_tokenizer.eos_token_id
    
    # Start with a batch size of 1, and the BOS token
    decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long).to(device)

    max_output_length = 100 # Max number of tokens to generate
    generated_tokens = []

    print("\nStarting generation...")
    print("NOTE: Output will likely be gibberish as the adapter is not trained.")

    with torch.no_grad():
        # First, get the audio prompts
        encoder_hidden_states = audio2text.encoder(input_features).last_hidden_state
        if audio2text.use_adapter:
            prompts = audio2text.prompt_proj(encoder_hidden_states)
        else:
            prompts = encoder_hidden_states

        for _ in range(max_output_length):
            # Get embeddings for current decoder_input_ids
            if hasattr(audio2text.decoder, 'model') and hasattr(audio2text.decoder.model, 'embed_tokens'): # LLaMA-like
                current_input_embeds = audio2text.decoder.model.embed_tokens(decoder_input_ids)
            elif hasattr(audio2text.decoder, 'transformer') and hasattr(audio2text.decoder.transformer, 'wte'): # GPT-2 like
                current_input_embeds = audio2text.decoder.transformer.wte(decoder_input_ids)
            else:
                raise ValueError("Decoder embedding method not found in generation loop.")

            # Prepend audio prompts
            inputs_embeds = torch.cat([prompts, current_input_embeds], dim=1)
            
            # Pass to decoder
            outputs = audio2text.decoder(inputs_embeds=inputs_embeds)
            logits = outputs.logits # Shape: (batch_size, seq_len, vocab_size)

            # Get the logits for the very last token position
            next_token_logits = logits[:, -1, :]
            
            # Greedy decoding: pick the token with the highest probability
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Append predicted token to decoder_input_ids for the next step
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)
            
            generated_tokens.append(next_token_id.item())

            # Stop if EOS token is generated
            if next_token_id.item() == gpt2_tokenizer.eos_token_id:
                print("EOS token generated.")
                break
        
        # Remove the initial BOS token from the generated sequence for decoding text
        if generated_tokens and generated_tokens[0] == decoder_start_token_id and len(decoder_input_ids[0]) > 1:
             # If we started with BOS and generated more, decode from the second token
             # This is slightly tricky because decoder_input_ids has the BOS from the start
             # The `generated_tokens` list contains only the *newly* generated tokens
             # So, we can just decode `generated_tokens`
             pass # generated_tokens is already what we want to decode

    # 6. Decode the generated token IDs to text
    generated_text = gpt2_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\n--- ASR Output ---")
    print(f"Generated token IDs: {generated_tokens}")
    print(f"Generated text: {generated_text}")
    print("--------------------")
    print("Reminder: The model (especially the adapter) needs fine-tuning for meaningful output.")

if __name__ == '__main__':
    # For dummy audio generation
    import numpy as np
    import os
    main_asr_example()