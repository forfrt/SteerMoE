deepspeed --include localhost:1,2 train_lora.py --config configs/lora_whisper_qwen7b_libri_train.yaml \
    &> lora_whisper_qwen7b_libri_train.log 

deepspeed --include localhost:1,2 train_lora.py --config configs/lora_whisper_qwen7b_libri_test.yaml \
    &> lora_whisper_qwen7b_libri_test.log &