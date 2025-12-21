deepspeed --include localhost:1,2 train_layer_wise.py --config ../configs/layer_wise_whisper_qwen7b_libri_train.yaml \
    &> layer_wise_whisper_qwen7b_libri_train_moe16.log 

deepspeed --include localhost:1,2 train_layer_wise.py --config ../configs/layer_wise_whisper_qwen7b_libri_test.yaml \
    &> layer_wise_whisper_qwen7b_libri_test_moe16.log &