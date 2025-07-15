export PATH=/root/autodl-nas/ruitao/SteerMoE:./:$PATH

# Training
python scripts/train_layer_wise.py --config configs/layer_wise.yaml --mode train

# Evaluation
# python scripts/train_layer_wise.py --config configs/layer_wise.yaml --mode eval --model_path ./results/final

# Analysis
# python scripts/train_layer_wise.py --config configs/layer_wise.yaml --mode analyze --model_path ./results/final
