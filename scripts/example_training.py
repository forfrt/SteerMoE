#!/usr/bin/env python3
"""
Example training script for SteerMoE models.
This script demonstrates how to use the updated train.py functionality.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import train_with_deepspeed, load_parquet_datasets_for_steermoe, filter_dataset_by_length

def example_steermoe_training():
    """
    Example: Train the original SteerMoE model
    """
    print("=== Training Original SteerMoE Model ===")
    
    # Example parquet directories (replace with your actual paths)
    parquet_dirs = [
        "/path/to/your/dataset1",
        "/path/to/your/dataset2",
        # Add more dataset paths as needed
    ]
    
    # Update config with your dataset paths
    config_path = "configs/default.yaml"
    
    # Train with original SteerMoE model
    train_with_deepspeed(
        config_path=config_path,
        deepspeed_config_path="configs/deepspeed_config.json",
        eval_dataset_name="librispeech_test_clean",  # Optional: evaluate on standard dataset
        custom_test_set_path=None,  # Optional: path to custom test set
        resume_from_checkpoint=None,  # Optional: resume from checkpoint
        use_hybrid=False  # Use original SteerMoE model
    )

def example_hybrid_training():
    """
    Example: Train the SteerMoEHybridModel
    """
    print("=== Training SteerMoE Hybrid Model ===")
    
    # Example parquet directories (replace with your actual paths)
    parquet_dirs = [
        "/path/to/your/dataset1",
        "/path/to/your/dataset2",
        # Add more dataset paths as needed
    ]
    
    # Update config with your dataset paths
    config_path = "configs/default.yaml"
    
    # Train with hybrid model
    train_with_deepspeed(
        config_path=config_path,
        deepspeed_config_path="configs/deepspeed_config.json",
        eval_dataset_name="librispeech_test_clean",  # Optional: evaluate on standard dataset
        custom_test_set_path=None,  # Optional: path to custom test set
        resume_from_checkpoint=None,  # Optional: resume from checkpoint
        use_hybrid=True  # Use hybrid model
    )

def example_dataset_loading():
    """
    Example: Demonstrate dataset loading functionality
    """
    print("=== Dataset Loading Example ===")
    
    # Example parquet directories
    parquet_dirs = [
        "/path/to/your/dataset1",
        "/path/to/your/dataset2",
    ]
    
    # Load datasets
    dataset = load_parquet_datasets_for_steermoe(parquet_dirs)
    print(f"Loaded dataset with {len(dataset['train'])} samples")
    
    # Filter dataset by length
    filtered_dataset = filter_dataset_by_length(
        dataset,
        max_audio_length=30.0,
        max_text_length=448
    )
    print(f"After filtering: {len(filtered_dataset['train'])} samples")

def example_with_custom_config():
    """
    Example: Create and use a custom config
    """
    import yaml
    
    # Create custom config
    custom_config = {
        'whisper_encoder': {
            'model_path': '/root/autodl-nas/ruitao/Kimi-Audio/kimia_infer/models/tokenizer/whisper_Lv3',
            'freeze': True
        },
        'llm_decoder': {
            'model_name': 'gpt2',
            'freeze': True
        },
        'aligner': {
            'num_experts': 16,  # More experts for better performance
            'feature_dim': 1024
        },
        'max_prompt_tokens': 256,  # Shorter prompts for hybrid model
        'use_adapter': True,
        'parquet_dirs': [
            "/path/to/your/dataset1",
            "/path/to/your/dataset2",
        ],
        'audio_column': "audio",
        'text_column': "text",
        'sample_rate': 16000,
        'filter_dataset': True,
        'max_audio_length': 30.0,
        'max_text_length': 448,
        'training': {
            'batch_size': 16,
            'lr': 5e-4,
            'epochs': 5,
            'load_balance_loss_weight': 0.01
        }
    }
    
    # Save custom config
    with open('configs/custom.yaml', 'w') as f:
        yaml.dump(custom_config, f, default_flow_style=False)
    
    print("Custom config saved to configs/custom.yaml")
    
    # Train with custom config
    train_with_deepspeed(
        config_path="configs/custom.yaml",
        use_hybrid=True
    )

def main():
    """
    Main function to run examples
    """
    print("SteerMoE Training Examples")
    print("=" * 50)
    
    # Uncomment the example you want to run:
    
    # 1. Train original SteerMoE model
    # example_steermoe_training()
    
    # 2. Train hybrid model
    # example_hybrid_training()
    
    # 3. Demonstrate dataset loading
    # example_dataset_loading()
    
    # 4. Use custom config
    # example_with_custom_config()
    
    print("\nTo run examples, uncomment the desired function call in main().")
    print("Make sure to update the parquet directory paths with your actual dataset paths.")

if __name__ == "__main__":
    main() 