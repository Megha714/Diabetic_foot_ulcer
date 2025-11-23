"""
Configuration Management for DFU Training
"""

import yaml
import json
from pathlib import Path


class Config:
    """Training configuration"""
    
    # Default configuration
    DEFAULT_CONFIG = {
        # Model Architecture
        'model': {
            'type': 'hybrid',  # 'hybrid' or 'custom'
            'num_classes': 2,
            'pretrained': True,
            'rejection_threshold': 0.7,
            'image_size': 224,
        },
        
        # Dataset
        'data': {
            'data_dir': './DFU/Patches',
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'batch_size': 16,  # Optimized for 16GB RAM
            'num_workers': 4,
            'pin_memory': True,
        },
        
        # Training Hyperparameters
        'training': {
            'num_epochs': 50,
            'learning_rate': 1e-4,
            'min_lr': 1e-6,
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'warmup_epochs': 5,
        },
        
        # Loss Weights
        'loss': {
            'classification_weight': 1.0,
            'validation_weight': 0.5,
        },
        
        # Optimizer & Scheduler
        'optimizer': {
            'type': 'adamw',
            'betas': [0.9, 0.999],
            'eps': 1e-8,
        },
        
        'scheduler': {
            'type': 'cosine_warm_restarts',
            't0': 10,
            't_mult': 1,
        },
        
        # Regularization
        'regularization': {
            'dropout': 0.3,
            'label_smoothing': 0.1,
        },
        
        # Early Stopping
        'early_stopping': {
            'patience': 10,
            'min_delta': 0.001,
        },
        
        # Checkpointing
        'checkpoint': {
            'save_dir': './checkpoints',
            'save_frequency': 5,
            'keep_last_n': 3,
        },
        
        # Logging
        'logging': {
            'log_dir': './logs',
            'tensorboard': False,
            'wandb': False,
        },
        
        # Reproducibility
        'seed': 42,
        
        # Hardware
        'hardware': {
            'use_amp': False,  # MPS doesn't support mixed precision yet
            'device': 'auto',  # 'auto', 'mps', 'cuda', or 'cpu'
        }
    }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls.DEFAULT_CONFIG.copy()
        config.update(config_dict)
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path):
        """Load config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path):
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def save_yaml(cls, config, yaml_path):
        """Save config to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    @classmethod
    def save_json(cls, config, json_path):
        """Save config to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=4)


# Quick configuration presets
CONFIGS = {
    'fast': {
        'model': {'type': 'hybrid', 'pretrained': True},
        'data': {'batch_size': 32, 'num_workers': 4},
        'training': {'num_epochs': 20, 'learning_rate': 1e-3},
    },
    
    'balanced': {
        'model': {'type': 'hybrid', 'pretrained': True},
        'data': {'batch_size': 16, 'num_workers': 4},
        'training': {'num_epochs': 50, 'learning_rate': 1e-4},
    },
    
    'high_quality': {
        'model': {'type': 'custom', 'pretrained': False},
        'data': {'batch_size': 8, 'num_workers': 4},
        'training': {'num_epochs': 100, 'learning_rate': 5e-5},
    },
    
    'low_memory': {
        'model': {'type': 'hybrid', 'pretrained': True},
        'data': {'batch_size': 8, 'num_workers': 2},
        'training': {'num_epochs': 50, 'learning_rate': 1e-4},
    }
}


def get_config(preset='balanced', **kwargs):
    """
    Get training configuration
    
    Args:
        preset: Configuration preset ('fast', 'balanced', 'high_quality', 'low_memory')
        **kwargs: Additional config overrides
    
    Returns:
        Configuration dictionary
    """
    # Start with default config
    config = Config.DEFAULT_CONFIG.copy()
    
    # Apply preset
    if preset in CONFIGS:
        for key, value in CONFIGS[preset].items():
            if key in config:
                config[key].update(value)
    
    # Apply custom overrides
    for key, value in kwargs.items():
        if '.' in key:
            # Handle nested keys like 'model.type'
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = value
        else:
            config[key] = value
    
    return config


if __name__ == "__main__":
    # Example usage
    print("Default Configuration:")
    print(json.dumps(Config.DEFAULT_CONFIG, indent=2))
    
    print("\n" + "="*80)
    print("Balanced Preset Configuration:")
    config = get_config('balanced')
    print(json.dumps(config, indent=2))
    
    # Save example config
    Config.save_yaml(config, 'config_example.yaml')
    print("\n✅ Saved example config to config_example.yaml")
