"""
Configuration settings for the tumor image classification pipeline.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    'resnet50': {
        'base_model': 'ResNet50',
        'input_shape': (224, 224, 3),
        'freeze_layers': True,
        'dropout_rate': 0.5
    },
    'efficientnet_b0': {
        'base_model': 'EfficientNetB0',
        'input_shape': (224, 224, 3),
        'freeze_layers': True,
        'dropout_rate': 0.3
    },
    'custom_cnn': {
        'input_shape': (224, 224, 3),
        'num_filters': [32, 64, 128, 256],
        'dropout_rate': 0.5
    }
}

# Training configurations
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'test_split': 0.1,
    'random_state': 42,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5
}

# Data preprocessing configurations
PREPROCESSING_CONFIG = {
    'target_size': (224, 224),
    'normalize': True,
    'augmentation': {
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'horizontal_flip': True,
        'zoom_range': 0.2,
        'fill_mode': 'nearest'
    }
}

# Dataset configurations
DATASET_CONFIGS = {
    'breakhis': {
        'url': 'https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/',
        'classes': ['benign', 'malignant'],
        'magnifications': ['40X', '100X', '200X', '400X']
    },
    'kaggle_histopathology': {
        'dataset_name': 'paultimothymooney/breast-histopathology-images',
        'classes': ['0', '1'],  # 0: benign, 1: malignant
    }
}

# Evaluation metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

# Visualization settings
VISUALIZATION_CONFIG = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis'
}