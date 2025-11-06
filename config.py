"""
Central configuration file for ML4CV assignment.
All shared constants and configuration parameters.
"""

import torch
from pathlib import Path

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_PATH = 'models/best_deeplabv3_streethazards_11_52_04-11-25_mIoU_3757.pth'
MODEL_ARCHITECTURE = 'deeplabv3_resnet50'  # Options: deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# Dataset paths
TRAIN_ROOT = 'streethazards_train/train'
TEST_ROOT = 'streethazards_test/test'

# Image dimensions
IMAGE_SIZE = 512

# =============================================================================
# CLASS CONFIGURATION
# =============================================================================
NUM_CLASSES = 13  # Classes 0-12 (known classes for training)
ANOMALY_CLASS_IDX = 13  # Class 13 is anomaly (only in test set)
IGNORE_INDEX = 13  # Ignore anomaly class during training

# =============================================================================
# TRAINING HYPERPARAMETERS (for deeplabv3plus.py)
# =============================================================================
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 20
NUM_WORKERS = 4
PRINT_FREQ = 500  # Print training stats every N iterations

# =============================================================================
# ANOMALY DETECTION CONFIGURATION
# =============================================================================
# Simple Max Logits threshold (from simple_max_logits.py results)
ANOMALY_THRESHOLD = -1.4834  # Optimal threshold for F1 score
MAX_PIXELS_EVALUATION = 1_000_000  # Subsample to this many pixels for evaluation
RANDOM_SEED = 42  # For reproducible subsampling

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================
OUTPUT_DIR_ANOMALY = Path('assets/anomaly_detection')
OUTPUT_DIR_QUALITATIVE = Path('assets/qualitative_eval')
OUTPUT_DIR_MODELS = Path('models')

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
NUM_QUALITATIVE_SAMPLES = 10  # Number of samples to visualize per split
