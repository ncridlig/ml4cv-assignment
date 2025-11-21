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
MODEL_PATH = 'ablation_study/checkpoints/+Scale__20_52_19-11-25_mIoU_0.5176_size_512x512.pth' # 'models/checkpoints/hiera_large_cropaug_streethazards_04_43_12-11-25_mIoU_4677.pth'
MODEL_ARCHITECTURE = 'deeplabv3_resnet50'  # Options: deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50
MODEL_THRESHOLD = -1.9271

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# Dataset paths
TRAIN_ROOT = 'streethazards_train/train'
TEST_ROOT = 'streethazards_test/test'

# Image dimensions
IMAGE_SIZE = (512, 512) # (1280, 720) is streethazards original resolution # 512 is the best for resnet50/101

# =============================================================================
# CLASS CONFIGURATION
# =============================================================================
NUM_TRAINED_CLASSES = 13  # 0-12 normal classes
NUM_CLASSES = 14  # For dataset purposes, 0-12 normal, 13 = anomaly (ignored in training)
ANOMALY_CLASS_IDX = 13  # Class 13 is anomaly (only in test set)
IGNORE_INDEX = 13  # Ignore anomaly class during training

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 3  # Good for batch norm stability
LEARNING_RATE = 1e-4
EPOCHS = 20  # Increase for stronger augmentation
NUM_WORKERS = 3
PRINT_FREQ = 500  # Print training stats every N iterations

# =============================================================================
# ANOMALY DETECTION CONFIGURATION
# =============================================================================
# Simple Max Logits threshold (from anomaly_detection/simple_max_logits.py results)
ANOMALY_THRESHOLD = -1.4834  # Optimal threshold for F1 score
MAX_PIXELS_EVALUATION = 1_000_000  # Subsample to this many pixels for evaluation
RANDOM_SEED = 42  # For reproducible subsampling

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================
OUTPUT_DIR_ANOMALY = Path('assets/anomaly_detection')
OUTPUT_DIR_QUALITATIVE = Path('assets/qualitative_eval')
OUTPUT_DIR_MODELS = Path('models/checkpoints')  # Model checkpoints saved here

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
NUM_QUALITATIVE_SAMPLES = 10  # Number of samples to visualize per split
