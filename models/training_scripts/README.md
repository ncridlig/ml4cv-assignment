# Training Scripts

This directory contains the PyTorch training scripts used to train various segmentation models for the ML4CV assignment.

## Scripts

### ResNet-based Models

**`deeplabv3plus_resnet50.py`**
- Architecture: DeepLabV3+ with ResNet50 backbone
- Standard training (no augmentation)
- Output: ~38% mIoU

**`deeplabv3plus_reesnet101.py`**
- Architecture: DeepLabV3+ with ResNet101 backbone
- Standard training (no augmentation)
- Output: ~37% mIoU

**`train_augmented_resnet50.py`**
- Architecture: DeepLabV3+ with ResNet50 backbone
- **Multi-scale augmentation training** (key innovation)
- Output: 50.26% mIoU (best model)
- Used for main anomaly detection experiments

### Transformer-based Models

**`segformerb5.py`**
- Architecture: SegFormer-B5
- Transformer-based segmentation
- Explored for comparison with CNNs

**`hierabase224.py`**
- Architecture: Hiera Base (224×224)
- Hierarchical vision transformer
- Explored for comparison with CNNs

## Usage

All scripts are standalone and can be run from the project root:

```bash
# From project root
.venv/bin/python3 models/training_scripts/train_augmented_resnet50.py
```

## Model Outputs

Trained models are saved to the `models/checkpoints` directory with filenames like:
- `deeplabv3_resnet50_augmented_10_47_09-11-25_mIoU_5026.pth`
- `deeplabv3_resnet101_*.pth`

## Notes

- All scripts use the configuration from `config.py` in the project root
- Training runs are logged to `models/runs/` directory with TensorBoard logs
- Scripts are self-contained and don't import each other
