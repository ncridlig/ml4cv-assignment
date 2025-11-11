# Semantic Segmentation of Unexpected Objects on Roads

ML4CV Assignment - Anomaly-Aware Semantic Segmentation for Autonomous Driving

## Project Overview

This project implements a DeepLabV3+ semantic segmentation model with ResNet50 backbone for road scene understanding, enhanced with zero-shot anomaly detection capabilities. The model is trained on the StreetHazards dataset to segment 12 known road classes (road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person) while detecting unexpected objects (anomalies) without explicit training on anomalous examples.

**Key Achievements**:
- **Closed-Set Segmentation**: 50.26% mIoU on test set (+12.69% over baseline through multi-scale augmentation)
- **Anomaly Detection**: 90.50% AUROC with Simple Max Logits method (beats authors' baseline of 89.30%)
- **Multi-Scale Augmentation**: Novel variable crop strategy (0.5-2.0× scale range) for improved robustness

## Model Weights

Due to file size limitations (161MB), the trained model weights are available via OneDrive:

**Download Link**: [INSERT ONEDRIVE LINK HERE]

After downloading, place the model file in the `models/` directory:
```
models/deeplabv3_resnet50_augmented_10_47_09-11-25_mIoU_5026.pth
```

## System Requirements

- **OS**: Linux (tested on WSL2 with Ubuntu)
- **CUDA**: v12.9
- **Python**: 3.12
- **GPU**: NVIDIA GPU with at least 8GB VRAM (for inference; training requires 16GB+)
- **RAM**: 16GB minimum
- **Disk Space**: ~10GB for datasets + models

## Setup Instructions

### 1. Install Dependencies

Create a virtual environment and install required packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2. Download Datasets

Download the StreetHazards and BDD-Anomaly datasets:
```bash
chmod +x download.sh
./download.sh
```

Expected dataset sizes:
- StreetHazards: ~5GB (5,125 train + 1,031 validation + 1,500 test images)
- BDD-Anomaly: ~3GB (additional test set)

Verify datasets loaded correctly:
```bash
python3 dataloader.py
```

Expected output:
```
Train samples: 8125
Validation samples: 4187
Test samples: 3000
Batch shape: torch.Size([32, 3, 224, 224])
```

### 3. Download Model Weights

Download the trained model from the OneDrive link above and place it in `models/` directory. The project uses this model path by default (configured in `config.py`):
```
models/deeplabv3_resnet50_augmented_10_47_09-11-25_mIoU_5026.pth
```

## Running the Project

### Main Jupyter Notebook (Primary Deliverable)

The complete project demonstration is in `main.ipynb`:

```bash
source .venv/bin/activate
jupyter notebook main.ipynb
```

**Expected Runtime**: ~30-45 minutes (with GPU)
- Model loading and setup: ~1 minute
- Qualitative evaluation visualizations: ~5 minutes
- Quantitative metrics computation: ~10 minutes
- Anomaly detection evaluation: ~15-20 minutes
- Ablation study plots: ~5 minutes

**Note**: The notebook loads pre-computed visualizations from `assets/` to reduce runtime. All code is included and can be re-run to regenerate results.

### Individual Scripts

**Closed-Set Segmentation Evaluation**:
```bash
python3 evaluate.py
```

**Anomaly Detection Methods**:
```bash
# Simple Max Logits (Best: AUROC 90.50%)
python3 anomaly_detection/simple_max_logits.py

# Maximum Softmax Probability
python3 anomaly_detection/maximum_softmax_probability.py

# Standardized Max Logits
python3 anomaly_detection/standardized_max_logits.py

# Energy Score
python3 anomaly_detection/energy_score_anomaly_detection.py

# HEAT (Hybrid Energy-Adaptive Thresholding)
python3 anomaly_detection/heat_anomaly_detection.py
```

**Ablation Studies**:
```bash
python3 ablation_studies.py
```

## Project Structure

```
.
├── main.ipynb                          # Primary deliverable (50+ cells)
├── config.py                           # Central configuration
├── dataloader.py                       # Dataset loading utilities
├── deeplabv3plus_resnet50.py          # Model architecture
├── train.py                            # Training script (multi-scale augmentation)
├── evaluate.py                         # Closed-set segmentation evaluation
├── ablation_studies.py                # Ablation study comparisons
├── models/                             # Model training and checkpoints
│   ├── training_scripts/              # PyTorch training scripts
│   └── checkpoints/                   # Trained model weights (.pth files)
├── anomaly_detection/                 # Anomaly detection methods
│   ├── simple_max_logits.py          # Method #1 (Best: AUROC 90.50%)
│   ├── maximum_softmax_probability.py # Method #2
│   ├── standardized_max_logits.py    # Method #3
│   ├── energy_score_anomaly_detection.py # Method #4
│   └── heat_anomaly_detection.py     # Method #5 (HEAT)
├── assets/                             # Pre-computed visualizations
│   ├── qualitative_eval/              # Segmentation samples
│   └── anomaly_detection/             # Anomaly detection samples
├── log.md                              # Detailed experiment log
└── README-ORIGINAL.md                  # Assignment instructions
```

## Results Summary

### Closed-Set Segmentation

| Model | Augmentation | Train mIoU | Val mIoU | Test mIoU |
|-------|-------------|------------|----------|-----------|
| ResNet50 Baseline | Weak (horizontal flip, crop) | 62.4% | 37.1% | 37.57% |
| **ResNet50 (Ours)** | **Multi-Scale (0.5-2.0×)** | **75.9%** | **50.2%** | **50.26%** |

**Improvement**: +12.69% mIoU through multi-scale augmentation

### Anomaly Detection

| Method | FPR95 | AUROC | AUPR |
|--------|-------|-------|------|
| **Simple Max Logits (Ours)** | 33.12% | **90.50%** | 8.43% |
| Maximum Softmax Probability | 33.57% | 86.71% | 6.21% |
| Standardized Max Logits | 83.91% | 80.25% | 5.41% |
| Authors' Baseline | **26.50%** | 89.30% | **10.60%** |

**Key Finding**: Simple Max Logits beats baseline AUROC (90.50% vs 89.30%)

## Methodology Highlights

### Multi-Scale Augmentation Strategy

Our key innovation is **variable crop sizes** proportional to scale factor:
- Scale 0.5×: Crop 256×256 → Resize to 512×512 (zoomed-out view)
- Scale 1.0×: Crop 512×512 → Resize to 512×512 (normal view)
- Scale 2.0×: Crop 1024×1024 → Resize to 512×512 (zoomed-in view)

This approach provides natural multi-scale context **without black padding artifacts**, teaching the model to handle objects at different distances.

### Zero-Shot Anomaly Detection

All three methods detect anomalies **without training on anomalous examples**:
- **Simple Max Logits**: Uses raw logit magnitudes as confidence measure
- **Maximum Softmax Probability**: Uses normalized probabilities
- **Standardized Max Logits**: Normalizes by class-specific statistics

## Citation

Dataset and baseline from:
```
@inproceedings{hendrycks2019anomaly,
  title={Scaling Out-of-Distribution Detection for Real-World Settings},
  author={Hendrycks, Dan and Basart, Steven and Mazeika, Mantas and others},
  booktitle={ICML},
  year={2019}
}
```

## License

Academic use only - ML4CV course assignment.