# Semantic Segmentation of Unexpected Objects on Roads

ML4CV Assignment - Anomaly-Aware Semantic Segmentation for Autonomous Driving

## Repository

This `README.md` and `main.ipynb` can be viewed statically. To run the project, clone the full repository:

```bash
# HTTPS
git clone https://github.com/ncridlig/ml4cv-assignment.git

# SSH
git clone git@github.com:ncridlig/ml4cv-assignment.git
```

---

## Prerequisites (Before Running)

### 1. Download Model Weights (Required)
Download trained weights from OneDrive and place in project:

| Model | Size | OneDrive Link | Path |
|-------|------|---------------|------|
| DeepLabV3+ ResNet50 | 161 MB | [link](https://liveunibo-my.sharepoint.com/:u:/g/personal/nicolasivan_cridlig_studio_unibo_it/IQCd1DjuDEWiR4pS0ZZTOSXuAcU_ho1uchFRqCiWSigcKBo?e=2jRYnP) | `ablation_study/checkpoints/+Scale__20_52_19-11-25_mIoU_0.5176_size_512x512.pth` |
| SegFormer-B5 | 324 MB | [link](https://liveunibo-my.sharepoint.com/:u:/g/personal/nicolasivan_cridlig_studio_unibo_it/IQDY4WynaES6Rpc1P5TJdImVASXArqDTHB3PRpq_-pklIzc?e=6MNbBT) | `models/checkpoints/segformer_b5_streethazards_augmented_10_06_12-11-25_mIoU_5412.pth` |

```bash
# Create directories first
mkdir -p ablation_study/checkpoints models/checkpoints
```

### 2. Download Dataset
```bash
./download_dataset.sh
```

> **Tip:** If you already have the dataset, simply copy `streethazards_test/` and `streethazards_train/` to the repository root.

### 3. Install Dependencies
**Requires Python 3.12+**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install project as package (required for imports)
pip install -e .
```

This installs the project in editable mode, enabling imports like:
```python
from utils.dataloader import StreetHazardsDataset
from config import DEVICE, MODEL_PATH
from anomaly_detection.simple_max_logits import compute_anomaly_scores
```

---

## Project Overview

This project implements a DeepLabV3+ semantic segmentation model with ResNet50 backbone for road scene understanding, enhanced with zero-shot anomaly detection capabilities. The model is trained on the StreetHazards dataset to segment 12 known road classes while detecting unexpected objects (anomalies) without explicit training on anomalous examples.

## Key Results

| Metric | Our Model | Authors' Baseline |
|--------|-----------|-------------------|
| **Segmentation mIoU** | **50.26%** | ~37% |
| **Anomaly AUROC** | **90.50%** | 89.30% |
| Anomaly AUPR | 8.43% | 10.60% |
| Anomaly FPR95 | 33.12% | 26.50% |

**Key Achievements**:
- +33.8% relative improvement in segmentation through multi-scale augmentation
- Beats authors' AUROC baseline with zero-shot Simple Max Logits method
- Systematic evaluation: 4 architectures, 3 anomaly methods, 2 ablation studies

## Quick Start

After completing prerequisites above:
```bash
source .venv/bin/activate
jupyter notebook main.ipynb
```

## Project Structure

```
ml4cv-assignment/
├── main.ipynb                      # Primary deliverable
├── config.py                       # Central configuration
├── utils/
│   ├── dataloader.py              # Dataset loading & augmentation
│   ├── model_utils.py             # Model loading utilities
│   └── visualize.py               # Visualization helpers
├── models/
│   ├── training_scripts/          # Training scripts
│   └── checkpoints/               # Model weights
├── anomaly_detection/
│   ├── simple_max_logits.py       # Best: 90.50% AUROC
│   ├── maximum_softmax_probability.py
│   ├── standardized_max_logits.py
│   ├── energy_score_anomaly_detection.py
│   └── heat_anomaly_detection.py
├── ablation_study/
│   ├── augmentation_ablation.py   # Augmentation study
│   ├── scale_range_ablation.py    # Scale range study
│   └── results/                   # Ablation results & plots
├── visualizations/                # Comparison scripts
├── assets/                        # Pre-computed visualizations
└── log.md                         # Development log
```

## Methodology

### Multi-Scale Augmentation
Variable crop sizes proportional to scale factor (0.5-2.0×):
- Scale 0.5×: 256×256 crop → 512×512 (fine details)
- Scale 1.0×: 512×512 crop → 512×512 (normal view)
- Scale 2.0×: 1024×1024 crop → 512×512 (context)

No black padding artifacts—crop size adapts to scale.

### Zero-Shot Anomaly Detection
Simple Max Logits: `anomaly_score = -max(logits)`
- No training on anomalies required
- Outperforms complex normalized methods
- 90.50% AUROC (beats 89.30% baseline)

## Ablation Studies

### Augmentation Ablation
| Configuration | Val mIoU | AUROC |
|--------------|----------|-------|
| No Augmentation | 56.29% | 88.5% |
| +Scale | 51.76% | **91.4%** |
| +Scale+Rotate+Flip+Color | 48.13% | 90.8% |

**Finding**: Scale-only augmentation optimizes anomaly detection.

### Scale Range Ablation
| Range | Val mIoU | AUROC |
|-------|----------|-------|
| (0.5, 1.5) Zoom-In | **51.43%** | 90.5% |
| (0.5, 2.0) Baseline | 49.90% | **91.2%** |
| (0.9, 1.1) Minimal | 49.27% | 90.6% |

**Finding**: Zoom-in emphasis (0.5-1.5) best for segmentation; baseline (0.5-2.0) best for anomaly detection.

## System Requirements

- Python 3.12, CUDA 12.9
- GPU: 8GB+ VRAM (inference), 16GB+ (training)
- RAM: 16GB minimum
- Disk: ~10GB for datasets + models

## Citation

```bibtex
@inproceedings{hendrycks2019anomaly,
  title={Scaling Out-of-Distribution Detection for Real-World Settings},
  author={Hendrycks, Dan and Basart, Steven and Mazeika, Mantas and others},
  booktitle={ICML},
  year={2019}
}
```

## License

Academic use only - ML4CV course assignment, University of Bologna (A.Y. 2024-2025).
