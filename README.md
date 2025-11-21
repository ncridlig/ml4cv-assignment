# Semantic Segmentation of Unexpected Objects on Roads

ML4CV Assignment - Anomaly-Aware Semantic Segmentation for Autonomous Driving

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

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

### 3. Download Model Weights
Download trained weights (161MB) from OneDrive: [INSERT LINK]

Place in: `models/checkpoints/deeplabv3_resnet50_augmented_10_47_09-11-25_mIoU_5026.pth`

### 4. Run Notebook
```bash
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
