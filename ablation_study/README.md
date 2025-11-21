# Ablation Study for DeepLabV3+ ResNet50

This directory contains ablation studies to analyze the contributions of different components to the best model's performance (50.26% mIoU).

## Overview

The best model achieved **50.26% mIoU** with multi-scale augmentation, compared to **37.57% mIoU** for the baseline with weak augmentation. These studies systematically isolate and measure the impact of individual components.

## Scripts

### 1. `augmentation_ablation.py` ✅ (PRIORITY 1)
**Purpose:** Identify which augmentation components contribute most to the +12.69% mIoU improvement.

**Experiments:**
- **Baseline**: No augmentation (resize + normalize only)
- **+Scale**: Add random scale (0.5-2.0×)
- **+Scale+Flip**: Add horizontal flip
- **+Scale+Flip+Crop**: Add random crop
- **+Scale+Flip+Crop+Color**: Full augmentation (current best)

**Metrics:**
- Train/Val/Test mIoU
- Training time per epoch
- Convergence speed (epochs to plateau)

**Estimated runtime:** 5 experiments × ~20 epochs (early stopping, patience=5) × 5 min/epoch = ~8 hours

**Output:**
- `ablation_study/results/augmentation_ablation_results.json`
- `ablation_study/results/augmentation_ablation_plot.png`
- Trained model checkpoints (optional)

---

### 2. `multiscale_range_ablation.py` (PRIORITY 2)
**Purpose:** Find optimal scale augmentation range for multi-scale training.

**Experiments:**
- **Conservative**: [0.75, 1.25]
- **Moderate**: [0.5, 1.5]
- **Current best**: [0.5, 2.0]
- **Aggressive**: [0.25, 2.5]
- **Downscale only**: [0.5, 1.0]
- **Upscale only**: [1.0, 2.0]

**Metrics:**
- Test mIoU
- Training stability (loss variance)
- Per-class performance (small vs large objects)

**Estimated runtime:** 6 experiments × 40 epochs × 5 min/epoch = ~20 hours

**Output:**
- `ablation_study/results/multiscale_range_results.json`
- `ablation_study/results/multiscale_range_plot.png`

---

### 3. `anomaly_threshold_sensitivity.py` (PRIORITY 3)
**Purpose:** Analyze sensitivity to decision thresholds and validate robustness for deployment.

**Experiments:**
- **Threshold sweep**: Test 50 thresholds from -5.0 to 0.0
- **Subsampling rates**: 100K, 500K, 1M, full pixels
- **Temperature scaling**: T = 0.5, 1.0, 2.0, 5.0

**Metrics:**
- AUROC, AUPR, FPR95 vs threshold
- F1 score at different operating points
- Precision-Recall curves
- Stability across subsampling rates

**Estimated runtime:** ~30 minutes (re-evaluation only, no training)

**Output:**
- `ablation_study/results/threshold_sensitivity_results.json`
- `ablation_study/results/threshold_sensitivity_plots.png` (multiple subplots)
- Recommended operating thresholds for different FPR targets

---

### 4. `architecture_components_ablation.py` (PRIORITY 4)
**Purpose:** Identify critical architectural components of DeepLabV3+.

**Experiments:**
- **Full model**: DeepLabV3+ ResNet50 (baseline)
- **-ASPP**: Remove ASPP module, use simple decoder
- **-AuxClassifier**: Remove auxiliary classifier
- **-Pretrained**: Train from scratch (no ImageNet initialization)
- **ResNet34**: Shallower backbone
- **ResNet101**: Deeper backbone

**Metrics:**
- Test mIoU
- Model parameters and FLOPs
- Training time
- GPU memory usage

**Estimated runtime:** 6 experiments × 40 epochs × 5 min/epoch = ~20 hours

**Output:**
- `ablation_study/results/architecture_ablation_results.json`
- `ablation_study/results/architecture_comparison_table.md`

---

## Usage

### Quick Start (Augmentation Ablation)
```bash
# Run data augmentation ablation (Priority 1)
.venv/bin/python3 ablation_study/augmentation_ablation.py

# Results will be saved to:
# - ablation_study/results/augmentation_ablation_results.json
# - ablation_study/results/augmentation_ablation_plot.png
```

### Run All Studies (Sequential)
```bash
.venv/bin/python3 ablation_study/augmentation_ablation.py
.venv/bin/python3 ablation_study/multiscale_range_ablation.py
.venv/bin/python3 ablation_study/anomaly_threshold_sensitivity.py
.venv/bin/python3 ablation_study/architecture_components_ablation.py
```

### Run Only Fast Studies (No Training)
```bash
# Threshold sensitivity only (~30 min)
.venv/bin/python3 ablation_study/anomaly_threshold_sensitivity.py
```

---

## Results Directory Structure

```
ablation_study/
├── README.md                           # This file
├── augmentation_ablation.py            # Script 1
├── multiscale_range_ablation.py        # Script 2
├── anomaly_threshold_sensitivity.py    # Script 3
├── architecture_components_ablation.py # Script 4
├── results/                            # Output directory
│   ├── augmentation_ablation_results.json
│   ├── augmentation_ablation_plot.png
│   ├── multiscale_range_results.json
│   ├── multiscale_range_plot.png
│   ├── threshold_sensitivity_results.json
│   ├── threshold_sensitivity_plots.png
│   ├── architecture_ablation_results.json
│   └── architecture_comparison_table.md
└── checkpoints/                        # Model checkpoints (optional)
    ├── ablation_baseline/
    ├── ablation_scale/
    ├── ablation_scale_flip/
    └── ...
```

---

## Configuration

All scripts use the same base configuration from `config.py`:
- Image size: 512×512
- Batch size: 4
- Optimizer: AdamW
- Learning rate: 1e-4
- Max epochs: 40 (with early stopping, patience=5)
- Device: CUDA (if available)

Individual scripts may override specific parameters for their experiments.

---

## Expected Outcomes

### Augmentation Ablation
- Hypothesis: Random scale contributes most to improvement
- Expected ranking: Baseline < +Flip < +Color < +Crop < +Scale

### Multi-Scale Range
- Hypothesis: Wider range improves robustness but may slow convergence
- Expected optimal: [0.5, 2.0] or [0.5, 1.5]

### Threshold Sensitivity
- Hypothesis: Performance stable across subsampling rates
- Expected: AUROC variance < 1% across subsampling

### Architecture Components
- Hypothesis: ASPP and pretrained weights are critical
- Expected: -ASPP drops ~5-10% mIoU, -Pretrained drops ~10-15% mIoU

---

## Notes

- All experiments use the same train/val/test splits for fair comparison
- Random seed fixed for reproducibility (RANDOM_SEED=42 from config.py)
- GPU memory: ~8GB required for training (RTX 4080 Super 16GB available)
- Results are automatically saved in JSON format for easy plotting
- Scripts support resuming from checkpoints if interrupted

---

## Time Budget

**With early stopping (patience=5):**
- **Priority 1** (Augmentation): ~8 hours → **DONE, look at assets/ablation_study_method_comparison.md**
- **Priority 2** (Multi-scale range): ~10 hours → **Complete if time permits**
- **Priority 3** (Threshold sensitivity): 30 minutes → **DONE, look at ablation_study/results**
- **Priority 4** (Architecture): ~10 hours → **Future work**

**Minimum viable ablation study:** Priority 1 + Priority 3 = ~8.5 hours total

---

## References

- Best model: `models/checkpoints/deeplabv3_resnet50_augmented_10_47_09-11-25_mIoU_5026.pth`
- Baseline model: `models/checkpoints/best_deeplabv3_streethazards_11_52_04-11-25_mIoU_3757.pth`
- Training script: `models/training_scripts/train_augmented_resnet50.py`
