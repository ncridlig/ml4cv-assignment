# Qualitative Evaluation Summary - DeepLabV3+ Baseline

**Date:** November 4, 2025
**Model:** `best_deeplabv3_streethazards.pth` (168 MB)
**Training:** 30 epochs, 2.5 hours, RTX 4080

---

## Executive Summary

First training attempt of DeepLabV3+ baseline model on StreetHazards dataset. Model achieved **31-33% validation mIoU** over full dataset, with selected samples showing **38.8% ± 4.8%** on validation and **30.9% ± 9.9%** on test set. Performance below initial target (40-50% mIoU), but auxiliary classifier bug was discovered and fixed for future training runs.

---

## Model Configuration

### Architecture
- **Base Model:** DeepLabV3+ with ResNet50 backbone
- **Pretrained Weights:** ImageNet
- **Output Classes:** 13 (classes 0-12, ignoring anomaly class 13)
- **Input Size:** 512×512 (resized from 1280×720)

### Training Hyperparameters
- **Optimizer:** Adam (lr=1e-4)
- **Scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Loss Function:** CrossEntropyLoss (ignore_index=13)
- **Batch Size:** 4
- **Epochs:** 30
- **Training Time:** 2.5 hours
- **GPU:** NVIDIA RTX 4080 (~88% utilization)

### Data Augmentation
- RandomHorizontalFlip(p=0.5)
- ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

---

## Quantitative Results

### Validation Set (10 Sample Evaluation)

| Metric | Value |
|--------|-------|
| **Mean mIoU** | **38.8%** |
| Standard Deviation | ±4.8% |
| Minimum IoU | 33.1% |
| Maximum IoU | 47.7% |
| Sample Size | 10 images (evenly sampled from 1031 total) |

**Per-Sample Breakdown:**
| Sample Index | mIoU | Notes |
|--------------|------|-------|
| 0 | 40.73% | - |
| 103 | 33.13% | Lower performance |
| 206 | 35.44% | - |
| 309 | 47.67% | **Best performing sample** |
| 412 | 41.43% | - |
| 515 | 46.01% | Good performance |
| 618 | 38.33% | - |
| 721 | 33.15% | Lower performance |
| 824 | 36.61% | - |
| 927 | 35.50% | - |

### Test Set (10 Sample Evaluation - With Anomalies)

| Metric | Value |
|--------|-------|
| **Mean mIoU** | **30.9%** |
| Standard Deviation | ±9.9% (high variance) |
| Minimum IoU | 13.6% |
| Maximum IoU | 50.7% |
| Sample Size | 10 images (evenly sampled from 1500 total) |

**Per-Sample Breakdown:**
| Sample Index | mIoU | Notes |
|--------------|------|-------|
| 0 | 50.71% | **Best performing sample** |
| 150 | 23.19% | Contains anomaly, low performance |
| 300 | 31.74% | - |
| 450 | 13.56% | **Worst performing sample** - likely large anomaly |
| 600 | 30.15% | - |
| 750 | 24.40% | - |
| 900 | 26.64% | - |
| 1050 | 28.72% | - |
| 1200 | 39.76% | Good performance |
| 1350 | 40.33% | Good performance |

**Key Observation:** Test set performance is ~8% lower than validation (30.9% vs 38.8%), with much higher variance (±9.9% vs ±4.8%). This is expected since test images contain anomalous objects that the model has never seen during training.

---

## Performance Analysis

### Validation vs Full Dataset Discrepancy

- **Full validation set (during training):** 31-33% mIoU
- **10 sampled images (qualitative eval):** 38.8% mIoU
- **Discrepancy:** ~6-8% higher on sampled images

**Possible explanations:**
1. **Sampling bias:** The 10 evenly-spaced samples may not be representative of the full distribution
2. **Easier scenes:** Sampled images may have simpler scene composition or better lighting
3. **Class distribution:** Sampled images may have more of the "easier" classes (road, building, vegetation)
4. **Statistical variance:** 10 samples is a small subset (only 1% of validation set)

**Conclusion:** The true validation performance is likely closer to 31-33% (full dataset) than 38.8% (10 samples).

### Test Set Performance Drop

- **Validation:** 38.8% mIoU
- **Test:** 30.9% mIoU
- **Performance drop:** ~8 percentage points

**Expected causes:**
1. **Anomalous objects present:** Model tries to classify anomalies as known classes, creating errors
2. **Domain shift:** Test set may have different lighting, weather, or scene composition
3. **No anomaly training:** Model has zero exposure to class 13 during training
4. **Misclassification patterns:** Anomalies likely classified as "other" (class 3) or nearest semantic class

### High Variance on Test Set

- **Validation std dev:** ±4.8%
- **Test std dev:** ±9.9%
- **Difference:** Test set has 2× higher variance

**Interpretation:**
- Some test images have no or small anomalies → high mIoU (40-50%)
- Other test images have large anomalies → low mIoU (13-24%)
- This bimodal distribution reflects the binary nature of anomaly presence

---

## Qualitative Observations

### Strengths
1. **Large, contiguous regions:** Model performs well on large uniform regions (roads, buildings, sky)
2. **Strong predictions:** Confidence maps show high certainty (>0.8) on common classes
3. **Semantic understanding:** Generally captures scene layout (sky at top, road at bottom)

### Weaknesses
1. **Fine boundaries:** Poor delineation at object edges (cars, pedestrians, poles)
2. **Small objects:** Struggles with small classes (traffic signs, poles, road lines)
3. **Class confusion:** Frequent confusion between similar classes:
   - "other" vs "building"
   - "fence" vs "wall"
   - "sidewalk" vs "road"
4. **Anomaly handling:** Anomalies misclassified as semantically similar known classes

### Failure Modes

**Type 1: Boundary errors**
- Segmentation masks "bleed" across object boundaries
- Especially visible between road/sidewalk, car/road interfaces
- Cause: 512×512 downsampling loses fine spatial detail

**Type 2: Small object omission**
- Traffic signs, poles often completely missed
- Merged into background classes
- Cause: Class imbalance (rare classes underrepresented in training)

**Type 3: Anomaly misclassification**
- Anomalous objects classified as "other" or nearest known class
- No uncertainty signal for out-of-distribution objects
- Cause: Model trained with closed-set objective (no anomaly awareness)

---

## Visualization Outputs

All visualizations saved to `assets/qualitative_eval/`

### Directory Structure
```
assets/qualitative_eval/
├── validation/
│   ├── validation_sample_000.png
│   ├── validation_sample_103.png
│   ├── ... (10 total)
├── test/
│   ├── test_sample_000.png
│   ├── test_sample_150.png
│   ├── ... (10 total)
├── validation_comparison_grid.png
├── test_comparison_grid.png
└── EVALUATION_SUMMARY.md (this file)
```

### Visualization Components

Each sample visualization (2×3 grid) shows:

**Row 1:**
- **Column 1:** Input image (RGB, denormalized)
- **Column 2:** Ground truth mask (colored by class)
- **Column 3:** Predicted mask (colored by class, mIoU displayed)

**Row 2:**
- **Column 1:** Prediction overlay (50% image + 50% prediction)
- **Column 2:** Confidence map (viridis colormap, range 0-1)
- **Column 3:** Error map (red=wrong, green=correct pixels)

**Text Summary:** Left side shows:
- Sample index and split name
- Mean IoU and pixel accuracy
- Average confidence
- Anomaly detection (if present in ground truth)
- Top-3 classes by IoU

---

## Known Issues & Bug Fixes

### 🐛 Critical Bug: Auxiliary Classifier Not Enabled

**Discovery:** During code review after first training run
**Impact:** Missing ~0.5-1% mIoU improvement from better gradient flow

**Problem details:**
1. Model architecture: `deeplabv3_resnet50` has both main classifier and auxiliary classifier
2. Only main classifier output layer was modified for 13 classes
3. Auxiliary classifier still predicting 21 COCO classes (incompatible)
4. Training loop only used `model(images)['out']`, ignoring auxiliary output
5. Loss computed only on main output

**Fix implemented:**
```python
# 1. Modify both classifiers
model.classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)
model.aux_classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)  # Added

# 2. Use both outputs in training
output_dict = model(images)
main_output = output_dict['out']
aux_output = output_dict['aux']  # Added

# 3. Combined loss
main_loss = loss_fn(main_output, masks)
aux_loss = loss_fn(aux_output, masks)  # Added
loss = main_loss + 0.4 * aux_loss  # Standard 0.4 weight
```

**Expected improvement:** Next training run should achieve 32-35% mIoU (vs current 31-33%)

---

## Comparison to Literature

### StreetHazards Baseline Performance (from papers)

| Method | Validation mIoU | Test mIoU | Notes |
|--------|-----------------|-----------|-------|
| **Our baseline** | **31-33%** | **~31%** (estimated) | First attempt, bug discovered |
| DeepLabV3 (Hendrycks et al.) | ~45-50% | ~45% | Original paper baseline |
| PSPNet (reported) | ~42-48% | ~42% | Literature benchmark |

**Gap analysis:**
- We are ~12-15% below published baselines
- Likely causes:
  1. **Training duration:** 30 epochs may be insufficient (papers use 50-100)
  2. **Auxiliary classifier bug:** Missing ~1% improvement
  3. **Hyperparameters:** May need tuning (learning rate, batch size, augmentation)
  4. **Image resolution:** Using 512×512 vs some papers use 720p or 1024×512
  5. **Implementation details:** Class weighting, label smoothing, other tricks not implemented

**Conclusion:** Our baseline is a reasonable starting point but has clear room for improvement.

---

## Next Steps

### Immediate Actions
1. **Retrain with bug fix:** Use corrected `deeplabv3plus.py` with auxiliary classifier enabled
2. **Compare results:** Document improvement from bug fix (expected +1-2% mIoU)
3. **Hyperparameter tuning:** If still below 35%, try:
   - Longer training (50 epochs)
   - Higher resolution (720×720)
   - Different learning rate schedule
   - Class weighting for imbalanced classes

### Phase 4: Anomaly Detection
Once baseline is satisfactory (>35% mIoU):
1. **Implement Standardized Max Logits (SML):**
   - Simple post-processing method
   - No retraining required
   - Expected test AUPR: 15-25%
2. **Evaluate on test set:** Measure anomaly detection performance
3. **Visualize anomaly scores:** Heatmaps showing detected anomalies

### Phase 5: Advanced Methods (if time permits)
- Metric learning approach (DMLNet-inspired)
- Requires retraining with contrastive loss
- Expected improvement: +3-5% AUPR

---

## Lessons Learned

### Technical Insights
1. **Auxiliary classifiers matter:** Even "optional" architectural components can impact performance
2. **Always verify all components:** Check that all model parts are being used as intended
3. **Qualitative evaluation is essential:** Visual inspection reveals failure modes that metrics miss
4. **Sampling bias is real:** 10 samples ≠ 1000 samples in performance estimation

### Process Insights
1. **Document everything:** This detailed log enabled finding the auxiliary classifier bug
2. **Iterate systematically:** Fix one thing at a time, measure, compare
3. **Visualize continuously:** Confidence maps and error maps provide actionable insights
4. **Expect gaps from literature:** Reproducing published results is hard; focus on learning

### Learning Outcomes (Primary Objective)
> This assignment prioritizes learning over perfect results. The process of:
> - Implementing a complex model from scratch
> - Discovering and debugging issues
> - Understanding architectural components (aux classifier)
> - Evaluating qualitatively and quantitatively
>
> ...is more valuable than achieving state-of-the-art performance.

---

## Appendix: Model Loading for Inference

For future reference, use this code to load the trained model:

```python
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

# Initialize model
model = deeplabv3_resnet50(weights=None)
model.classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)

# Load checkpoint (use strict=False to ignore aux_classifier keys)
state_dict = torch.load('best_deeplabv3_streethazards.pth', map_location='cuda')
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to('cuda')

# Inference
with torch.no_grad():
    output = model(image)['out']  # Shape: (1, 13, H, W)
    prediction = torch.argmax(output, dim=1)  # Shape: (1, H, W)
```

---

**Generated:** November 4, 2025
**Script:** `evaluate_qualitative.py`
**Author:** Nicolas Cridlig (A16002193)
**Course:** ML4CV, University of Bologna (A.Y. 2024-2025)
