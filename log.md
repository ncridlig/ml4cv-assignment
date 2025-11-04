# ML4CV Assignment - Work Log

## Project Overview
- **Course**: Machine Learning for Computer Vision, University of Bologna (A.Y. 2024-2025)
- **Task**: Semantic segmentation of unexpected objects on roads (anomaly segmentation)
- **Dataset**: StreetHazards (5125 train, 1031 val, 1500 test images)
- **Goal**: Segment 12 known classes + detect anomalous objects (unseen during training)
- **Time Budget**: 50 hours total
- **Evaluation**: Qualitative results, mIoU (closed-set), AUPR (anomaly detection), ablation studies, code clarity

---

## Day 0 - Project Planning (Current)

### Date: 2025-11-04

### What We Did Today
1. Reviewed README-ORIGINAL.md and understood project requirements
2. Created comprehensive 50-hour project plan with 7 phases:
   - Phase 1: Setup & Data Pipeline (6h)
   - Phase 2: Baseline Model (8h)
   - Phase 3: Literature Review (4h)
   - Phase 4: Simple Anomaly Detection (8h)
   - Phase 5: Advanced Anomaly Detection (10h)
   - Phase 6: Ablation Studies (8h)
   - Phase 7: Documentation (6h)
3. Created this log file for session continuity

### Current Status
- **Phase**: Planning complete, ready to start Phase 1
- **Hours Used**: 0 / 50
- **Files Created**:
  - `log.md` (this file)
- **Existing Files**:
  - `README-ORIGINAL.md` (assignment description)
  - `README.md` (modified, need to check)
  - `main.ipynb` (exists, need to check state)
  - `requirements.txt` (exists, need to check)
  - `dataloader.py` (exists, need to check)
  - `download.sh` (exists)
  - `streethazards_train/` (downloaded)
  - `streethazards_test/` (downloaded)
  - `.venv/` (virtual environment exists)

### Git Status
```
M README.md
M main.ipynb
M requirements.txt
?? .venv/
?? README-ORIGINAL.md
?? dataloader.py
?? download.sh
?? streethazards_test/
?? streethazards_train/
```

### Next Session - Start Here
**Target Phase**: Phase 1: Setup & Data Pipeline (Hours 1-6)

**Immediate Tasks**:
1. Review existing files (`main.ipynb`, `dataloader.py`, `requirements.txt`) to see what's already implemented
2. Check if datasets are properly downloaded and structured
3. Verify virtual environment has necessary dependencies
4. Complete or fix data pipeline if partially done
5. **Goal for Phase 1**: "I can load and visualize StreetHazards images with their segmentation masks"

**Questions to Answer Next Session**:
- Are the datasets downloaded and extracted properly?
- What's already implemented in `dataloader.py` and `main.ipynb`?
- What dependencies are in `requirements.txt`?
- Do we need to install additional packages?

### Important Notes & Decisions
- Using StreetHazards dataset (CARLA synthetic data)
- 12 known classes + 1 anomaly class (test only)
- Metrics: mIoU for segmentation, AUPR for anomaly detection
- Not a competition - focus on ablations and code clarity
- Grade: 0-10 points added to oral exam

### Reference Papers (Quick Access)
1. [Hendrycks et al. - Scaling OOD Detection](https://arxiv.org/abs/1911.11132) - StreetHazards dataset paper
2. [Cen et al. - Deep Metric Learning](https://arxiv.org/abs/2108.04562) - Embedding-based approach
3. [Jung et al. - Standardized Max Logits](https://arxiv.org/abs/2107.11264) - Simplest method (START HERE)
4. [Liu et al. - Residual Pattern Learning](https://arxiv.org/abs/2211.14512) - Advanced method
5. [Sodano et al. - Class Similarity](https://arxiv.org/abs/2403.07532) - Recent CVPR 2024

### Key Reminders
- Test set with anomalies MUST remain unseen during training
- Notebook must be runnable with training code disabled
- First cell must have: student ID, full name, institutional email
- Model weights >20MB → upload to OneDrive, link in README
- This is individual work, no teams allowed

---

## Day 1 - Phase 1: Data Pipeline Setup ✓ COMPLETED

### Date: 2025-11-04

### What We Did Today
1. **Created comprehensive dataloader.py** with complete functionality:
   - Custom `StreetHazardsDataset` class for train/val/test splits
   - Proper image-mask pair loading from StreetHazards folder structure
   - Data augmentation support (random flip, color jitter for training)
   - Visualization functions: `plot_samples()`, `show_legend()`, `mask_to_rgb()`
   - Factory function `get_dataloaders()` for easy batch loading

2. **Resolved class mapping through research**:
   - Verified official StreetHazards class indices from PyTorch-OOD documentation
   - Confirmed 14 classes total: 0-12 (known), 13 (anomaly, test only)
   - Dataset files use 1-indexed values (1-14), remapped to 0-indexed (0-13) via `-1`
   - Class 0: "unlabeled" (was missing initially)
   - Class 13: "anomaly" - appears in 0/5125 training images, all 1500 test images

3. **Created utility tools**:
   - `utils/class_counter.py`: Analyzes class distribution across splits
   - Saves results to `assets/class_distribution.txt`
   - Shows pixel counts, percentages, and class presence comparison

4. **Project organization**:
   - Created `.gitignore` to exclude datasets, models, venv
   - Set up `assets/` directory for visualizations
   - Configured proper directory structure

5. **Verified data integrity**:
   - Training: 5125 images with classes 0-12 (13 known classes)
   - Validation: 1031 images with classes 0-12
   - Test: 1500 images with classes 0-13 (adds anomaly class)
   - All image-mask pairs load correctly
   - Color mapping matches official StreetHazards specification

### Current Status
- **Phase**: Phase 1 COMPLETE ✓
- **Hours Used**: 2 / 50
- **Files Created**:
  - `dataloader.py` (complete rewrite with full functionality)
  - `.gitignore` (excludes data, models, outputs)
  - `utils/class_counter.py` (class distribution analysis)
  - `utils/__init__.py`
  - `assets/` directory (for outputs)

- **Files Modified**:
  - `log.md` (this file - tracking progress)
  - `requirements.txt` (already had necessary packages)

### Challenges & Solutions
- **Challenge**: Confusion about class indices (1-indexed vs 0-indexed)
  - **Solution**: Researched official PyTorch-OOD implementation, confirmed dataset uses 1-indexed files, we remap to 0-indexed with `-1`

- **Challenge**: Initial incorrect class ordering (missing "unlabeled" at index 0)
  - **Solution**: Web search found official class names, corrected CLASS_NAMES array

- **Challenge**: Understanding train vs test split (anomaly presence)
  - **Solution**: Analyzed all 5125 training masks - confirmed class 14 appears in 0 training images, only in test

### Code Snippets to Remember
```python
# Official StreetHazards class mapping (0-indexed after -1 remapping)
CLASS_NAMES = [
    'unlabeled',      # 0 (originally 1 in files)
    'building',       # 1 (originally 2)
    'fence',          # 2
    'other',          # 3
    'pedestrian',     # 4
    'pole',           # 5
    'road line',      # 6
    'road',           # 7
    'sidewalk',       # 8
    'vegetation',     # 9
    'car',            # 10
    'wall',           # 11
    'traffic sign',   # 12
    'anomaly'         # 13 (originally 14) - TEST ONLY
]

# Loading data with proper remapping
from dataloader import get_dataloaders, CLASS_NAMES, NUM_CLASSES, ANOMALY_CLASS_IDX

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=8,
    num_workers=4,
    image_size=512
)

# Visualization
from dataloader import show_legend, plot_samples
show_legend()  # Saves to assets/class_color_map.png
plot_samples(train_dataset, num_samples=5)
```

### Dataset Statistics
- **Training**: 5125 images, classes 0-12 (no anomaly)
- **Validation**: 1031 images, classes 0-12 (no anomaly)
- **Test**: 1500 images, classes 0-13 (includes anomaly at index 13)
- **Image size**: 1280x720 (resized to 512x512 in transforms)
- **Anomaly types**: 250 different object types (cats, dogs, etc.)

### Next Session - Start Here
**Target Phase**: Phase 2: Baseline Model (Hours 3-10, ~8 hours)

**Immediate Tasks**:
1. Implement baseline segmentation model (DeepLabV3+ or U-Net with ResNet50 backbone)
2. Use pretrained ImageNet weights
3. Modify output layer for 13 classes (train on 0-12, ignore anomaly class)
4. Set up training loop with:
   - Cross-entropy loss (with class weighting if needed)
   - Adam optimizer with learning rate scheduler
   - mIoU metric calculation
   - Model checkpointing (save best model)
5. Train for 20-30 epochs
6. **Goal**: Achieve 40-50% mIoU on validation set

**Key Decisions for Next Session**:
- Which architecture? (DeepLabV3+ recommended - good for segmentation)
- Learning rate? (Start with 1e-4)
- Batch size? (8 or 16 depending on GPU memory)
- Loss function? (Cross-entropy, possibly with class weights for imbalance)

**Files to Create**:
- `model.py` - Model architecture
- `train.py` - Training loop
- `metrics.py` - mIoU calculation
- `main.ipynb` - Update with training code

---

## Session Template (Copy for each new day)

### Day X - [Phase Name]

### Date: YYYY-MM-DD

### What We Did Today
- [List of accomplishments]

### Current Status
- **Phase**:
- **Hours Used**: X / 50
- **Files Modified/Created**:
- **Current Metrics** (if applicable):
  - Validation mIoU: X%
  - Test AUPR: X%

### Challenges & Solutions
- **Challenge**: [What blocked you]
- **Solution**: [How you solved it]

### Next Session - Start Here
- [Specific task to start with]
- [Files to check/modify]
- [Expected outcome]

### Code Snippets to Remember
```python
# Important code patterns or solutions discovered today
```

### Hyperparameters & Settings
- Learning rate:
- Batch size:
- Epochs:
- Augmentations:

---

---

## Day 2 - Phase 2: Baseline Model - First Training Attempt ✓

### Date: 2025-11-04

### What We Did Today

1. **Implemented DeepLabV3+ baseline model** (`deeplabv3plus.py`):
   - Architecture: ResNet50 backbone with pretrained ImageNet weights
   - Modified final classifier layer to predict 13 classes (0-12, ignoring anomaly)
   - Training configuration:
     - Optimizer: Adam with learning rate 1e-4
     - Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
     - Loss: CrossEntropyLoss (ignore_index=13)
     - Batch size: 4
     - Epochs: 30
     - Image size: 512x512
     - Data augmentation: RandomHorizontalFlip, ColorJitter

2. **First training run completed**:
   - **Training time**: 2.5 hours (30 epochs)
   - **Best validation mIoU**: 31-33%
   - Model checkpoint saved: `best_deeplabv3_streethazards.pth` (168 MB)
   - TensorBoard logs: `runs/streethazards_experiment/`
   - GPU: RTX 4080 at ~88% utilization

3. **Created qualitative evaluation script** (`evaluate_qualitative.py`):
   - Generates detailed visualizations for model predictions
   - Shows: Input | Ground Truth | Prediction | Overlay | Confidence Map | Error Map
   - Computes per-sample metrics: mIoU, pixel accuracy, confidence scores
   - Detects and highlights anomalies in test set
   - Creates comparison grids for quick overview

4. **Qualitative evaluation results** (10 samples per split):
   - **Validation set**: Mean mIoU = 38.8% ± 4.8% (range: 33.1% - 47.7%)
   - **Test set**: Mean mIoU = 30.9% ± 9.9% (range: 13.6% - 50.7%)
   - Test performance ~8% lower than validation (expected due to anomalies)
   - Visualizations saved to: `assets/qualitative_eval/`

5. **🐛 CRITICAL BUG DISCOVERED: Auxiliary Classifier Not Enabled**
   - **Problem identified**: During code review, discovered that the auxiliary classifier was not being used during training
   - **Root cause analysis**:
     - Line 63: Only modified main classifier output (`model.classifier[-1]`), but forgot auxiliary classifier
     - Line 101: Only used `model(images)['out']`, ignoring auxiliary output
     - Line 102: Only computed loss on main output
   - **Impact**: Missing out on improved gradient flow to middle layers (~0.5-1% mIoU improvement expected)

6. **Bug fix implemented** (with Claude's assistance):
   - Added `model.aux_classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)` to reconfigure aux output
   - Modified `train_one_epoch()` to compute both main and auxiliary losses
   - Implemented weighted loss combination: `loss = main_loss + 0.4 * aux_loss` (standard 0.4 weight)
   - Added separate tracking for main/aux/total losses in TensorBoard
   - Updated training loop to use new return values

7. **Fixed evaluation script bug**:
   - Issue: `model.load_state_dict()` failed with strict mode due to aux_classifier keys
   - Solution: Added `strict=False` parameter to ignore auxiliary classifier during inference
   - Reasoning: Aux classifier only needed during training, not inference

### Current Status
- **Phase**: Phase 2 (Baseline Model) - First iteration complete, bug fixed
- **Hours Used**: ~6 / 50
- **Files Created**:
  - `deeplabv3plus.py` (training script - now fixed with aux classifier)
  - `evaluate_qualitative.py` (qualitative evaluation tool)
  - `best_deeplabv3_streethazards.pth` (trained model - 168 MB)
  - `assets/qualitative_eval/` (visualizations directory)

- **Files Modified**:
  - `log.md` (this file)

### Training Results Summary

#### Hyperparameters (First Attempt)
- Model: DeepLabV3+ (ResNet50 backbone)
- Learning rate: 1e-4 (Adam)
- Batch size: 4
- Epochs: 30
- Image size: 512x512
- Augmentations: RandomHorizontalFlip(0.5), ColorJitter(0.3)
- Loss: CrossEntropyLoss (ignore anomaly class 13)
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)

#### Performance Metrics
| Split | Mean mIoU | Std Dev | Min IoU | Max IoU | Notes |
|-------|-----------|---------|---------|---------|-------|
| Validation (all) | 31-33% | - | - | - | Reported during training |
| Validation (10 samples) | 38.8% | ±4.8% | 33.1% | 47.7% | From qualitative eval |
| Test (10 samples) | 30.9% | ±9.9% | 13.6% | 50.7% | Lower due to anomalies |

**Note**: The 10-sample evaluation shows higher mIoU (38.8%) than the full validation average (31-33%), suggesting the sampled images may have been easier than the overall distribution.

### Challenges & Solutions

#### Challenge 1: Model Performance Below Target
- **Expected**: 40-50% mIoU on validation
- **Achieved**: 31-33% mIoU on validation (~20% below target)
- **Possible causes**:
  - Bug: Auxiliary classifier not enabled (missing ~0.5-1% improvement)
  - Dataset difficulty: StreetHazards has 13 classes with significant class imbalance
  - Hyperparameters: May need tuning (longer training, different LR, batch size)
  - Augmentation: Current augmentation may be insufficient
- **Solution**: Bug fixed. Next training run should show improvement.

#### Challenge 2: Loading Model Checkpoint Failed
- **Problem**: `RuntimeError: Unexpected key(s) in state_dict: "aux_classifier.*"`
- **Cause**: Trained model has aux_classifier weights, but inference model was initialized without modifying aux_classifier
- **Solution**: Use `model.load_state_dict(state_dict, strict=False)` to ignore aux_classifier keys during inference

#### Challenge 3: Understanding Auxiliary Classifier Purpose
- **Question**: Why does DeepLabV3 have an auxiliary classifier?
- **Answer**: Provides additional supervision signal from intermediate layers (ResNet block 3)
  - Helps gradients flow better through deep networks
  - Reduces vanishing gradient problem
  - Training loss = main_loss + 0.4 × aux_loss
  - Only used during training; ignored during inference
  - Expected improvement: ~0.5-1% mIoU (based on literature)

### Learning Outcomes & Insights 📚

> **Note**: Learning is a primary objective of this assignment. The process of discovery, debugging, and understanding is as valuable as the final results.

**Key lessons from this session:**

1. **Always verify training details**: Initially missed that auxiliary classifier wasn't being used. Code review revealed the bug.

2. **Understanding architectural components**: Learned about auxiliary classifiers in semantic segmentation:
   - Purpose: Improve gradient flow in deep networks
   - Implementation: Extra prediction head at intermediate layer
   - Training: Weighted sum of main and auxiliary losses
   - Inference: Only use main output

3. **Model evaluation techniques**: Created comprehensive qualitative evaluation:
   - Visual inspection reveals failure modes (misclassified regions, boundary errors)
   - Confidence maps show model uncertainty
   - Error maps highlight systematic mistakes
   - Important complement to quantitative metrics

4. **Debugging deep learning code**: Systematic approach to bug discovery:
   - Check model architecture matches training configuration
   - Verify all model components are being used
   - Compare implementation to official documentation/papers
   - Use visualization to understand model behavior

5. **Performance expectations**: Initial results (31-33% mIoU) below target:
   - Not necessarily bad - dataset may be harder than expected
   - Bug discovery explains partial gap
   - Multiple factors affect performance (architecture, hyperparameters, data)
   - Iterative improvement is normal in deep learning

### Next Steps

**Immediate priority**: Retrain model with bug fix
- Expected improvement: 32-35% mIoU (with proper aux classifier)
- Same hyperparameters for fair comparison
- Will take another ~2.5 hours

**After retraining**:
- Run qualitative evaluation on new model
- Compare results with first attempt
- Decide: acceptable baseline or tune hyperparameters?

**Then proceed to Phase 4**: Simple Anomaly Detection
- Implement Standardized Max Logits method
- No retraining required
- Expected test AUPR: 15-25%

### Code Snippets to Remember

#### Proper Auxiliary Classifier Setup
```python
# Model initialization (CORRECT)
model = deeplabv3_resnet50(weights='DEFAULT')
model.classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)  # Main classifier
model.aux_classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)  # Auxiliary classifier (IMPORTANT!)

# Training loop (CORRECT)
output_dict = model(images)
main_output = output_dict['out']
aux_output = output_dict['aux']

main_loss = loss_fn(main_output, masks)
aux_loss = loss_fn(aux_output, masks)
loss = main_loss + 0.4 * aux_loss  # Weighted combination

# Inference (CORRECT)
model.eval()
with torch.no_grad():
    output = model(images)['out']  # Only use main output, ignore aux
```

#### Loading Model with Auxiliary Classifier
```python
# If model was trained with aux_classifier but you only need main output
model.load_state_dict(state_dict, strict=False)  # Ignores aux_classifier keys
```

### Saved Artifacts

**Model checkpoints:**
- `best_deeplabv3_streethazards.pth` - 168 MB, validation mIoU 31-33%

**Visualizations:**
- `assets/qualitative_eval/validation/` - 10 validation sample predictions
- `assets/qualitative_eval/test/` - 10 test sample predictions (with anomalies)
- `assets/qualitative_eval/validation_comparison_grid.png` - Overview of validation results
- `assets/qualitative_eval/test_comparison_grid.png` - Overview of test results

**Logs:**
- `runs/streethazards_experiment/` - TensorBoard logs (30 epochs)

### Time Tracking
- Data pipeline setup: 2 hours (Day 1)
- Model implementation: 1 hour
- First training run: 2.5 hours
- Qualitative evaluation: 0.5 hours
- Bug discovery & fix: 0.5 hours
- **Total so far**: ~6.5 / 50 hours

---

*Last updated: 2025-11-04*
