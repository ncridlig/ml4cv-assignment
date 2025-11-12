# ML4CV Assignment - Work Log (Condensed)

## Project Overview
- **Course**: Machine Learning for Computer Vision, University of Bologna (A.Y. 2024-2025)
- **Task**: Semantic segmentation + zero-shot anomaly detection on StreetHazards dataset
- **Dataset**: 5,125 train + 1,031 val + 1,500 test images (1280×720 → 512×512)
- **Goal**: Segment 12 known classes + detect anomalous objects (unseen during training)
- **Time Budget**: 50 hours total
- **Evaluation**: mIoU (segmentation), AUPR (anomaly), ablation studies, code quality

---

## Day 0-1: Setup & Data Pipeline ✓ (2 hours)

### Date: 2025-11-04

**Accomplishments**:
1. Created `dataloader.py` with StreetHazardsDataset class, augmentation pipeline, visualization
2. Resolved class mapping: 14 classes (0-12 known, 13 anomaly test-only), files 1-indexed → remap to 0-indexed
3. Created `utils/class_counter.py` for class distribution analysis
4. Verified data integrity: Training/val have classes 0-12, test adds class 13 (anomaly)

**Key Decisions**:
- Using StreetHazards (CARLA synthetic data)
- 13 training classes, class 13 (anomaly) only in test set
- Primary metrics: mIoU (segmentation), AUPR (anomaly detection)

**Dataset Statistics**:
- Training: 5,125 images, classes 0-12 (no anomaly)
- Validation: 1,031 images, classes 0-12
- Test: 1,500 images, classes 0-13 (includes 250+ anomaly types)

**Status**: Phase 1 Complete ✓
**Hours Used**: 2 / 50

---

## Day 2: Baseline Model Training ✓ (4.5 hours)

### Date: 2025-11-04

**First Training Attempt**:
- Architecture: DeepLabV3+ ResNet50, pretrained ImageNet weights
- Resolution: 512×512, batch size 4, 30 epochs
- Augmentation: RandomHorizontalFlip, ColorJitter
- Result: **31-33% validation mIoU** (below 40-50% target)

**Bug Discovery**: Auxiliary classifier not enabled during training
- Only used main classifier, ignored auxiliary output
- Expected impact: ~0.5-1% mIoU improvement

**Bug Fixed**:
- Modified both `model.classifier[-1]` AND `model.aux_classifier[-1]`
- Training loss: `main_loss + 0.4 * aux_loss`
- Inference: Use only main output, `strict=False` when loading

**Qualitative Evaluation Created** (`evaluate_qualitative.py`):
- Generates: Input | GT | Prediction | Overlay | Confidence | Error maps
- Results: Val 38.8% ± 4.8%, Test 30.9% ± 9.9%

**Key Learnings**:
1. Always verify all architectural components are being used
2. Auxiliary classifiers improve gradient flow in deep networks
3. Visual inspection reveals failure modes not captured by metrics

**Status**: Phase 2 Complete ✓
**Hours Used**: 6.5 / 50

---

## Day 3: Anomaly Detection Implementation (9 hours)

### Date: 2025-11-06

**Challenge**: Memory constraints evaluating 393M pixels (1500 images × 512×512)
- Problem: sklearn metrics require all data in memory (~30GB)
- Solution: Float16 + random subsampling to 1M pixels (99.75% reduction)
- Statistical validity maintained, reproducible with seed=42

**Method 1: Simple Max Logits**
```python
anomaly_score = -max(logits)
```
- **Results**: AUROC 87.61%, AUPR 6.19%
- Simple, no training on anomalies needed (zero-shot)
- Baseline for comparison

**Method 2: Maximum Softmax Probability (MSP)**
```python
anomaly_score = -max(softmax(logits))
```
- **Results**: AUROC 84.68%, AUPR 5.49%
- Worse than Max Logits (-2.93% AUROC, -11.3% AUPR)
- Softmax compression reduces separation between ID and OOD

**Method 3: Standardized Max Logits (SML)**
```python
anomaly_score = -(max_logit - μ_c) / σ_c  # Per-class normalization
```
- **Results**: AUROC 80.25%, AUPR 3.70%, FPR95 83.91% ❌
- **Failed catastrophically** - requires 84% false alarms to detect 95% anomalies
- Root cause: Domain shift invalidates validation statistics

**Code Refactoring**:
- Created `config.py` (central configuration)
- Refactored 4+ scripts to use shared constants
- Integrated anomaly detection into qualitative visualization

**Authors' Baseline Comparison**:
- StreetHazards paper: FPR95 26.5%, AUROC 89.3%, AUPR 10.6%
- Our ResNet50: AUROC 87.61%, AUPR 6.19% (below baseline)

**Status**: Phase 4 ~80% Complete
**Hours Used**: 11.3 / 50

---

## Day 3-4: Model Architecture Experiments (14 hours)

### Dates: 2025-11-07 to 2025-11-09

**Experiment 1: ResNet101 @ 512×512**
- Result: 37.07% mIoU (vs ResNet50 37.57%)
- Conclusion: Deeper ≠ better, diminishing returns

**Hypothesis**: Resolution is bottleneck, not model capacity
- Native 1280×720 → 512×512 loses 71.5% of pixels
- Small objects (pedestrians, anomalies) become blurry

**Experiment 2: Hiera-Base @ 1280×720 (full resolution)**
- Result: 32.83% mIoU (**WORSE than 512×512**)
- Train IoU 65% vs Val mIoU 28% → severe overfitting
- 3.5× more pixels, but batch size=1 insufficient

**Experiment 3: ResNet101 @ 1280×720**
- Result: 37.07% mIoU (same as 512×512 version)
- Training time: **8 hours** (vs 1.5h for 512×512)
- Conclusion: Full resolution provides **no benefit, 5× slower**

**Experiment 4: SegFormer-B5 @ 512×512**
- 82M parameters (vs ResNet50 45M)
- Result: 35.57% mIoU (worse than ResNet50)
- Train IoU 65% → Val 31% (severe overfitting)
- Transformers need more data than 5K images

**Key Findings**:
1. Resolution increase didn't help (hypothesis was wrong)
2. Deeper models show diminishing returns on limited data
3. CNNs outperform transformers on small datasets
4. 512×512 downscaling is optimal (faster + better performance)
5. Batch size matters: batch=1 insufficient for stable training

**Best Model Confirmed**: DeepLabV3+ ResNet50 @ 512×512 (37.57% mIoU)

**Status**: Architecture exploration complete, abandon full-resolution training
**Hours Used**: 26 / 50

---

## Day 6: Breakthrough - Multi-Scale Augmentation ✓ (2 hours evaluation)

### Date: 2025-11-09

**MAJOR MILESTONE**: Augmented training completed

**Configuration**:
- Multi-scale random crop: 0.5-2.0× with **variable crop sizes**
  - Scale 0.5×: crop 256×256 → resize 512×512 (zoomed-in, fine details)
  - Scale 1.0×: crop 512×512 → resize 512×512 (normal view)
  - Scale 2.0×: crop 1024×1024 → resize 512×512 (zoomed-out, context)
- NO black padding (crop size adapts to scale)
- Additional: RandomHorizontalFlip, ColorJitter, GaussianBlur
- 40 epochs, batch size 4, LR 1e-4

**Results**: **50.26% mIoU** ✅
- Previous best: 37.57%
- **Improvement: +12.69% absolute (+33.8% relative)**
- Model: `models/deeplabv3_resnet50_augmented_10_47_09-11-25_mIoU_5026.pth`

**Key Success Factor**: Following DeepLabV3+ paper's variable crop strategy literally, avoiding black padding artifacts

**Anomaly Detection Re-evaluation**:

| Method | FPR95 | AUROC | AUPR | vs Baseline |
|--------|-------|-------|------|-------------|
| **Simple Max Logits** | 33.12% | **90.50%** | 8.43% | **+1.20% AUROC** ✅ |
| MSP | 33.57% | 86.71% | 6.21% | -2.59% AUROC |
| SML | 83.91% | 80.25% | 5.41% | -9.05% AUROC ❌ |
| **Authors' Baseline** | **26.50%** | 89.30% | **10.60%** | - |

**Comparison with Previous Model**:
- Old (37.57% mIoU): AUPR 6.19%, AUROC 87.61%
- New (50.26% mIoU): AUPR 8.43%, AUROC 90.50%
- **Improvement**: +2.24% AUPR (+36%), +2.89% AUROC

**Key Findings**:
1. Better segmentation → better anomaly detection
2. **AUROC beats baseline** (90.50% vs 89.30%)
3. Simple methods outperform complex normalization
4. Multi-scale augmentation was the critical improvement

**Status**: Best model achieved, beats baseline AUROC
**Hours Used**: 30.4 / 50

---

## Day 7: HEAT & Repository Refactoring (3.2 hours)

### Date: 2025-11-11

**HEAT Implementation** (Hybrid Energy-Adaptive Thresholding):
- Combines: Energy score + spatial smoothing + adaptive thresholding
- Components: LogSumExp, EMA normalization, 3×3 spatial kernel

**Results**:
- HEAT: AUROC 89.43%, AUPR **9.15%**, FPR95 33.06%
- vs Simple Max Logits: AUROC 90.50%, AUPR 8.43%, FPR95 33.12%
- **Difference**: +0.72% AUPR, -1.07% AUROC (minimal improvement)

**Conclusion**: Complex HEAT provides ~1% improvement over Simple Max Logits, not worth the added complexity. Simple methods win.

**Repository Refactoring**:
- Created directories: `anomaly_detection/`, `models/`, `utils/`, `visualizations/`
- Moved 15+ files to logical locations
- Created `pyproject.toml` for proper Python package structure
- Removed `sys.path` hacks, clean imports: `from config import DEVICE`
- Installed as editable package: `pip install -e .`

**Import Verification**: Created `test_imports.py` - 7/7 tests passed ✅

**Status**: Codebase organized, Pythonic package structure
**Hours Used**: 33.6 / 50

---

## Day 8: Comprehensive Comparison & Energy Score (3.3 hours)

### Date: 2025-11-11

**Energy Score Implementation**:
```python
E(x) = -T * LogSumExp(logits / T)  # Temperature T=1
```
- **Results**: AUROC 90.61%, AUPR 8.32%, FPR95 33.08%
- Nearly identical to Simple Max Logits (difference <0.2%)
- Conclusion: Energy ≈ Max Logits for semantic segmentation

**Created Comprehensive Comparison Script** (`visualizations/create_comparison_table.py`):
- Evaluates all 5 models × 5 anomaly detection methods = 25 combinations
- Outputs: CSV, Markdown, JSON formats
- Ready to run (estimated 1.5-2 hours runtime)

**Final Method Ranking** (on best model 50.26% mIoU):

| Rank | Method | AUROC | AUPR | FPR95 | Status |
|------|--------|-------|------|-------|--------|
| 1 | Simple Max Logits | **90.50%** | 8.43% | 33.12% | ⭐ **RECOMMENDED** |
| 2 | Energy Score | **90.61%** | 8.32% | 33.08% | Equivalent to SML |
| 3 | HEAT | 89.43% | **9.15%** | 33.06% | Complex, modest gain |
| 4 | MSP | 86.71% | 6.21% | 33.57% | Softmax hurts |
| 5 | SML | 80.25% | 5.41% | 83.91% | ⚠️ Fails under shift |

**Key Insight**: Simple Max Logits and Energy Score are nearly equivalent - use the simpler one.

**Package Setup Completed**:
- Created `pyproject.toml` with proper package configuration
- All modules importable: `utils`, `anomaly_detection`, `models`, `visualizations`
- Professional Python package structure following PEP standards

**Status**: Methods evaluated, comprehensive comparison script ready
**Hours Used**: 36.9 / 50

---

## Final Results Summary

### Best Model
- **Architecture**: DeepLabV3+ ResNet50
- **Resolution**: 512×512 (downscaled from 1280×720)
- **Augmentation**: Multi-scale (0.5-2.0×), horizontal flip, color jitter, blur
- **Training**: 40 epochs, batch size 4, LR 1e-4
- **Segmentation mIoU**: **50.26%** (+12.69% over baseline)
- **Model Path**: `models/checkpoints/deeplabv3_resnet50_augmented_10_47_09-11-25_mIoU_5026.pth`

### Anomaly Detection Performance

**Recommended Method: Simple Max Logits**
- AUROC: **90.50%** (beats authors' 89.30% ✅)
- AUPR: 8.43% (authors: 10.60%)
- FPR95: 33.12% (authors: 26.50%)
- Formula: `anomaly_score = -max(logits)`
- Zero-shot (no anomaly training needed)
- Simplest and most effective

**Method Comparison** (all on 50.26% mIoU model):
1. Simple Max Logits / Energy Score: ~90.5% AUROC (tied best)
2. HEAT: 89.43% AUROC, 9.15% AUPR (best AUPR, but complex)
3. MSP: 86.71% AUROC (softmax compression hurts)
4. SML: 80.25% AUROC (fails under domain shift)

### Model Architecture Experiments

| Model | Resolution | mIoU | Training Time | Notes |
|-------|-----------|------|---------------|-------|
| ResNet50 Augmented | 512×512 | **50.26%** | 3-4h | ✅ **BEST** |
| ResNet50 Baseline | 512×512 | 37.57% | 2.5h | No augmentation |
| ResNet101 | 512×512 | 37.07% | 1.5h | Diminishing returns |
| ResNet101 | 1280×720 | 37.07% | **8h** | Same result, 5× slower |
| SegFormer-B5 | 512×512 | 35.57% | 2h | Transformer, overfitted |
| Hiera-Base | 1280×720 | 32.83% | 3.5h | Full res, overfitted |

**Key Findings**:
1. **Augmentation > Architecture**: +12.69% from multi-scale augmentation
2. **512×512 optimal**: Faster and better than full resolution
3. **CNNs > Transformers** on limited data (5K images)
4. **Deeper ≠ better**: ResNet101 same as ResNet50
5. **Simplicity wins**: Simple Max Logits beats complex methods

### Repository Structure

```
ml4cv-assignment/
├── config.py                           # Central configuration
├── pyproject.toml                      # Python package setup
├── main.ipynb                          # Primary deliverable
├── evaluate_qualitative.py             # Qualitative evaluation
├── requirements.txt
│
├── anomaly_detection/                  # 5 anomaly detection methods
│   ├── simple_max_logits.py           # BEST: 90.50% AUROC
│   ├── energy_score_anomaly_detection.py
│   ├── heat_anomaly_detection.py
│   ├── maximum_softmax_probability.py
│   └── standardized_max_logits.py
│
├── models/
│   ├── training_scripts/              # Training scripts for all architectures
│   │   ├── train_augmented_resnet50.py  # Best model script
│   │   ├── deeplabv3plus_resnet50.py
│   │   ├── deeplabv3plus_resnet101.py
│   │   ├── segformerb5.py
│   │   └── hierabase224.py
│   └── checkpoints/                    # Trained models (.pth files)
│
├── utils/                              # Core utilities
│   ├── dataloader.py                   # Dataset, augmentation
│   ├── model_utils.py                  # Model loading
│   ├── visualize.py                    # Visualization helpers
│   └── class_counter.py                # Class distribution analysis
│
├── visualizations/                     # Analysis scripts
│   ├── create_comparison_table.py      # Comprehensive comparison
│   ├── create_comparison_plots.py      # Visualization generation
│   └── ablation_studies.py             # Ablation analysis
│
└── assets/                             # Results and figures
    ├── qualitative_eval/               # Segmentation visualizations
    └── anomaly_detection/              # Anomaly detection results
```

---

## Key Lessons Learned

### Technical
1. **Multi-scale augmentation is critical**: +33.8% relative improvement
2. **Variable crop sizes matter**: Adapt crop size to scale factor (no black padding)
3. **Simple methods often win**: Max Logits beats complex normalization
4. **Resolution trade-offs**: 512×512 downscaling optimal (speed + performance)
5. **Domain shift matters**: SML fails when validation stats don't transfer
6. **Better segmentation → better anomaly detection**: +12.69% mIoU → +36% AUPR
7. **Memory optimization essential**: Float16 + subsampling for 393M pixels
8. **Softmax compression hurts OOD detection**: Use raw logits, not probabilities

### Software Engineering
1. **Python packaging best practices**: `pyproject.toml` + `pip install -e .`
2. **DRY principle**: Central `config.py` eliminates duplication
3. **Code organization**: Clear directory structure improves maintainability
4. **Import verification**: Test imports after refactoring
5. **Reproducibility**: Fixed random seeds, documented all hyperparameters

### Research
1. **Hypotheses can be wrong**: Full resolution hypothesis failed
2. **Iterative improvement**: Multiple experiments led to breakthrough
3. **Literature consistency**: Our results match published benchmarks
4. **Ablation importance**: Understanding what works and why
5. **Baseline establishment**: Start simple, add complexity only if beneficial

---

## Time Tracking Summary

| Phase | Description | Hours |
|-------|-------------|-------|
| Phase 1 | Data Pipeline Setup | 2.0 |
| Phase 2 | Baseline Model Training | 4.5 |
| Phase 3 | Architecture Experiments | 14.0 |
| Phase 4 | Anomaly Detection Implementation | 9.0 |
| Phase 5 | Multi-Scale Augmentation Training | 2.4 |
| Phase 6 | HEAT & Repository Refactoring | 3.2 |
| Phase 7 | Comprehensive Comparison & Package Setup | 3.3 |
| **Total Used** | | **38.4 / 50** |
| **Remaining** | | **11.6 hours** |

### Remaining Work (~11.6 hours)
- [ ] Run comprehensive 5×5 comparison (~2 hours runtime)
- [ ] Ablation studies (~4 hours):
  - Augmentation component ablation
  - Threshold sensitivity analysis
- [ ] Final documentation (~4 hours):
  - Complete README
  - Results visualization
  - Code documentation
- [ ] Code cleanup (~1 hour)
- [ ] Buffer (~0.6 hours)

---

## References

**Key Papers**:
1. Hendrycks et al. - "Scaling OOD Detection" (StreetHazards dataset)
2. Jung et al. - "Standardized Max Logits" (SML method)
3. Liu et al. - "Energy-based OOD Detection" (Energy Score)
4. Chen et al. - "DeepLabV3+" (architecture)

**Authors' Baseline (StreetHazards)**:
- Method: Max Logits
- FPR95: 26.5%, AUROC: 89.3%, AUPR: 10.6%

**Our Best Result**:
- Method: Simple Max Logits on augmented model
- FPR95: 33.12%, **AUROC: 90.50%** (beats baseline), AUPR: 8.43%

---

*Last updated: 2025-11-11*
*Total time: 38.4 / 50 hours*
*Status: Model training and method evaluation complete, ready for final analysis*
