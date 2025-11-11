# Anomaly Detection Methods

This directory contains implementations of various anomaly detection methods for out-of-distribution (OOD) pixel detection in semantic segmentation.

## Methods Implemented

### 1. Simple Max Logits (SML) - **RECOMMENDED**

**File**: `simple_max_logits.py`

- **Formula**: `anomaly_score = -max(logits)`
- **Performance**: AUROC 90.50%, AUPR 8.43%, FPR95 33.12%
- **Ranking**: #1 (Best overall)
- **Description**: Uses only the maximum logit value as anomaly score. Simplest and most effective baseline.

```bash
.venv/bin/python3 anomaly_detection/simple_max_logits.py
```

**When to use**:
- Default choice for anomaly detection
- Best precision-recall trade-off (AUPR 8.43%)
- Trivial implementation, minimal computation
- Robust to moderate domain shifts

### 2. Energy Score

**File**: `energy_score_anomaly_detection.py`

- **Formula**: `E(x) = -T * LogSumExp(logits / T)`
- **Performance**: AUROC 90.61%, AUPR 8.32%, FPR95 33.08%
- **Ranking**: #2
- **Description**: Energy-based method using logarithmic sum of exponentials. Theoretically grounded but provides no practical advantage over Simple Max Logits for this task.
- **Reference**: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020

```bash
.venv/bin/python3 anomaly_detection/energy_score_anomaly_detection.py
```

**Key finding**: Energy Score ≈ Simple Max Logits (equivalent performance)
- In semantic segmentation, max logit dominates LogSumExp
- Well-calibrated models produce peaked distributions
- Energy Score adds computational cost with no benefit

### 3. Maximum Softmax Probability (MSP)

**File**: `maximum_softmax_probability.py`

- **Formula**: `anomaly_score = -max(softmax(logits))`
- **Performance**: AUROC 86.71%, AUPR 6.21%, FPR95 33.57%
- **Ranking**: #3
- **Description**: Normalized confidence score using softmax. Commonly used baseline but underperforms raw logits.

```bash
.venv/bin/python3 anomaly_detection/maximum_softmax_probability.py
```

**Why MSP underperforms**:
- Softmax normalization compresses confidence scores
- Reduces separation between in-distribution and OOD
- Information loss from normalization
- Conclusion: Keep logits, don't use softmax for OOD detection

### 4. Standardized Max Logits (SML)

**File**: `standardized_max_logits.py`

- **Formula**: `anomaly_score = -(logits - μ_c) / σ_c`
- **Performance**: AUROC 80.25%, AUPR 5.41%, FPR95 83.91% ⚠️
- **Ranking**: #4 (Worst)
- **Description**: Class-wise normalization using validation statistics. **NOT RECOMMENDED** - catastrophic failure under domain shift.

```bash
.venv/bin/python3 anomaly_detection/standardized_max_logits.py
```

**Why SML fails**:
- FPR95 = 84%: Need 84% false alarms to detect 95% of anomalies!
- Validation statistics computed on different domain
- Test domain shift invalidates normalization parameters
- Class means/stds don't transfer to test distribution

### 5. HEAT (Hybrid Energy-Adaptive Thresholding)

**File**: `heat_anomaly_detection.py`

- **Components**: Energy Score + Mahalanobis Distance + Spatial Consistency + Adaptive Normalization
- **Performance**: AUROC 89.43%, AUPR 9.15%, FPR95 33.06%
- **Ranking**: Mixed (best AUPR but lower AUROC than Simple Max Logits)
- **Description**: Advanced multi-component anomaly detection combining multiple signals.

```bash
.venv/bin/python3 anomaly_detection/heat_anomaly_detection.py
```

**HEAT components**:
1. **Energy Score**: `-T * LogSumExp(logits/T)`
2. **Mahalanobis Distance**: Feature-space outlier detection
3. **Spatial Consistency**: KL divergence with neighborhood
4. **Adaptive Normalization**: EMA-based test-time adaptation

**Results vs Simple Max Logits**:
- AUPR: +0.72% improvement (9.15% vs 8.43%)
- AUROC: -1.07% degradation (89.43% vs 90.50%)
- Conclusion: Modest improvement, not worth complexity

## Usage

All scripts can be run from the project root:

```bash
# From project root
.venv/bin/python3 anomaly_detection/<script_name>.py
```

Each script:
- Loads the trained model from `config.MODEL_PATH`
- Evaluates on StreetHazards test set
- Saves results to `assets/anomaly_detection/`
- Generates comprehensive metrics (AUROC, AUPR, FPR95)

## Results Summary

### Performance Ranking (by AUPR - primary metric)

| Rank | Method | AUROC | AUPR | FPR95 | Status |
|------|--------|-------|------|-------|--------|
| **#1** | **Simple Max Logits** | **90.50%** | **8.43%** | **33.12%** | ⭐ **RECOMMENDED** |
| #2 | Energy Score | 90.61% | 8.32% | 33.08% | Equivalent to SML |
| #3 | HEAT | 89.43% | 9.15% | 33.06% | Complex, modest gain |
| #4 | MSP | 86.71% | 6.21% | 33.57% | Softmax hurts |
| #5 | Standardized ML | 80.25% | 5.41% | 83.91% | ⚠️ Fails under shift |

**Baseline (StreetHazards paper)**: AUROC 89.30%, AUPR 10.60%, FPR95 26.50%

## Metrics Interpretation

### AUROC (Area Under ROC Curve)
- Measures ranking quality across all thresholds
- Range: 0.5 (random) to 1.0 (perfect)
- Our best: 90.61% (Energy Score)
- Interpretation: Model ranks 90.6% of (anomaly, normal) pairs correctly

### AUPR (Area Under Precision-Recall Curve)
- **PRIMARY METRIC** for imbalanced anomaly detection
- Better than AUROC when anomalies are rare (<2%)
- Our best: 9.15% (HEAT)
- Random baseline: 1.03% (class frequency)
- 9× better than random, but still low absolute performance

### FPR95 (False Positive Rate at 95% TPR)
- Practical metric: "Cost of high-recall operation"
- Our best: 33.06% (HEAT)
- Interpretation: To catch 95% of anomalies, 33% of normal pixels will be falsely flagged
- For autonomous driving: 33% false alarm rate may be too high

## Configuration

All methods use configuration from `config.py`:

```python
MODEL_PATH = 'models/checkpoints/deeplabv3_resnet50_augmented_10_47_09-11-25_mIoU_5026.pth'
MODEL_ARCHITECTURE = 'deeplabv3_resnet50'
ANOMALY_THRESHOLD = -1.4834  # Optimal threshold from simple_max_logits.py
MAX_PIXELS_EVALUATION = 1_000_000  # Subsample for memory efficiency
RANDOM_SEED = 42
```

## Key Findings

### Finding 1: Energy Score ≈ Simple Max Logits
- **Difference**: AUROC +0.11%, AUPR -0.11% (negligible)
- **Reason**: In segmentation, max logit dominates LogSumExp
- **Conclusion**: Use Simple Max Logits (simpler, faster)

### Finding 2: Softmax Normalization Hurts
- MSP underperforms by -3.79% AUROC vs Max Logits
- Softmax compresses confidence scores
- Keep raw logits for better OOD detection

### Finding 3: Domain Shift Kills SML
- SML: 83.91% FPR95 (unusable!)
- Validation statistics don't transfer to test
- Energy Score and Max Logits more robust

### Finding 4: HEAT Shows Modest Improvement
- +0.72% AUPR over Simple Max Logits
- But -1.07% AUROC degradation
- Not worth the implementation complexity

## Recommendations

### For Deployment
✅ **Use Simple Max Logits**
- Best AUPR (8.43%)
- Simplest implementation
- Fastest computation
- Energy Score provides no benefit

❌ **Avoid Standardized Max Logits**
- Fails catastrophically under domain shift
- Validation statistics don't transfer

### For Improvement
1. **Test-time adaptation (ATTA)**
   - Adapt batch normalization at test time
   - May reduce FPR95 from 33% → 26%

2. **Threshold calibration**
   - Tune threshold on validation set
   - Optimize for specific precision/recall trade-off

3. **Investigate better models**
   - Does better segmentation mIoU → better anomaly detection?
   - Compare multiple checkpoints

## Implementation Complexity

| Method | Implementation | Computation | Memory | Robustness |
|--------|---------------|-------------|--------|------------|
| Simple Max Logits | Trivial (1 line) | Minimal | Minimal | Moderate |
| Energy Score | Simple (2 lines) | Low | Minimal | Moderate |
| MSP | Simple (2 lines) | Low | Minimal | Moderate |
| SML | Complex | Moderate | Low | **Poor** |
| HEAT | Very Complex | High | High | Good |

**Recommendation**: Start simple (Max Logits), upgrade only if needed.

## Output Files

Results are saved to `assets/anomaly_detection/`:

- `simple_max_logits_results.txt` - Simple Max Logits metrics
- `energy_score_results.txt` - Energy Score metrics
- `maximum_softmax_probability_results.txt` - MSP metrics
- `sml_results.txt` - Standardized Max Logits metrics
- `heat_results.txt` - HEAT metrics
- `comprehensive_method_comparison.txt` - Complete comparison and analysis

## References

1. Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020
   https://arxiv.org/abs/2010.03759

2. Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and OOD Examples", ICLR 2017

3. Tian et al., "Pixel-Wise Energy-Biased Abstention Learning for Anomaly Segmentation", ECCV 2022

4. Lee et al., "A Simple Unified Framework for Detecting OOD Samples and Adversarial Attacks", NeurIPS 2018

5. Chan et al., "Entropy Maximization and Meta Classification for OOD Detection", ICCV 2021

## Notes

- All scripts are standalone and don't import each other
- Memory-efficient evaluation with random pixel subsampling
- Float16 precision for anomaly scores
- Supports CUDA acceleration
- Pre-computed results included in assets/
