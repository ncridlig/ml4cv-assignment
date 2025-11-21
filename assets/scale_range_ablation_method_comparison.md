# Model vs Anomaly Detection Method Comparison

## Main Results

**Format**: FPR95 / AUROC / AUPR / F1 (all in %)

**Lower FPR95 is better**, **Higher AUROC/AUPR/F1 is better**

| Model                                      | Simple Max Logits        |
|:-------------------------------------------|:-------------------------|
| Scale 0.5-1.5 (51.43% mIoU) Zoom-In        | 31.3 / 90.5 / 8.0 / 15.2 |
| Scale 0.75-1.25 (51.37% mIoU) Conservative | 33.6 / 89.8 / 7.4 / 14.2 |
| Scale 0.7-2.0 (51.05% mIoU) Zoom-Out       | 27.9 / 90.7 / 7.7 / 14.4 |
| Scale 0.5-2.0 (49.90% mIoU) Baseline       | 28.1 / 91.2 / 8.9 / 16.2 |
| Scale 0.3-3.0 (49.88% mIoU) Aggressive     | 31.3 / 90.1 / 7.8 / 14.6 |
| Scale 0.4-2.5 (49.55% mIoU) Extended       | 32.0 / 89.8 / 7.5 / 14.1 |
| Scale 0.9-1.1 (49.27% mIoU) Minimal        | 30.5 / 90.6 / 8.2 / 15.1 |

## F1 Scores (%) - For Threshold Selection

**Higher is better**. This is the metric used to find optimal thresholds.

| Model                                      |   Simple Max Logits |
|:-------------------------------------------|--------------------:|
| Scale 0.5-1.5 (51.43% mIoU) Zoom-In        |               15.2  |
| Scale 0.75-1.25 (51.37% mIoU) Conservative |               14.17 |
| Scale 0.7-2.0 (51.05% mIoU) Zoom-Out       |               14.45 |
| Scale 0.5-2.0 (49.90% mIoU) Baseline       |               16.21 |
| Scale 0.3-3.0 (49.88% mIoU) Aggressive     |               14.56 |
| Scale 0.4-2.5 (49.55% mIoU) Extended       |               14.13 |
| Scale 0.9-1.1 (49.27% mIoU) Minimal        |               15.13 |

## Optimal Thresholds

These are the thresholds that maximize F1 score for each method/model combination.

| Model                                      |   Simple Max Logits |
|:-------------------------------------------|--------------------:|
| Scale 0.5-1.5 (51.43% mIoU) Zoom-In        |             -2.0226 |
| Scale 0.75-1.25 (51.37% mIoU) Conservative |             -2.2975 |
| Scale 0.7-2.0 (51.05% mIoU) Zoom-Out       |             -2.1721 |
| Scale 0.5-2.0 (49.90% mIoU) Baseline       |             -1.9777 |
| Scale 0.3-3.0 (49.88% mIoU) Aggressive     |             -2.2651 |
| Scale 0.4-2.5 (49.55% mIoU) Extended       |             -2.0767 |
| Scale 0.9-1.1 (49.27% mIoU) Minimal        |             -1.7982 |

---

## Interpretation

- **FPR95**: False Positive Rate at 95% True Positive Rate (lower is better)
  - To detect 95% of anomalies, what % of normal pixels are flagged as anomalies?
- **AUROC**: Area Under ROC Curve (higher is better, 50% = random, 100% = perfect)
  - Measures overall ranking quality across all thresholds
- **AUPR**: Area Under Precision-Recall Curve (higher is better)
  - Primary metric for imbalanced data (~1% anomaly rate)
- **F1**: Harmonic mean of precision and recall at optimal threshold (higher is better)
  - This is what you optimize to find the best operating threshold
  - Best balance between false positives and false negatives
- **Threshold**: The anomaly score threshold that maximizes F1
  - Use this threshold for deployment/inference
