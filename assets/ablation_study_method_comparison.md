# Model vs Anomaly Detection Method Comparison

## Main Results

**Format**: FPR95 / AUROC / AUPR / F1 (all in %)

**Lower FPR95 is better**, **Higher AUROC/AUPR/F1 is better**

| Model                                           | Simple Max Logits        |
|:------------------------------------------------|:-------------------------|
| No Aug (56.29% mIoU) Ablation                   | 38.3 / 88.5 / 6.6 / 13.5 |
| +Scale (51.76% mIoU) Ablation                   | 27.3 / 91.4 / 8.5 / 15.6 |
| +Scale+Rotate (50.43% mIoU) Ablation            | 30.1 / 90.4 / 7.6 / 14.2 |
| +Scale+Rotate+Flip (49.01% mIoU) Ablation       | 34.5 / 89.4 / 7.1 / 13.7 |
| +Scale+Rotate+Flip+Color (48.13% mIoU) Ablation | 30.6 / 90.8 / 8.4 / 15.7 |

## F1 Scores (%) - For Threshold Selection

**Higher is better**. This is the metric used to find optimal thresholds.

| Model                                           |   Simple Max Logits |
|:------------------------------------------------|--------------------:|
| No Aug (56.29% mIoU) Ablation                   |               13.48 |
| +Scale (51.76% mIoU) Ablation                   |               15.61 |
| +Scale+Rotate (50.43% mIoU) Ablation            |               14.18 |
| +Scale+Rotate+Flip (49.01% mIoU) Ablation       |               13.75 |
| +Scale+Rotate+Flip+Color (48.13% mIoU) Ablation |               15.73 |

## Optimal Thresholds

These are the thresholds that maximize F1 score for each method/model combination.

| Model                                           |   Simple Max Logits |
|:------------------------------------------------|--------------------:|
| No Aug (56.29% mIoU) Ablation                   |             -2.2411 |
| +Scale (51.76% mIoU) Ablation                   |             -1.9271 |
| +Scale+Rotate (50.43% mIoU) Ablation            |             -1.912  |
| +Scale+Rotate+Flip (49.01% mIoU) Ablation       |             -2.4909 |
| +Scale+Rotate+Flip+Color (48.13% mIoU) Ablation |             -1.8121 |

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
