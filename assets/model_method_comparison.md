# Model vs Anomaly Detection Method Comparison

## Main Results

**Format**: FPR95 / AUROC / AUPR / F1 (all in %)

**Lower FPR95 is better**, **Higher AUROC/AUPR/F1 is better**

| Model                                | Simple Max Logits        | Maximum Softmax Probability   | Standardized Max Logits   | Energy Score             | HEAT                     |
|:-------------------------------------|:-------------------------|:------------------------------|:--------------------------|:-------------------------|:-------------------------|
| ResNet50 (50.26% mIoU) Augmented     | 33.1 / 90.5 / 8.4 / 16.4 | 33.9 / 87.5 / 6.3 / 11.7      | 83.9 / 80.2 / 5.4 / 12.2  | 33.1 / 90.6 / 8.3 / 16.2 | 33.1 / 90.6 / 8.3 / 16.2 |
| ResNet50 (37.57% mIoU) Baseline      | 37.3 / 87.6 / 6.2 / 12.0 | 43.0 / 84.7 / 5.5 / 11.6      | 75.9 / 78.4 / 3.7 / 7.5   | 36.7 / 87.6 / 5.9 / 11.5 | 36.7 / 87.6 / 5.9 / 11.5 |
| ResNet101 (37.07% mIoU) Baseline     | 36.1 / 87.7 / 6.3 / 12.1 | 43.2 / 83.7 / 5.0 / 10.5      | 64.7 / 79.0 / 3.8 / 8.0   | 35.6 / 88.0 / 6.3 / 12.0 | 35.6 / 88.0 / 6.3 / 12.0 |
| SegFormer-B5 (35.57% mIoU) Baseline  | 29.8 / 90.9 / 8.4 / 15.0 | 36.3 / 88.0 / 6.9 / 14.0      | 68.0 / 82.7 / 4.7 / 9.9   | 28.5 / 91.1 / 8.3 / 15.0 | 28.5 / 91.1 / 8.3 / 15.0 |
| SegFormer-B5 (54.12% mIoU) Augmented | 68.7 / 83.9 / 7.9 / 15.8 | 53.8 / 86.0 / 6.7 / 12.9      | 79.7 / 74.8 / 4.1 / 10.0  | 68.9 / 83.4 / 7.1 / 14.8 | 68.9 / 83.4 / 7.1 / 14.8 |
| Hiera-Base (32.83% mIoU) 224         | 39.7 / 87.7 / 6.4 / 12.4 | 41.0 / 85.6 / 5.2 / 10.8      | 77.8 / 78.4 / 3.8 / 8.1   | 39.9 / 87.9 / 7.1 / 13.3 | 39.9 / 87.9 / 7.1 / 13.3 |
| Hiera-Large (46.77% mIoU) 224        | 34.5 / 90.0 / 8.6 / 15.9 | 38.3 / 86.8 / 6.0 / 12.6      | 80.2 / 79.3 / 4.5 / 9.4   | 34.4 / 90.1 / 8.9 / 16.3 | 34.4 / 90.1 / 8.9 / 16.3 |

## F1 Scores (%) - For Threshold Selection

**Higher is better**. This is the metric used to find optimal thresholds.

| Model                                |   Simple Max Logits |   Maximum Softmax Probability |   Standardized Max Logits |   Energy Score |   HEAT |
|:-------------------------------------|--------------------:|------------------------------:|--------------------------:|---------------:|-------:|
| ResNet50 (50.26% mIoU) Augmented     |               16.35 |                         11.69 |                     12.17 |          16.17 |  16.17 |
| ResNet50 (37.57% mIoU) Baseline      |               11.98 |                         11.62 |                      7.49 |          11.51 |  11.51 |
| ResNet101 (37.07% mIoU) Baseline     |               12.13 |                         10.52 |                      8.04 |          12.02 |  12.02 |
| SegFormer-B5 (35.57% mIoU) Baseline  |               15.04 |                         13.98 |                      9.86 |          14.96 |  14.96 |
| SegFormer-B5 (54.12% mIoU) Augmented |               15.84 |                         12.92 |                     10.01 |          14.83 |  14.83 |
| Hiera-Base (32.83% mIoU) 224         |               12.37 |                         10.8  |                      8.05 |          13.26 |  13.26 |
| Hiera-Large (46.77% mIoU) 224        |               15.93 |                         12.56 |                      9.37 |          16.28 |  16.28 |

## Optimal Thresholds

These are the thresholds that maximize F1 score for each method/model combination.

| Model                                |   Simple Max Logits |   Maximum Softmax Probability |   Standardized Max Logits |   Energy Score |    HEAT |
|:-------------------------------------|--------------------:|------------------------------:|--------------------------:|---------------:|--------:|
| ResNet50 (50.26% mIoU) Augmented     |             -2.1747 |                       -0.5038 |                    1.9194 |        -2.8143 | -2.8143 |
| ResNet50 (37.57% mIoU) Baseline      |             -1.4835 |                       -0.3893 |                    1.5675 |        -2.3581 | -2.3581 |
| ResNet101 (37.07% mIoU) Baseline     |             -1.5334 |                       -0.4458 |                    1.8664 |        -2.0844 | -2.0844 |
| SegFormer-B5 (35.57% mIoU) Baseline  |             -1.7328 |                       -0.4411 |                    1.9862 |        -2.471  | -2.471  |
| SegFormer-B5 (54.12% mIoU) Augmented |             -1.0063 |                       -0.508  |                    1.9371 |        -1.5593 | -1.5593 |
| Hiera-Base (32.83% mIoU) 224         |             -1.9047 |                       -0.4376 |                    1.7115 |        -2.6357 | -2.6357 |
| Hiera-Large (46.77% mIoU) 224        |             -2.6316 |                       -0.4989 |                    2.2008 |        -3.1592 | -3.1592 |

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
