"""
Energy Score Anomaly Detection

Based on: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020
https://arxiv.org/abs/2010.03759

Energy Score: E(x) = -T * log(sum_c exp(z_c / T))
              = -T * LogSumExp(z / T)

Lower energy indicates higher confidence in in-distribution (ID) samples.
Higher energy indicates out-of-distribution (OOD) samples.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from dataloader import StreetHazardsDataset, get_transforms
from utils.model_utils import load_model
from config import (
    DEVICE,
    MODEL_PATH,
    NUM_CLASSES,
    ANOMALY_CLASS_IDX,
    OUTPUT_DIR_ANOMALY as OUTPUT_DIR,
    MAX_PIXELS_EVALUATION as MAX_PIXELS,
    RANDOM_SEED,
    IMAGE_SIZE,
    TEST_ROOT
)

print("="*60)
print("ENERGY SCORE ANOMALY DETECTION")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Anomaly class index: {ANOMALY_CLASS_IDX}")
print(f"Max pixels for evaluation: {MAX_PIXELS:,} (random subsampling)")

# -----------------------------
# ENERGY SCORE COMPUTATION
# -----------------------------
def compute_energy_score(logits, temperature=1.0):
    """
    Compute energy score for OOD detection.

    Energy = -T * log(sum(exp(z_c / T)))
           = -T * LogSumExp(z / T)

    Args:
        logits: (B, C, H, W) - model logits
        temperature: float - temperature parameter (default: 1.0)

    Returns:
        energy: (B, H, W) - energy scores
                Lower energy = more in-distribution
                Higher energy = more out-of-distribution
    """
    # Use logsumexp for numerical stability
    # LogSumExp(z/T) = max(z/T) + log(sum(exp(z/T - max(z/T))))
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return energy

# -----------------------------
# ANOMALY DETECTION METHOD
# -----------------------------
@torch.no_grad()
def detect_anomalies_energy_score(model, dataloader, device, temperature=1.0):
    """
    Method: Energy Score
    energy[i] = -T * log(sum_c exp(logits[i,c] / T))
    anomaly_score[i] = energy[i]  (higher energy = more anomalous)

    Args:
        temperature: Temperature parameter for energy computation
                     Higher T = smoother score distribution
                     Lower T = sharper score distribution
    """
    print(f"\n{'='*60}")
    print("METHOD: ENERGY SCORE")
    print(f"{'='*60}")
    print(f"Temperature: {temperature}")

    model.eval()

    all_anomaly_scores = []
    all_ground_truth = []

    for images, masks, _ in tqdm(dataloader, desc="Energy Score"):
        images = images.to(device)
        masks = masks.numpy()  # (B, H, W)

        # Get predictions
        output = model(images)['out']  # (B, 13, H, W)

        # Compute energy scores
        energy = compute_energy_score(output, temperature=temperature)  # (B, H, W)
        energy = energy.cpu().numpy()

        # Anomaly score = energy (use float16 for memory efficiency)
        # Higher energy = more anomalous
        anomaly_scores = energy.astype(np.float16)  # (B, H, W)

        # Ground truth: 1 if anomaly (class 13), 0 otherwise
        ground_truth = (masks == ANOMALY_CLASS_IDX).astype(int)

        # Flatten and collect
        all_anomaly_scores.append(anomaly_scores.flatten())
        all_ground_truth.append(ground_truth.flatten())

    # Concatenate all batches
    all_anomaly_scores = np.concatenate(all_anomaly_scores)
    all_ground_truth = np.concatenate(all_ground_truth)

    total_pixels = len(all_ground_truth)
    print(f"Total pixels: {total_pixels:,}")
    print(f"Anomaly pixels: {all_ground_truth.sum():,} ({100*all_ground_truth.mean():.2f}%)")

    # Random subsampling if needed (for memory efficiency)
    if total_pixels > MAX_PIXELS:
        print(f"\nSubsampling {MAX_PIXELS:,} pixels from {total_pixels:,} (ratio: {MAX_PIXELS/total_pixels:.2%})")
        np.random.seed(RANDOM_SEED)  # For reproducibility
        indices = np.random.choice(total_pixels, size=MAX_PIXELS, replace=False)
        all_anomaly_scores = all_anomaly_scores[indices]
        all_ground_truth = all_ground_truth[indices]
        print(f"After subsampling - Anomaly pixels: {all_ground_truth.sum():,} ({100*all_ground_truth.mean():.2f}%)")
    else:
        print(f"No subsampling needed (total pixels <= {MAX_PIXELS:,})")

    return all_anomaly_scores, all_ground_truth

# -----------------------------
# EVALUATION METRICS
# -----------------------------
def evaluate_anomaly_detection(anomaly_scores, ground_truth, method_name):
    """
    Evaluate anomaly detection performance.

    Metrics:
    - AUROC: Area Under ROC Curve (overall ranking quality)
    - AUPR: Area Under Precision-Recall Curve (primary metric for imbalanced data)
    - FPR95: False Positive Rate at 95% True Positive Rate (cost of high-recall operation)
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION: {method_name}")
    print(f"{'='*60}")

    # Remove NaN/Inf if any
    valid_mask = np.isfinite(anomaly_scores)
    anomaly_scores = anomaly_scores[valid_mask]
    ground_truth = ground_truth[valid_mask]

    # Compute metrics
    auroc = roc_auc_score(ground_truth, anomaly_scores)
    aupr = average_precision_score(ground_truth, anomaly_scores)

    print(f"AUROC: {auroc:.4f} (0.5 = random, 1.0 = perfect)")
    print(f"AUPR:  {aupr:.4f} (primary metric for imbalanced data)")

    # Compute precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(ground_truth, anomaly_scores)

    # Compute ROC curve
    fpr, tpr, roc_thresholds = roc_curve(ground_truth, anomaly_scores)

    # Calculate FPR95 (False Positive Rate at 95% True Positive Rate)
    target_tpr = 0.95
    idx_tpr95 = np.argmin(np.abs(tpr - target_tpr))
    fpr95 = fpr[idx_tpr95]
    actual_tpr = tpr[idx_tpr95]

    print(f"FPR95: {fpr95:.4f} ({fpr95*100:.2f}%)")
    print(f"  → False Positive Rate when TPR = {actual_tpr:.4f}")
    print(f"  → To detect {actual_tpr*100:.1f}% of anomalies,")
    print(f"     {fpr95*100:.1f}% of normal pixels are false alarms")

    # Find optimal threshold (max F1 score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last point
    optimal_threshold = pr_thresholds[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    print(f"\nOptimal operating point (max F1):")
    print(f"  Threshold: {optimal_threshold:.4f}")
    print(f"  Precision: {optimal_precision:.4f}")
    print(f"  Recall:    {optimal_recall:.4f}")
    print(f"  F1 Score:  {optimal_f1:.4f}")

    # Baseline comparison (authors' results from StreetHazards paper)
    baseline_fpr95 = 0.265
    baseline_auroc = 0.893
    baseline_aupr = 0.106

    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON (Authors' Results)")
    print(f"{'='*60}")
    print(f"{'Metric':<10} {'Your Model':>12} {'Baseline':>12} {'Difference':>12}")
    print(f"{'-'*60}")
    print(f"{'FPR95':<10} {fpr95*100:>11.2f}% {baseline_fpr95*100:>11.2f}% {(fpr95-baseline_fpr95)*100:>+11.2f}%")
    print(f"{'AUROC':<10} {auroc*100:>11.2f}% {baseline_auroc*100:>11.2f}% {(auroc-baseline_auroc)*100:>+11.2f}%")
    print(f"{'AUPR':<10} {aupr*100:>11.2f}% {baseline_aupr*100:>11.2f}% {(aupr-baseline_aupr)*100:>+11.2f}%")

    return {
        'auroc': auroc,
        'aupr': aupr,
        'fpr95': fpr95,
        'optimal_f1': optimal_f1,
        'optimal_threshold': optimal_threshold,
        'baseline_fpr95': baseline_fpr95,
        'baseline_auroc': baseline_auroc,
        'baseline_aupr': baseline_aupr
    }

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(MODEL_PATH, DEVICE)

    # Load test dataset
    print("\nLoading test dataset...")
    val_test_transform, val_test_mask_transform = get_transforms(IMAGE_SIZE, is_training=False)

    test_dataset = StreetHazardsDataset(
        root_dir=TEST_ROOT,
        split='test',
        transform=val_test_transform,
        mask_transform=val_test_mask_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    print(f"Loaded {len(test_dataset)} test samples")

    # Run anomaly detection with default temperature
    TEMPERATURE = 1.0
    scores_energy, gt_energy = detect_anomalies_energy_score(
        model, test_loader, DEVICE, temperature=TEMPERATURE
    )
    results_energy = evaluate_anomaly_detection(scores_energy, gt_energy, "Energy Score")

    # Save results summary
    summary_path = OUTPUT_DIR / 'energy_score_results.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ENERGY SCORE ANOMALY DETECTION RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test set: StreetHazards (1500 images)\n")
        f.write(f"Anomaly class: {ANOMALY_CLASS_IDX}\n")
        f.write(f"Temperature: {TEMPERATURE}\n")
        f.write(f"Max pixels for evaluation: {MAX_PIXELS:,} (random subsampling)\n")
        f.write(f"Random seed: {RANDOM_SEED} (for reproducibility)\n\n")

        f.write("METHOD DESCRIPTION\n")
        f.write("-"*80 + "\n")
        f.write("Energy Score [Liu et al., NeurIPS 2020]:\n")
        f.write("  E(x) = -T * log(sum_c exp(z_c / T))\n")
        f.write("       = -T * LogSumExp(z / T)\n\n")
        f.write("Key differences from Max Logits:\n")
        f.write("  - Max Logits: Uses only the maximum logit value\n")
        f.write("  - Energy Score: Uses ALL logits via LogSumExp\n")
        f.write("  - Energy has theoretical grounding in energy-based models\n")
        f.write("  - Better calibrated than softmax-based methods\n")
        f.write("  - More robust to domain shift (PEBAL: only 0.4% drop)\n\n")

        f.write("RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"AUROC: {results_energy['auroc']:.4f} ({results_energy['auroc']*100:.2f}%)\n")
        f.write(f"AUPR:  {results_energy['aupr']:.4f} ({results_energy['aupr']*100:.2f}%)\n")
        f.write(f"FPR95: {results_energy['fpr95']:.4f} ({results_energy['fpr95']*100:.2f}%)\n")
        f.write(f"F1:    {results_energy['optimal_f1']:.4f} ({results_energy['optimal_f1']*100:.2f}%)\n")
        f.write(f"Optimal Threshold: {results_energy['optimal_threshold']:.4f}\n\n")

        f.write("BASELINE COMPARISON (Authors' Results)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<10} {'Your Model':>15} {'Baseline':>15} {'Difference':>15}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'FPR95':<10} {results_energy['fpr95']*100:>14.2f}% "
                f"{results_energy['baseline_fpr95']*100:>14.2f}% "
                f"{(results_energy['fpr95']-results_energy['baseline_fpr95'])*100:>+14.2f}%\n")
        f.write(f"{'AUROC':<10} {results_energy['auroc']*100:>14.2f}% "
                f"{results_energy['baseline_auroc']*100:>14.2f}% "
                f"{(results_energy['auroc']-results_energy['baseline_auroc'])*100:>+14.2f}%\n")
        f.write(f"{'AUPR':<10} {results_energy['aupr']*100:>14.2f}% "
                f"{results_energy['baseline_aupr']*100:>14.2f}% "
                f"{(results_energy['aupr']-results_energy['baseline_aupr'])*100:>+14.2f}%\n\n")

        f.write("METRIC EXPLANATIONS\n")
        f.write("-"*80 + "\n")
        f.write("AUROC (Area Under ROC Curve):\n")
        f.write("  Measures the model's ability to rank anomaly scores correctly.\n")
        f.write("  Range: 0.5 (random) to 1.0 (perfect)\n")
        f.write("  Interpretation: Overall ranking quality across all thresholds.\n\n")

        f.write("AUPR (Area Under Precision-Recall Curve):\n")
        f.write("  Primary metric for imbalanced anomaly detection.\n")
        f.write("  Better than AUROC for datasets with rare anomalies (<2%).\n")
        f.write("  Interpretation: Trade-off between precision and recall.\n\n")

        f.write("FPR95 (False Positive Rate at 95% True Positive Rate):\n")
        f.write("  Answers: 'To detect 95% of anomalies, what % of normal pixels\n")
        f.write("  will be incorrectly flagged as anomalies?'\n")
        f.write("  Lower is better (fewer false alarms at high recall).\n")
        f.write("  Important for safety-critical applications (autonomous driving).\n")
        f.write(f"  Your result: {results_energy['fpr95']*100:.1f}% of normal pixels are false alarms\n")
        f.write(f"               to achieve 95% anomaly detection.\n\n")

        f.write("F1 Score:\n")
        f.write("  Harmonic mean of precision and recall at optimal threshold.\n")
        f.write("  Balances false positives and false negatives.\n")
        f.write("  Interpretation: Overall detection quality at best operating point.\n\n")

        f.write("="*80 + "\n")
        f.write("LITERATURE REFERENCES\n")
        f.write("="*80 + "\n")
        f.write("[1] Liu et al., 'Energy-based Out-of-distribution Detection',\n")
        f.write("    NeurIPS 2020. https://arxiv.org/abs/2010.03759\n\n")
        f.write("[2] Tian et al., 'Pixel-Wise Energy-Biased Abstention Learning',\n")
        f.write("    ECCV 2022. (PEBAL method demonstrating domain shift robustness)\n\n")
        f.write("="*80 + "\n")

    print(f"\n✅ Results summary saved: {summary_path}")

    print(f"\n{'='*60}")
    print("✅ ENERGY SCORE COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR}/")
