"""
Standardized Max Logits (SML) Anomaly Detection
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from dataloader import StreetHazardsDataset, get_transforms, mask_to_rgb, CLASS_COLORS
from utils.model_utils import load_model
from config import (
    DEVICE,
    MODEL_PATH,
    NUM_CLASSES,
    ANOMALY_CLASS_IDX,
    OUTPUT_DIR_ANOMALY as OUTPUT_DIR,
    IMAGE_SIZE,
    TRAIN_ROOT,
    TEST_ROOT,
    MAX_PIXELS_EVALUATION,
    RANDOM_SEED
)

# -----------------------------
# LOG HEADER
# -----------------------------
print("=" * 60)
print("PHASE 4: ANOMALY DETECTION - STANDARDIZED MAX LOGITS (SML)")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Anomaly class index: {ANOMALY_CLASS_IDX}")
print(f"Max pixels for evaluation: {MAX_PIXELS_EVALUATION:,}")
print(f"Memory optimizations: float16 precision + pixel subsampling")

# -----------------------------
# STEP 1 — COMPUTE CLASS STATISTICS
# -----------------------------
@torch.no_grad()
def compute_class_statistics(model, dataloader, device, num_classes=13):
    """Compute per-class mean and std of max logits for Standardized Max Logits (SML)."""
    print(f"\nComputing per-class statistics for SML...")

    class_count = {c: 0 for c in range(num_classes)}
    class_mean = {c: 0.0 for c in range(num_classes)}
    class_m2 = {c: 0.0 for c in range(num_classes)}

    model.eval()
    for images, _, _ in tqdm(dataloader, desc="Computing statistics"):
        images = images.to(device)
        output = model(images)['out']
        max_logits, pred_classes = output.max(dim=1)

        max_logits = max_logits.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()

        for c in range(num_classes):
            mask = (pred_classes == c)
            n_pixels = mask.sum()
            if n_pixels == 0:
                continue
            values = max_logits[mask].flatten()

            n_old = class_count[c]
            n_new = n_old + n_pixels
            delta = values - class_mean[c]
            class_mean[c] += np.sum(delta) / n_new
            delta2 = values - class_mean[c]
            class_m2[c] += np.sum(delta * delta2)
            class_count[c] = n_new

    # Finalize
    class_means, class_stds = {}, {}
    print("\nPer-class statistics:")
    print(f"{'Class':<8} {'Count':<12} {'Mean':<12} {'Std':<12}")
    print("-" * 50)
    for c in range(num_classes):
        if class_count[c] > 1:
            class_means[c] = class_mean[c]
            class_stds[c] = np.sqrt(class_m2[c] / class_count[c])
        else:
            class_means[c] = class_mean[c]
            class_stds[c] = 1.0
        print(f"{c:<8} {class_count[c]:<12} {class_means[c]:<12.4f} {class_stds[c]:<12.4f}")
    return class_means, class_stds

# -----------------------------
# STEP 2 — STANDARDIZED MAX LOGITS
# -----------------------------
@torch.no_grad()
def detect_anomalies_sml(model, dataloader, device, class_means, class_stds):
    """Compute anomaly scores using Standardized Max Logits (SML)."""
    print(f"\n{'='*60}\nMETHOD: STANDARDIZED MAX LOGITS (SML)\n{'='*60}")

    model.eval()
    all_scores, all_gt = [], []

    for images, masks, _ in tqdm(dataloader, desc="SML Detection"):
        images = images.to(device)
        masks = masks.numpy()

        output = model(images)['out']
        max_logits, pred_classes = output.max(dim=1)
        max_logits = max_logits.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()

        sml = np.zeros_like(max_logits)
        for c in range(NUM_CLASSES):
            mask = (pred_classes == c)
            if mask.any():
                sml[mask] = (max_logits[mask] - class_means[c]) / (class_stds[c] + 1e-8)

        anomaly_scores = (-sml).astype(np.float16)
        ground_truth = (masks == ANOMALY_CLASS_IDX).astype(np.uint8)

        all_scores.append(anomaly_scores.flatten())
        all_gt.append(ground_truth.flatten())

    all_scores = np.concatenate(all_scores)
    all_gt = np.concatenate(all_gt)

    total = len(all_gt)
    print(f"Total pixels: {total:,}, Anomalies: {all_gt.sum():,} ({100*all_gt.mean():.2f}%)")

    if total > MAX_PIXELS_EVALUATION:
        print(f"Subsampling {MAX_PIXELS_EVALUATION:,} pixels (ratio={MAX_PIXELS_EVALUATION/total:.2%})")
        np.random.seed(RANDOM_SEED)
        idx = np.random.choice(total, size=MAX_PIXELS_EVALUATION, replace=False)
        all_scores = all_scores[idx]
        all_gt = all_gt[idx]

    return all_scores, all_gt

# -----------------------------
# STEP 3 — EVALUATION
# -----------------------------
def evaluate_anomaly_detection(scores, gt):
    """Compute AUROC, AUPR, FPR95, F1, and optimal threshold."""
    print(f"\n{'='*60}\nEVALUATION: STANDARDIZED MAX LOGITS\n{'='*60}")
    valid = np.isfinite(scores)
    scores, gt = scores[valid], gt[valid]

    auroc = roc_auc_score(gt, scores)
    aupr = average_precision_score(gt, scores)
    precision, recall, pr_thresholds = precision_recall_curve(gt, scores)

    # Compute ROC curve for FPR95
    fpr, tpr, roc_thresholds = roc_curve(gt, scores)

    # Calculate FPR95 (False Positive Rate at 95% True Positive Rate)
    target_tpr = 0.95
    idx_tpr95 = np.argmin(np.abs(tpr - target_tpr))
    fpr95 = fpr[idx_tpr95]
    actual_tpr = tpr[idx_tpr95]

    # Find optimal threshold (max F1 score)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1[:-1])
    best_thr = pr_thresholds[best_idx]

    print(f"AUROC: {auroc:.4f} ({auroc*100:.2f}%)")
    print(f"AUPR:  {aupr:.4f} ({aupr*100:.2f}%)")
    print(f"FPR95: {fpr95:.4f} ({fpr95*100:.2f}%)")
    print(f"Optimal F1: {f1[best_idx]:.4f} @ threshold={best_thr:.4f}")

    print(f"\nFPR95 Interpretation:")
    print(f"  → False Positive Rate when TPR = {actual_tpr:.4f}")
    print(f"  → To detect {actual_tpr*100:.1f}% of anomalies,")
    print(f"     {fpr95*100:.1f}% of normal pixels are false alarms")

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
        "auroc": auroc,
        "aupr": aupr,
        "fpr95": fpr95,
        "f1": f1[best_idx],
        "threshold": best_thr,
        "precision": precision,
        "recall": recall,
        "baseline_fpr95": baseline_fpr95,
        "baseline_auroc": baseline_auroc,
        "baseline_aupr": baseline_aupr
    }

# -----------------------------
# STEP 4 — VISUALIZATION
# -----------------------------
def visualize_samples(model, dataset, class_means, class_stds, device, num_samples=5, save_dir=None):
    """Visualize anomaly maps using SML."""
    print(f"\nVisualizing {num_samples} samples with anomalies...")

    if save_dir:
        sample_dir = save_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    idxs = [i for i in range(len(dataset)) if ANOMALY_CLASS_IDX in dataset.get_raw_item(i)[1]][:num_samples]

    for i, idx in enumerate(idxs):
        image_tensor, mask_tensor, _ = dataset[idx]
        raw_image, raw_mask, _ = dataset.get_raw_item(idx)

        with torch.no_grad():
            out = model(image_tensor.unsqueeze(0).to(device))['out']
            max_logits, pred_classes = out.max(dim=1)
            max_logits = max_logits.squeeze(0).cpu().numpy()
            pred_classes = pred_classes.squeeze(0).cpu().numpy()

            sml = np.zeros_like(max_logits)
            for c in range(NUM_CLASSES):
                mask = (pred_classes == c)
                if mask.any():
                    sml[mask] = (max_logits[mask] - class_means[c]) / (class_stds[c] + 1e-8)
            anomaly_map = -sml

        fig, axes = plt.subplots(1, 3, figsize=(14, 6))
        axes[0].imshow(raw_image)
        axes[0].set_title("Input")
        axes[1].imshow(mask_to_rgb(raw_mask, CLASS_COLORS))
        axes[1].set_title("Ground Truth")
        im = axes[2].imshow(anomaly_map, cmap="hot")
        axes[2].set_title("SML Anomaly Map")
        plt.colorbar(im, ax=axes[2])
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        if save_dir:
            path = sample_dir / f"sml_{idx:04d}.png"
            plt.savefig(path, dpi=150)
            print(f"  Saved: {path}")
        plt.close()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(MODEL_PATH, DEVICE)
    val_t, val_mask_t = get_transforms(IMAGE_SIZE, is_training=False)
    val_dataset = StreetHazardsDataset(TRAIN_ROOT, "validation", val_t, val_mask_t)
    test_dataset = StreetHazardsDataset(TEST_ROOT, "test", val_t, val_mask_t)

    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)

    class_means, class_stds = compute_class_statistics(model, val_loader, DEVICE, NUM_CLASSES)
    scores, gt = detect_anomalies_sml(model, test_loader, DEVICE, class_means, class_stds)
    results = evaluate_anomaly_detection(scores, gt)
    visualize_samples(model, test_dataset, class_means, class_stds, DEVICE, num_samples=10, save_dir=OUTPUT_DIR)

    summary_path = OUTPUT_DIR / "sml_results.txt"
    with open(summary_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("STANDARDIZED MAX LOGITS (SML) ANOMALY DETECTION RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test set: StreetHazards (1500 images)\n")
        f.write(f"Anomaly class: {ANOMALY_CLASS_IDX}\n")
        f.write(f"Max pixels for evaluation: {MAX_PIXELS_EVALUATION:,} (random subsampling)\n")
        f.write(f"Random seed: {RANDOM_SEED} (for reproducibility)\n\n")

        f.write("METHOD DESCRIPTION\n")
        f.write("-"*80 + "\n")
        f.write("SML normalizes max logits by class-specific statistics (mean and std).\n")
        f.write("This accounts for different confidence levels across classes.\n")
        f.write("Formula: SML(x) = (max_logit(x) - mean_c) / std_c\n")
        f.write("where c is the predicted class.\n\n")

        f.write("RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"AUROC: {results['auroc']:.4f} ({results['auroc']*100:.2f}%)\n")
        f.write(f"AUPR:  {results['aupr']:.4f} ({results['aupr']*100:.2f}%)\n")
        f.write(f"FPR95: {results['fpr95']:.4f} ({results['fpr95']*100:.2f}%)\n")
        f.write(f"F1:    {results['f1']:.4f} ({results['f1']*100:.2f}%)\n")
        f.write(f"Optimal Threshold: {results['threshold']:.4f}\n\n")

        f.write("BASELINE COMPARISON (Authors' Results)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<10} {'Your Model':>15} {'Baseline':>15} {'Difference':>15}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'FPR95':<10} {results['fpr95']*100:>14.2f}% "
                f"{results['baseline_fpr95']*100:>14.2f}% "
                f"{(results['fpr95']-results['baseline_fpr95'])*100:>+14.2f}%\n")
        f.write(f"{'AUROC':<10} {results['auroc']*100:>14.2f}% "
                f"{results['baseline_auroc']*100:>14.2f}% "
                f"{(results['auroc']-results['baseline_auroc'])*100:>+14.2f}%\n")
        f.write(f"{'AUPR':<10} {results['aupr']*100:>14.2f}% "
                f"{results['baseline_aupr']*100:>14.2f}% "
                f"{(results['aupr']-results['baseline_aupr'])*100:>+14.2f}%\n\n")

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
        f.write(f"  Your result: {results['fpr95']*100:.1f}% of normal pixels are false alarms\n")
        f.write(f"               to achieve 95% anomaly detection.\n\n")

        f.write("F1 Score:\n")
        f.write("  Harmonic mean of precision and recall at optimal threshold.\n")
        f.write("  Balances false positives and false negatives.\n")
        f.write("  Interpretation: Overall detection quality at best operating point.\n\n")

        f.write("="*80 + "\n")

    print(f"\n✅ Results summary saved: {summary_path}")
