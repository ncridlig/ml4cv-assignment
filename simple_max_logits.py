"""
Simple Max Logits Anomaly Detection
Extracts only the Simple Max Logits method from the full script.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from dataloader import StreetHazardsDataset, get_transforms
from utils.model_utils import load_model

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models/best_deeplabv3_streethazards_11_52_04-11-25_mIoU_3757.pth'
NUM_CLASSES = 13
ANOMALY_CLASS_IDX = 13
OUTPUT_DIR = Path('assets/anomaly_detection')
MAX_PIXELS = 1_000_000  # Subsample to this many pixels for evaluation (reduces memory usage)

print("="*60)
print("SIMPLE MAX LOGITS ANOMALY DETECTION")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Anomaly class index: {ANOMALY_CLASS_IDX}")
print(f"Max pixels for evaluation: {MAX_PIXELS:,} (random subsampling)")

# -----------------------------
# ANOMALY DETECTION METHOD
# -----------------------------
@torch.no_grad()
def detect_anomalies_simple_max_logits(model, dataloader, device):
    """
    Method: Simple Max Logits
    anomaly_score[i] = -max(logits[i])

    Lower max logit = higher anomaly score
    """
    print(f"\n{'='*60}")
    print("METHOD: SIMPLE MAX LOGITS")
    print(f"{'='*60}")

    model.eval()

    all_anomaly_scores = []
    all_ground_truth = []

    for images, masks, _ in tqdm(dataloader, desc="Simple Max Logits"):
        images = images.to(device)
        masks = masks.numpy()  # (B, H, W)

        # Get predictions
        output = model(images)['out']  # (B, 13, H, W)

        # Compute max logits
        max_logits, _ = output.max(dim=1)  # (B, H, W)
        max_logits = max_logits.cpu().numpy()

        # Anomaly score = negative max logit (use float16 for memory efficiency)
        anomaly_scores = (-max_logits).astype(np.float16)  # (B, H, W)

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
        np.random.seed(42)  # For reproducibility
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
    - AUROC: Area Under ROC Curve
    - AUPR: Area Under Precision-Recall Curve (primary metric)
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
    print(f"AUPR:  {aupr:.4f} (primary metric)")

    # Compute precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(ground_truth, anomaly_scores)

    # Compute ROC curve
    fpr, tpr, roc_thresholds = roc_curve(ground_truth, anomaly_scores)

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

    return {
        'auroc': auroc,
        'aupr': aupr,
        'optimal_f1': optimal_f1,
        'optimal_threshold': optimal_threshold
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
    val_test_transform, val_test_mask_transform = get_transforms(512, is_training=False)

    test_dataset = StreetHazardsDataset(
        root_dir='streethazards_test/test',
        split='test',
        transform=val_test_transform,
        mask_transform=val_test_mask_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    print(f"Loaded {len(test_dataset)} test samples")

    # Run anomaly detection
    scores_simple, gt_simple = detect_anomalies_simple_max_logits(model, test_loader, DEVICE)
    results_simple = evaluate_anomaly_detection(scores_simple, gt_simple, "Simple Max Logits")

    # Save results summary
    summary_path = OUTPUT_DIR / 'simple_max_logits_results.txt'
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SIMPLE MAX LOGITS ANOMALY DETECTION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test set: StreetHazards (1500 images)\n")
        f.write(f"Anomaly class: {ANOMALY_CLASS_IDX}\n")
        f.write(f"Max pixels for evaluation: {MAX_PIXELS:,} (random subsampling)\n")
        f.write(f"Random seed: 42 (for reproducibility)\n\n")
        f.write("RESULTS:\n")
        f.write(f"  AUROC: {results_simple['auroc']:.4f}\n")
        f.write(f"  AUPR:  {results_simple['aupr']:.4f}\n")
        f.write(f"  F1:    {results_simple['optimal_f1']:.4f}\n")
        f.write(f"  Optimal Threshold: {results_simple['optimal_threshold']:.4f}\n")

    print(f"\n✅ Results summary saved: {summary_path}")

    print(f"\n{'='*60}")
    print("✅ SIMPLE MAX LOGITS COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR}/")
