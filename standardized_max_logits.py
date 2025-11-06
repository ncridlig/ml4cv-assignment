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
    """Compute AUROC, AUPR, F1, and optimal threshold."""
    print(f"\n{'='*60}\nEVALUATION: STANDARDIZED MAX LOGITS\n{'='*60}")
    valid = np.isfinite(scores)
    scores, gt = scores[valid], gt[valid]

    auroc = roc_auc_score(gt, scores)
    aupr = average_precision_score(gt, scores)
    precision, recall, thresholds = precision_recall_curve(gt, scores)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1[:-1])
    best_thr = thresholds[best_idx]

    print(f"AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")
    print(f"Optimal F1: {f1[best_idx]:.4f} @ threshold={best_thr:.4f}")
    return {
        "auroc": auroc,
        "aupr": aupr,
        "f1": f1[best_idx],
        "threshold": best_thr,
        "precision": precision,
        "recall": recall
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
        f.write("="*60 + "\n")
        f.write("STANDARDIZED MAX LOGITS (SML) RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"AUROC: {results['auroc']:.4f}\n")
        f.write(f"AUPR:  {results['aupr']:.4f}\n")
        f.write(f"F1:    {results['f1']:.4f}\n")
        f.write(f"Threshold: {results['threshold']:.4f}\n")
    print(f"\n✅ Results summary saved: {summary_path}")
