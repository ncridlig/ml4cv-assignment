"""
Create comprehensive comparison visualizations for anomaly detection methods.

Generates:
1. PR curves comparing Simple Max Logits vs SML
2. ROC curves comparing both methods
3. Side-by-side comparison with metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
from torch.utils.data import DataLoader

from dataloader import StreetHazardsDataset, get_transforms
from utils.model_utils import load_model
from config import (
    DEVICE, MODEL_PATH, NUM_CLASSES, ANOMALY_CLASS_IDX,
    IMAGE_SIZE, TRAIN_ROOT, TEST_ROOT, RANDOM_SEED,
    OUTPUT_DIR_ANOMALY as OUTPUT_DIR, MAX_PIXELS_EVALUATION
)

print("="*80)
print("CREATING COMPARISON VISUALIZATIONS")
print("="*80)

# ============================================================================
# STEP 1: Compute validation statistics for SML
# ============================================================================
@torch.no_grad()
def compute_validation_statistics(model, val_loader, device):
    """Compute per-class statistics for SML."""
    print("\nComputing validation statistics...")

    model.eval()
    class_count = {c: 0 for c in range(NUM_CLASSES)}
    class_mean = {c: 0.0 for c in range(NUM_CLASSES)}
    class_m2 = {c: 0.0 for c in range(NUM_CLASSES)}

    for images, _, _ in tqdm(val_loader, desc="Computing stats"):
        images = images.to(device)
        output = model(images)['out']
        max_logits, pred_classes = output.max(dim=1)
        max_logits = max_logits.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()

        for c in range(NUM_CLASSES):
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

    class_means, class_stds = {}, {}
    for c in range(NUM_CLASSES):
        if class_count[c] > 1:
            class_means[c] = class_mean[c]
            class_stds[c] = np.sqrt(class_m2[c] / class_count[c])
        else:
            class_means[c] = class_mean[c] if class_count[c] == 1 else 0.0
            class_stds[c] = 1.0

    return class_means, class_stds


# ============================================================================
# STEP 2: Compute anomaly scores for both methods
# ============================================================================
@torch.no_grad()
def compute_all_scores(model, test_loader, class_means, class_stds, device):
    """Compute anomaly scores using both methods."""
    print("\nComputing anomaly scores for both methods...")

    model.eval()

    all_simple_scores = []
    all_sml_scores = []
    all_gt = []

    for images, masks, _ in tqdm(test_loader, desc="Computing scores"):
        images = images.to(device)
        masks = masks.numpy()

        output = model(images)['out']
        max_logits, pred_classes = output.max(dim=1)
        max_logits = max_logits.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()

        # Simple Max Logits
        simple_scores = (-max_logits).astype(np.float16)

        # Standardized Max Logits
        sml = np.zeros_like(max_logits)
        for c in range(NUM_CLASSES):
            mask = (pred_classes == c)
            if mask.any():
                sml[mask] = (max_logits[mask] - class_means[c]) / (class_stds[c] + 1e-8)
        sml_scores = (-sml).astype(np.float16)

        # Ground truth
        gt = (masks == ANOMALY_CLASS_IDX).astype(np.uint8)

        all_simple_scores.append(simple_scores.flatten())
        all_sml_scores.append(sml_scores.flatten())
        all_gt.append(gt.flatten())

    # Concatenate
    all_simple_scores = np.concatenate(all_simple_scores)
    all_sml_scores = np.concatenate(all_sml_scores)
    all_gt = np.concatenate(all_gt)

    total_pixels = len(all_gt)
    print(f"Total pixels: {total_pixels:,}")
    print(f"Anomaly pixels: {all_gt.sum():,} ({100*all_gt.mean():.2f}%)")

    # Subsample if needed
    if total_pixels > MAX_PIXELS_EVALUATION:
        print(f"Subsampling to {MAX_PIXELS_EVALUATION:,} pixels...")
        np.random.seed(RANDOM_SEED)
        idx = np.random.choice(total_pixels, MAX_PIXELS_EVALUATION, replace=False)
        all_simple_scores = all_simple_scores[idx]
        all_sml_scores = all_sml_scores[idx]
        all_gt = all_gt[idx]
        print(f"After subsampling - Anomaly pixels: {all_gt.sum():,} ({100*all_gt.mean():.2f}%)")

    return all_simple_scores, all_sml_scores, all_gt


# ============================================================================
# STEP 3: Create comparison visualization
# ============================================================================
def create_comparison_plot(simple_scores, sml_scores, gt):
    """Create comprehensive comparison visualization."""
    print("\nCreating comparison visualization...")

    # Remove invalid values
    valid_simple = np.isfinite(simple_scores)
    valid_sml = np.isfinite(sml_scores)
    valid = valid_simple & valid_sml

    simple_scores = simple_scores[valid]
    sml_scores = sml_scores[valid]
    gt = gt[valid]

    # Compute metrics for both methods
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)

    # Simple Max Logits
    auroc_simple = roc_auc_score(gt, simple_scores)
    aupr_simple = average_precision_score(gt, simple_scores)
    prec_simple, rec_simple, thresh_pr_simple = precision_recall_curve(gt, simple_scores)
    fpr_simple, tpr_simple, thresh_roc_simple = roc_curve(gt, simple_scores)
    f1_simple = 2 * (prec_simple * rec_simple) / (prec_simple + rec_simple + 1e-8)
    best_f1_simple = np.max(f1_simple[:-1])

    print("\nSimple Max Logits:")
    print(f"  AUROC: {auroc_simple:.4f}")
    print(f"  AUPR:  {aupr_simple:.4f}")
    print(f"  Best F1: {best_f1_simple:.4f}")

    # Standardized Max Logits
    auroc_sml = roc_auc_score(gt, sml_scores)
    aupr_sml = average_precision_score(gt, sml_scores)
    prec_sml, rec_sml, thresh_pr_sml = precision_recall_curve(gt, sml_scores)
    fpr_sml, tpr_sml, thresh_roc_sml = roc_curve(gt, sml_scores)
    f1_sml = 2 * (prec_sml * rec_sml) / (prec_sml + rec_sml + 1e-8)
    best_f1_sml = np.max(f1_sml[:-1])

    print("\nStandardized Max Logits (SML):")
    print(f"  AUROC: {auroc_sml:.4f}")
    print(f"  AUPR:  {aupr_sml:.4f}")
    print(f"  Best F1: {best_f1_sml:.4f}")

    print("\nDifference (SML - Simple):")
    print(f"  ΔAUROC: {auroc_sml - auroc_simple:+.4f} ({100*(auroc_sml-auroc_simple)/auroc_simple:+.1f}%)")
    print(f"  ΔAUPR:  {aupr_sml - aupr_simple:+.4f} ({100*(aupr_sml-aupr_simple)/aupr_simple:+.1f}%)")
    print(f"  ΔF1:    {best_f1_sml - best_f1_simple:+.4f} ({100*(best_f1_sml-best_f1_simple)/best_f1_simple:+.1f}%)")

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ========================================================================
    # Plot 1: Precision-Recall Curves
    # ========================================================================
    ax = axes[0, 0]
    ax.plot(rec_simple, prec_simple, 'b-', linewidth=2.5,
            label=f'Simple Max Logits (AUPR={aupr_simple:.4f})')
    ax.plot(rec_sml, prec_sml, 'r--', linewidth=2.5,
            label=f'Standardized (SML) (AUPR={aupr_sml:.4f})')

    # Baseline (random classifier)
    baseline_aupr = gt.mean()
    ax.axhline(y=baseline_aupr, color='gray', linestyle=':', linewidth=2,
               label=f'Random Baseline (AUPR={baseline_aupr:.4f})')

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # ========================================================================
    # Plot 2: ROC Curves
    # ========================================================================
    ax = axes[0, 1]
    ax.plot(fpr_simple, tpr_simple, 'b-', linewidth=2.5,
            label=f'Simple Max Logits (AUROC={auroc_simple:.4f})')
    ax.plot(fpr_sml, tpr_sml, 'r--', linewidth=2.5,
            label=f'Standardized (SML) (AUROC={auroc_sml:.4f})')
    ax.plot([0, 1], [0, 1], 'gray', linestyle=':', linewidth=2,
            label='Random Baseline (AUROC=0.5000)')

    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # ========================================================================
    # Plot 3: Performance Metrics Comparison
    # ========================================================================
    ax = axes[1, 0]
    metrics = ['AUROC', 'AUPR', 'F1']
    simple_vals = [auroc_simple, aupr_simple, best_f1_simple]
    sml_vals = [auroc_sml, aupr_sml, best_f1_sml]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, simple_vals, width, label='Simple Max Logits',
                   color='#2E86DE', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, sml_vals, width, label='Standardized (SML)',
                   color='#EE5A6F', edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ========================================================================
    # Plot 4: Score Distribution Comparison
    # ========================================================================
    ax = axes[1, 1]

    # Separate scores for anomalies and normal pixels
    anomaly_mask = gt == 1
    normal_mask = gt == 0

    # Subsample for visualization (too many points otherwise)
    n_vis = min(10000, len(gt))
    np.random.seed(42)
    vis_idx = np.random.choice(len(gt), n_vis, replace=False)

    # Plot score distributions
    ax.scatter(simple_scores[vis_idx][normal_mask[vis_idx]],
              sml_scores[vis_idx][normal_mask[vis_idx]],
              alpha=0.3, s=1, c='blue', label='Normal pixels')
    ax.scatter(simple_scores[vis_idx][anomaly_mask[vis_idx]],
              sml_scores[vis_idx][anomaly_mask[vis_idx]],
              alpha=0.6, s=10, c='red', label='Anomaly pixels', marker='x')

    ax.set_xlabel('Simple Max Logits Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('SML Score', fontsize=12, fontweight='bold')
    ax.set_title('Score Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add diagonal line
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'gray', linestyle='--',
            linewidth=1, alpha=0.5, label='y=x')

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'method_comparison_comprehensive.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()

    return {
        'simple': {'auroc': auroc_simple, 'aupr': aupr_simple, 'f1': best_f1_simple},
        'sml': {'auroc': auroc_sml, 'aupr': aupr_sml, 'f1': best_f1_sml}
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model = load_model(MODEL_PATH, DEVICE)

    # Load datasets
    print("Loading datasets...")
    val_t, val_mask_t = get_transforms(IMAGE_SIZE, is_training=False)
    val_dataset = StreetHazardsDataset(TRAIN_ROOT, "validation", val_t, val_mask_t)
    test_dataset = StreetHazardsDataset(TEST_ROOT, "test", val_t, val_mask_t)

    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4, shuffle=False)

    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    # Compute statistics
    class_means, class_stds = compute_validation_statistics(model, val_loader, DEVICE)

    # Compute scores
    simple_scores, sml_scores, gt = compute_all_scores(
        model, test_loader, class_means, class_stds, DEVICE
    )

    # Create visualization
    results = create_comparison_plot(simple_scores, sml_scores, gt)

    print("\n" + "="*80)
    print("✅ COMPARISON VISUALIZATION COMPLETE")
    print("="*80)
