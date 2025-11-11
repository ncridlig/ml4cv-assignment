"""
Ablation Studies for Simple Max Logits Anomaly Detection

Studies:
1. Threshold Sensitivity Analysis
2. Subsampling Ratio Impact
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score
)
from torch.utils.data import DataLoader

from dataloader import StreetHazardsDataset, get_transforms
from utils.model_utils import load_model
from config import (
    DEVICE, MODEL_PATH, NUM_CLASSES, ANOMALY_CLASS_IDX,
    IMAGE_SIZE, TEST_ROOT, RANDOM_SEED,
    OUTPUT_DIR_ANOMALY as OUTPUT_DIR
)

print("="*80)
print("ABLATION STUDIES: SIMPLE MAX LOGITS")
print("="*80)


# ============================================================================
# Collect test data once
# ============================================================================
@torch.no_grad()
def collect_test_data(model, test_loader, device):
    """Collect all test predictions once."""
    print("\nCollecting test data...")

    model.eval()
    all_scores = []
    all_gt = []

    for images, masks, _ in tqdm(test_loader, desc="Collecting"):
        images = images.to(device)
        masks = masks.numpy()

        output = model(images)['out']
        max_logits, _ = output.max(dim=1)
        max_logits = max_logits.cpu().numpy()

        scores = (-max_logits).astype(np.float16)
        gt = (masks == ANOMALY_CLASS_IDX).astype(np.uint8)

        all_scores.append(scores.flatten())
        all_gt.append(gt.flatten())

    all_scores = np.concatenate(all_scores)
    all_gt = np.concatenate(all_gt)

    print(f"Total pixels: {len(all_gt):,}")
    print(f"Anomaly pixels: {all_gt.sum():,} ({100*all_gt.mean():.2f}%)")

    return all_scores, all_gt


# ============================================================================
# ABLATION 1: Threshold Sensitivity Analysis
# ============================================================================
def ablation_threshold_sensitivity(scores, gt):
    """Study how performance varies with different thresholds."""
    print("\n" + "="*80)
    print("ABLATION 1: THRESHOLD SENSITIVITY ANALYSIS")
    print("="*80)

    # Remove invalid values
    valid = np.isfinite(scores)
    scores = scores[valid]
    gt = gt[valid]

    # Get PR curve
    precision, recall, thresholds = precision_recall_curve(gt, scores)

    # Compute F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    f1_scores = f1_scores[:-1]  # Remove last element (matches thresholds length)

    # Find optimal threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]

    print(f"\nOptimal Threshold Analysis:")
    print(f"  Best Threshold: {best_threshold:.4f}")
    print(f"  Best F1:        {best_f1:.4f}")
    print(f"  Precision:      {best_precision:.4f}")
    print(f"  Recall:         {best_recall:.4f}")

    # Test threshold variations
    print(f"\nThreshold Variation Study:")
    print(f"{'Threshold':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Δ from best'}")
    print("-"*65)

    test_thresholds = [
        best_threshold - 1.0,
        best_threshold - 0.5,
        best_threshold - 0.25,
        best_threshold,
        best_threshold + 0.25,
        best_threshold + 0.5,
        best_threshold + 1.0
    ]

    threshold_results = []
    for thresh in test_thresholds:
        predictions = (scores >= thresh).astype(int)
        prec = np.sum((predictions == 1) & (gt == 1)) / (np.sum(predictions == 1) + 1e-8)
        rec = np.sum((predictions == 1) & (gt == 1)) / (np.sum(gt == 1) + 1e-8)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        delta = f1 - best_f1

        threshold_results.append({
            'threshold': thresh,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

        marker = " ← OPTIMAL" if abs(thresh - best_threshold) < 0.01 else ""
        print(f"{thresh:<15.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {delta:+.4f}{marker}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: F1 vs Threshold
    ax = axes[0]
    ax.plot(thresholds, f1_scores, 'b-', linewidth=2)
    ax.axvline(best_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Optimal: {best_threshold:.4f}')
    ax.axhline(best_f1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Precision/Recall vs Threshold
    ax = axes[1]
    ax.plot(thresholds, precision[:-1], 'b-', linewidth=2, label='Precision')
    ax.plot(thresholds, recall[:-1], 'r-', linewidth=2, label='Recall')
    ax.axvline(best_threshold, color='green', linestyle='--', linewidth=2,
               label=f'Optimal: {best_threshold:.4f}')
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Precision/Recall vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Threshold variation impact
    ax = axes[2]
    thresh_vals = [r['threshold'] for r in threshold_results]
    f1_vals = [r['f1'] for r in threshold_results]
    colors = ['red' if abs(t - best_threshold) < 0.01 else 'blue' for t in thresh_vals]
    ax.bar(range(len(thresh_vals)), f1_vals, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(best_f1, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Threshold Variation', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score at Different Thresholds', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(thresh_vals)))
    ax.set_xticklabels([f'{t:.2f}' for t in thresh_vals], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'ablation_threshold_sensitivity.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()

    return threshold_results


# ============================================================================
# ABLATION 2: Subsampling Ratio Impact
# ============================================================================
def ablation_subsampling_ratio(scores, gt):
    """Study how subsampling ratio affects metric stability."""
    print("\n" + "="*80)
    print("ABLATION 2: SUBSAMPLING RATIO IMPACT")
    print("="*80)

    # Remove invalid values
    valid = np.isfinite(scores)
    scores = scores[valid]
    gt = gt[valid]

    total_pixels = len(scores)
    print(f"\nTotal pixels available: {total_pixels:,}")

    # Test different subsampling ratios
    ratios = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    n_trials = 5  # Multiple trials for stability estimation

    results = []

    print(f"\n{'Ratio':<10} {'Pixels':<12} {'AUROC':<12} {'AUPR':<12} {'F1':<12} {'Std (AUPR)'}")
    print("-"*75)

    for ratio in ratios:
        n_pixels = int(total_pixels * ratio)
        n_pixels = min(n_pixels, total_pixels)

        aurocs = []
        auprs = []
        f1s = []

        for trial in range(n_trials):
            np.random.seed(RANDOM_SEED + trial)
            if n_pixels < total_pixels:
                idx = np.random.choice(total_pixels, n_pixels, replace=False)
                sample_scores = scores[idx]
                sample_gt = gt[idx]
            else:
                sample_scores = scores
                sample_gt = gt

            auroc = roc_auc_score(sample_gt, sample_scores)
            aupr = average_precision_score(sample_gt, sample_scores)

            # Compute F1 at optimal threshold
            prec, rec, thresh = precision_recall_curve(sample_gt, sample_scores)
            f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
            best_f1 = np.max(f1[:-1])

            aurocs.append(auroc)
            auprs.append(aupr)
            f1s.append(best_f1)

        mean_auroc = np.mean(aurocs)
        mean_aupr = np.mean(auprs)
        mean_f1 = np.mean(f1s)
        std_aupr = np.std(auprs)

        results.append({
            'ratio': ratio,
            'n_pixels': n_pixels,
            'auroc': mean_auroc,
            'aupr': mean_aupr,
            'f1': mean_f1,
            'std_aupr': std_aupr
        })

        print(f"{ratio:<10.3f} {n_pixels:<12,} {mean_auroc:<12.4f} {mean_aupr:<12.4f} "
              f"{mean_f1:<12.4f} ±{std_aupr:.4f}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ratios_plot = [r['ratio'] for r in results]
    aurocs_plot = [r['auroc'] for r in results]
    auprs_plot = [r['aupr'] for r in results]
    f1s_plot = [r['f1'] for r in results]
    stds_plot = [r['std_aupr'] for r in results]

    # Plot 1: AUROC vs Ratio
    ax = axes[0]
    ax.plot(ratios_plot, aurocs_plot, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Subsampling Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title('AUROC vs Subsampling Ratio', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(aurocs_plot[-1], color='gray', linestyle='--', alpha=0.5, label='Full data')
    ax.legend()

    # Plot 2: AUPR vs Ratio (with error bars)
    ax = axes[1]
    ax.errorbar(ratios_plot, auprs_plot, yerr=stds_plot, fmt='ro-',
                linewidth=2, markersize=8, capsize=5, capthick=2)
    ax.set_xlabel('Subsampling Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUPR', fontsize=12, fontweight='bold')
    ax.set_title('AUPR vs Subsampling Ratio (with std)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(auprs_plot[-1], color='gray', linestyle='--', alpha=0.5, label='Full data')
    ax.legend()

    # Plot 3: Metric stability (coefficient of variation)
    ax = axes[2]
    cv = [s / a * 100 for s, a in zip(stds_plot, auprs_plot)]  # CV in percentage
    ax.plot(ratios_plot, cv, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Subsampling Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Metric Stability vs Subsampling Ratio', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'ablation_subsampling_ratio.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()

    return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model = load_model(MODEL_PATH, DEVICE)

    # Load test dataset
    print("Loading test dataset...")
    val_t, val_mask_t = get_transforms(IMAGE_SIZE, is_training=False)
    test_dataset = StreetHazardsDataset(TEST_ROOT, "test", val_t, val_mask_t)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4, shuffle=False)
    print(f"Test: {len(test_dataset)} samples")

    # Collect data once
    scores, gt = collect_test_data(model, test_loader, DEVICE)

    # Run ablation studies
    threshold_results = ablation_threshold_sensitivity(scores, gt)
    subsampling_results = ablation_subsampling_ratio(scores, gt)

    # Save summary
    summary_path = OUTPUT_DIR / 'ablation_studies_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ABLATION STUDIES SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write("STUDY 1: THRESHOLD SENSITIVITY\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Threshold':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        for r in threshold_results:
            f.write(f"{r['threshold']:<15.4f} {r['precision']:<12.4f} "
                   f"{r['recall']:<12.4f} {r['f1']:<12.4f}\n")

        f.write("\n" + "="*80 + "\n\n")

        f.write("STUDY 2: SUBSAMPLING RATIO IMPACT\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Ratio':<10} {'Pixels':<12} {'AUROC':<12} {'AUPR':<12} {'F1':<12}\n")
        for r in subsampling_results:
            f.write(f"{r['ratio']:<10.3f} {r['n_pixels']:<12,} {r['auroc']:<12.4f} "
                   f"{r['aupr']:<12.4f} {r['f1']:<12.4f}\n")

    print(f"\n✅ Summary saved: {summary_path}")

    print("\n" + "="*80)
    print("✅ ABLATION STUDIES COMPLETE")
    print("="*80)
