import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from torchvision.models.segmentation import deeplabv3_resnet50
from dataloader import StreetHazardsDataset, get_transforms

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_deeplabv3_streethazards.pth'  # Model trained today (without aux fix)
NUM_CLASSES = 13
ANOMALY_CLASS_IDX = 13
OUTPUT_DIR = Path('assets/anomaly_detection')

print("="*60)
print("PHASE 4: ANOMALY DETECTION WITH MAX LOGITS")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Anomaly class index: {ANOMALY_CLASS_IDX}")

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model(model_path, device):
    """Load trained DeepLabV3+ model."""
    print(f"\nLoading model from {model_path}...")
    model = deeplabv3_resnet50(weights=None)
    model.classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)

    # Load checkpoint (strict=False to ignore aux_classifier)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    return model

# -----------------------------
# COMPUTE CLASS STATISTICS (FOR SML)
# -----------------------------
@torch.no_grad()
def compute_class_statistics(model, dataloader, device, num_classes=13):
    """
    Compute per-class mean and std of max logits on a dataset.
    This is needed for Standardized Max Logits (SML).

    Args:
        model: Trained segmentation model
        dataloader: DataLoader (typically validation set)
        device: Device to run on
        num_classes: Number of classes

    Returns:
        class_means: Dict mapping class_id -> mean of max logits
        class_stds: Dict mapping class_id -> std of max logits
    """
    print(f"\nComputing per-class statistics for SML...")

    # Collect max logits for each class
    class_logits = {c: [] for c in range(num_classes)}

    model.eval()
    for images, masks, _ in tqdm(dataloader, desc="Computing statistics"):
        images = images.to(device)

        # Get predictions
        output = model(images)['out']  # (B, 13, H, W)

        # Get max logits and predicted classes
        max_logits, pred_classes = output.max(dim=1)  # Both: (B, H, W)

        # Convert to numpy
        max_logits = max_logits.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()

        # Collect logits per class
        for c in range(num_classes):
            mask = (pred_classes == c)
            if mask.sum() > 0:
                class_logits[c].extend(max_logits[mask].flatten().tolist())

    # Compute mean and std for each class
    class_means = {}
    class_stds = {}

    print("\nPer-class statistics:")
    print(f"{'Class':<15} {'Count':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 50)

    for c in range(num_classes):
        if len(class_logits[c]) > 0:
            class_means[c] = np.mean(class_logits[c])
            class_stds[c] = np.std(class_logits[c])
            print(f"{c:<15} {len(class_logits[c]):<10} {class_means[c]:<10.4f} {class_stds[c]:<10.4f}")
        else:
            # Fallback if class never predicted (shouldn't happen)
            class_means[c] = 0.0
            class_stds[c] = 1.0
            print(f"{c:<15} {0:<10} {'N/A':<10} {'N/A':<10}")

    return class_means, class_stds

# -----------------------------
# ANOMALY DETECTION METHODS
# -----------------------------
@torch.no_grad()
def detect_anomalies_simple_max_logits(model, dataloader, device):
    """
    Method 1: Simple Max Logits
    anomaly_score[i] = -max(logits[i])

    Lower max logit = higher anomaly score
    """
    print(f"\n{'='*60}")
    print("METHOD 1: SIMPLE MAX LOGITS")
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

        # Anomaly score = negative max logit
        anomaly_scores = -max_logits  # (B, H, W)

        # Ground truth: 1 if anomaly (class 13), 0 otherwise
        ground_truth = (masks == ANOMALY_CLASS_IDX).astype(int)

        # Flatten and collect
        all_anomaly_scores.append(anomaly_scores.flatten())
        all_ground_truth.append(ground_truth.flatten())

    # Concatenate all batches
    all_anomaly_scores = np.concatenate(all_anomaly_scores)
    all_ground_truth = np.concatenate(all_ground_truth)

    print(f"Total pixels: {len(all_ground_truth):,}")
    print(f"Anomaly pixels: {all_ground_truth.sum():,} ({100*all_ground_truth.mean():.2f}%)")

    return all_anomaly_scores, all_ground_truth

@torch.no_grad()
def detect_anomalies_standardized_max_logits(model, dataloader, device, class_means, class_stds):
    """
    Method 2: Standardized Max Logits (SML)

    For each pixel i with predicted class c:
    1. max_logit[i] = max(logits[i])
    2. SML[i] = (max_logit[i] - μ_c) / σ_c
    3. anomaly_score[i] = -SML[i]

    Lower SML = higher anomaly score
    """
    print(f"\n{'='*60}")
    print("METHOD 2: STANDARDIZED MAX LOGITS (SML)")
    print(f"{'='*60}")

    model.eval()

    all_anomaly_scores = []
    all_ground_truth = []

    for images, masks, _ in tqdm(dataloader, desc="Standardized Max Logits"):
        images = images.to(device)
        masks = masks.numpy()  # (B, H, W)

        # Get predictions
        output = model(images)['out']  # (B, 13, H, W)

        # Get max logits and predicted classes
        max_logits, pred_classes = output.max(dim=1)  # Both: (B, H, W)
        max_logits = max_logits.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()

        # Standardize per predicted class
        sml = np.zeros_like(max_logits)

        for c in range(NUM_CLASSES):
            mask = (pred_classes == c)
            if mask.sum() > 0:
                # Standardize: (x - μ) / σ
                sml[mask] = (max_logits[mask] - class_means[c]) / (class_stds[c] + 1e-8)

        # Anomaly score = negative SML
        anomaly_scores = -sml  # (B, H, W)

        # Ground truth: 1 if anomaly (class 13), 0 otherwise
        ground_truth = (masks == ANOMALY_CLASS_IDX).astype(int)

        # Flatten and collect
        all_anomaly_scores.append(anomaly_scores.flatten())
        all_ground_truth.append(ground_truth.flatten())

    # Concatenate all batches
    all_anomaly_scores = np.concatenate(all_anomaly_scores)
    all_ground_truth = np.concatenate(all_ground_truth)

    print(f"Total pixels: {len(all_ground_truth):,}")
    print(f"Anomaly pixels: {all_ground_truth.sum():,} ({100*all_ground_truth.mean():.2f}%)")

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
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1
    }

# -----------------------------
# VISUALIZATION
# -----------------------------
def plot_comparison(results_simple, results_sml, save_dir):
    """Create comparison plots for both methods."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Precision-Recall Curves
    ax = axes[0]
    ax.plot(results_simple['recall'], results_simple['precision'],
            label=f"Simple Max Logits (AUPR={results_simple['aupr']:.4f})",
            linewidth=2, color='blue')
    ax.plot(results_sml['recall'], results_sml['precision'],
            label=f"Standardized Max Logits (AUPR={results_sml['aupr']:.4f})",
            linewidth=2, color='red')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve (Primary Metric)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Plot 2: ROC Curves
    ax = axes[1]
    ax.plot(results_simple['fpr'], results_simple['tpr'],
            label=f"Simple Max Logits (AUROC={results_simple['auroc']:.4f})",
            linewidth=2, color='blue')
    ax.plot(results_sml['fpr'], results_sml['tpr'],
            label=f"Standardized Max Logits (AUROC={results_sml['auroc']:.4f})",
            linewidth=2, color='red')
    ax.plot([0, 1], [0, 1], 'k--', label='Random (0.5)', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    save_path = save_dir / 'method_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved: {save_path}")
    plt.close()

def visualize_sample_predictions(model, dataset, class_means, class_stds, device, num_samples=5, save_dir=None):
    """Visualize anomaly detection on sample images."""
    from dataloader import mask_to_rgb, CLASS_COLORS

    print(f"\n{'='*60}")
    print(f"VISUALIZING SAMPLE PREDICTIONS")
    print(f"{'='*60}")

    # Create save directory
    if save_dir:
        sample_dir = save_dir / 'samples'
        sample_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Select samples with anomalies
    indices_with_anomalies = []
    for idx in range(len(dataset)):
        _, mask, _ = dataset.get_raw_item(idx)
        if ANOMALY_CLASS_IDX in mask:
            indices_with_anomalies.append(idx)
        if len(indices_with_anomalies) >= num_samples:
            break

    print(f"Found {len(indices_with_anomalies)} samples with anomalies")

    for i, idx in enumerate(indices_with_anomalies):
        # Get sample
        image_tensor, mask_tensor, img_path = dataset[idx]
        raw_image, raw_mask, _ = dataset.get_raw_item(idx)

        # Run inference
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(device)
            output = model(image_batch)['out']

            # Get max logits and predictions
            max_logits, pred_classes = output.max(dim=1)
            max_logits = max_logits.squeeze(0).cpu().numpy()
            pred_classes = pred_classes.squeeze(0).cpu().numpy()

            # Simple Max Logits
            simple_score = -max_logits

            # Standardized Max Logits
            sml = np.zeros_like(max_logits)
            for c in range(NUM_CLASSES):
                mask = (pred_classes == c)
                if mask.sum() > 0:
                    sml[mask] = (max_logits[mask] - class_means[c]) / (class_stds[c] + 1e-8)
            sml_score = -sml

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Row 1: Input, Ground Truth, Prediction
        axes[0, 0].imshow(raw_image)
        axes[0, 0].set_title('Input Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        gt_rgb = mask_to_rgb(raw_mask, CLASS_COLORS)
        axes[0, 1].imshow(gt_rgb)
        has_anomaly = (raw_mask == ANOMALY_CLASS_IDX).sum()
        axes[0, 1].set_title(f'Ground Truth (Anomaly: {has_anomaly} pixels)',
                            fontsize=12, fontweight='bold', color='red')
        axes[0, 1].axis('off')

        pred_rgb = mask_to_rgb(pred_classes, CLASS_COLORS)
        axes[0, 2].imshow(pred_rgb)
        axes[0, 2].set_title('Predicted Segmentation', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # Row 2: Simple Max Logits, Standardized Max Logits, Ground Truth Anomaly
        im1 = axes[1, 0].imshow(simple_score, cmap='hot', vmin=simple_score.min(), vmax=simple_score.max())
        axes[1, 0].set_title('Simple Max Logits\n(red = high anomaly score)', fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im2 = axes[1, 1].imshow(sml_score, cmap='hot', vmin=sml_score.min(), vmax=sml_score.max())
        axes[1, 1].set_title('Standardized Max Logits\n(red = high anomaly score)', fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

        anomaly_gt = (raw_mask == ANOMALY_CLASS_IDX).astype(float)
        im3 = axes[1, 2].imshow(anomaly_gt, cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title('Ground Truth Anomaly Mask\n(white = anomaly)', fontsize=11, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_dir:
            save_path = sample_dir / f'sample_{idx:04d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [{i+1}/{len(indices_with_anomalies)}] Saved: {save_path}")

        plt.close()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(MODEL_PATH, DEVICE)

    # Load datasets
    print("\nLoading datasets...")
    val_transform, val_mask_transform = get_transforms(512, is_training=False)

    val_dataset = StreetHazardsDataset(
        root_dir='streethazards_train/train',
        split='validation',
        transform=val_transform,
        mask_transform=val_mask_transform
    )

    test_dataset = StreetHazardsDataset(
        root_dir='streethazards_test/test',
        split='test',
        transform=val_transform,
        mask_transform=val_mask_transform
    )

    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Step 1: Compute class statistics on validation set (for SML)
    class_means, class_stds = compute_class_statistics(model, val_loader, DEVICE, NUM_CLASSES)

    # Step 2: Run anomaly detection on test set - Method 1 (Simple Max Logits)
    scores_simple, gt_simple = detect_anomalies_simple_max_logits(model, test_loader, DEVICE)
    results_simple = evaluate_anomaly_detection(scores_simple, gt_simple, "Simple Max Logits")

    # Step 3: Run anomaly detection on test set - Method 2 (Standardized Max Logits)
    scores_sml, gt_sml = detect_anomalies_standardized_max_logits(
        model, test_loader, DEVICE, class_means, class_stds
    )
    results_sml = evaluate_anomaly_detection(scores_sml, gt_sml, "Standardized Max Logits (SML)")

    # Step 4: Compare methods
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'AUROC':<12} {'AUPR':<12} {'F1 Score':<12}")
    print("-" * 66)
    print(f"{'Simple Max Logits':<30} {results_simple['auroc']:<12.4f} {results_simple['aupr']:<12.4f} {results_simple['optimal_f1']:<12.4f}")
    print(f"{'Standardized Max Logits':<30} {results_sml['auroc']:<12.4f} {results_sml['aupr']:<12.4f} {results_sml['optimal_f1']:<12.4f}")
    print("-" * 66)

    improvement_aupr = ((results_sml['aupr'] - results_simple['aupr']) / results_simple['aupr']) * 100
    improvement_f1 = ((results_sml['optimal_f1'] - results_simple['optimal_f1']) / results_simple['optimal_f1']) * 100

    print(f"\nSML Improvement over Simple:")
    print(f"  AUPR: {improvement_aupr:+.2f}%")
    print(f"  F1 Score: {improvement_f1:+.2f}%")

    # Step 5: Create comparison plots
    plot_comparison(results_simple, results_sml, OUTPUT_DIR)

    # Step 6: Visualize sample predictions
    visualize_sample_predictions(model, test_dataset, class_means, class_stds, DEVICE,
                                 num_samples=10, save_dir=OUTPUT_DIR)

    # Step 7: Save results summary
    summary_path = OUTPUT_DIR / 'results_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ANOMALY DETECTION RESULTS - PHASE 4\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test set: StreetHazards (1500 images)\n")
        f.write(f"Anomaly class: {ANOMALY_CLASS_IDX}\n\n")
        f.write("METHOD 1: SIMPLE MAX LOGITS\n")
        f.write(f"  AUROC: {results_simple['auroc']:.4f}\n")
        f.write(f"  AUPR:  {results_simple['aupr']:.4f}\n")
        f.write(f"  F1:    {results_simple['optimal_f1']:.4f}\n\n")
        f.write("METHOD 2: STANDARDIZED MAX LOGITS (SML)\n")
        f.write(f"  AUROC: {results_sml['auroc']:.4f}\n")
        f.write(f"  AUPR:  {results_sml['aupr']:.4f}\n")
        f.write(f"  F1:    {results_sml['optimal_f1']:.4f}\n\n")
        f.write(f"SML Improvement:\n")
        f.write(f"  AUPR: {improvement_aupr:+.2f}%\n")
        f.write(f"  F1:   {improvement_f1:+.2f}%\n")

    print(f"\n✅ Results summary saved: {summary_path}")

    print(f"\n{'='*60}")
    print("✅ PHASE 4 COMPLETE!")
    print(f"{'='*60}")
    print(f"All results saved to: {OUTPUT_DIR}/")
    print(f"  - method_comparison.png: PR and ROC curves")
    print(f"  - samples/: Visualizations of anomaly detection on 10 test images")
    print(f"  - results_summary.txt: Text summary of metrics")
