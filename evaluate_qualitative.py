import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataloader import (
    StreetHazardsDataset,
    get_transforms,
    CLASS_NAMES,
    CLASS_COLORS,
    mask_to_rgb
)
from utils.model_utils import load_model
from config import (
    DEVICE,
    MODEL_PATH,
    NUM_QUALITATIVE_SAMPLES as NUM_SAMPLES,
    OUTPUT_DIR_QUALITATIVE as OUTPUT_DIR,
    ANOMALY_THRESHOLD,
    ANOMALY_CLASS_IDX,
    IMAGE_SIZE,
    TRAIN_ROOT,
    TEST_ROOT
)

# -----------------------------
# INFERENCE
# -----------------------------
@torch.no_grad()
def predict(model, image, device, anomaly_threshold=-1.4834):
    """
    Run inference on a single image with anomaly detection.

    Args:
        model: Trained segmentation model
        image: Image tensor (C, H, W)
        device: Device to run on
        anomaly_threshold: Threshold for anomaly detection

    Returns:
        pred_mask: Predicted segmentation mask (H, W)
        logits: Raw logits (13, H, W)
        anomaly_scores: Anomaly scores per pixel (H, W)
        anomaly_pred: Binary anomaly prediction (H, W)
    """
    model.eval()
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    output = model(image)['out']  # Shape: (1, 13, H, W)
    logits = output.squeeze(0)  # Remove batch dimension: (13, H, W)
    pred_mask = torch.argmax(logits, dim=0)  # (H, W)

    # Anomaly detection: Simple Max Logits
    max_logits, _ = logits.max(dim=0)  # (H, W)
    anomaly_scores = -max_logits.cpu().numpy()  # Higher = more anomalous
    anomaly_pred = (anomaly_scores > anomaly_threshold).astype(int)  # Binary

    return pred_mask.cpu().numpy(), logits.cpu().numpy(), anomaly_scores, anomaly_pred

# -----------------------------
# METRICS
# -----------------------------
def compute_iou_per_class(pred_mask, gt_mask, num_classes=13, ignore_index=13):
    """
    Compute IoU for each class.

    Args:
        pred_mask: Predicted mask (H, W)
        gt_mask: Ground truth mask (H, W)
        num_classes: Number of classes
        ignore_index: Class to ignore

    Returns:
        iou_per_class: Dict mapping class_id -> IoU
        mean_iou: Mean IoU across all classes
    """
    iou_per_class = {}
    ious = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_inds = (pred_mask == cls)
        target_inds = (gt_mask == cls)

        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()

        if union == 0:
            continue  # Class not present

        iou = intersection / union
        iou_per_class[cls] = iou
        ious.append(iou)

    mean_iou = np.mean(ious) if ious else 0.0
    return iou_per_class, mean_iou

# -----------------------------
# VISUALIZATION
# -----------------------------
def denormalize_image(img_tensor):
    """Denormalize image tensor for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).numpy()

    return (img * 255).astype(np.uint8)

def visualize_prediction(image_tensor, gt_mask, pred_mask, logits, anomaly_scores, anomaly_pred,
                        sample_idx, split_name, save_dir, anomaly_threshold=-1.4834):
    """
    Create visualization with image, ground truth, prediction, confidence, and anomaly detection.

    Args:
        image_tensor: Normalized image tensor (C, H, W)
        gt_mask: Ground truth mask (H, W)
        pred_mask: Predicted mask (H, W)
        logits: Raw logits (13, H, W)
        anomaly_scores: Anomaly scores per pixel (H, W)
        anomaly_pred: Binary anomaly prediction (H, W)
        sample_idx: Sample index
        split_name: Name of the split (val/test)
        save_dir: Directory to save the figure
        anomaly_threshold: Anomaly detection threshold
    """
    # Denormalize image
    image = denormalize_image(image_tensor)

    # Convert masks to RGB
    gt_rgb = mask_to_rgb(gt_mask)
    pred_rgb = mask_to_rgb(pred_mask)

    # Compute confidence (max probability)
    probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
    confidence = probs.max(axis=0)

    # Compute per-class IoU
    iou_per_class, mean_iou = compute_iou_per_class(pred_mask, gt_mask)

    # Get unique classes in ground truth and prediction
    gt_classes = np.unique(gt_mask)
    pred_classes = np.unique(pred_mask)

    # Check for anomalies
    has_anomaly = (13 in gt_classes)
    anomaly_detected = (13 in pred_classes)

    # Create figure with 3 rows, 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    # Row 1: Image, Ground Truth, Prediction
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_rgb)
    title = 'Ground Truth'
    if has_anomaly:
        title += ' (Contains Anomaly!)'
    axes[0, 1].set_title(title, fontsize=12, fontweight='bold', color='red' if has_anomaly else 'black')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(pred_rgb)
    axes[0, 2].set_title(f'Prediction (mIoU: {mean_iou:.3f})', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Overlay, Confidence Map, Error Map
    overlay = (image * 0.5 + pred_rgb * 0.5).astype(np.uint8)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    conf_plot = axes[1, 1].imshow(confidence, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title('Prediction Confidence', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(conf_plot, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Error map (red = wrong, green = correct)
    error_map = (pred_mask != gt_mask) & (gt_mask != 13)  # Ignore anomaly class
    error_vis = np.zeros((*error_map.shape, 3), dtype=np.uint8)
    error_vis[error_map] = [255, 0, 0]  # Red for errors
    error_vis[~error_map] = [0, 255, 0]  # Green for correct

    # Blend with image for better visualization
    error_overlay = (image * 0.6 + error_vis * 0.4).astype(np.uint8)
    axes[1, 2].imshow(error_overlay)
    accuracy = (~error_map).sum() / error_map.size
    axes[1, 2].set_title(f'Error Map (Accuracy: {accuracy:.3f})', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    # Row 3: ANOMALY DETECTION
    # Ground truth anomalies
    gt_anomaly = (gt_mask == ANOMALY_CLASS_IDX).astype(int)

    # Column 1: Anomaly Score Heatmap
    score_plot = axes[2, 0].imshow(anomaly_scores, cmap='RdYlGn_r', vmin=-8, vmax=0)
    axes[2, 0].set_title(f'Anomaly Scores (Threshold: {anomaly_threshold:.2f})',
                         fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    plt.colorbar(score_plot, ax=axes[2, 0], fraction=0.046, pad=0.04)

    # Column 2: Binary Anomaly Detection
    anomaly_vis = np.zeros((*anomaly_pred.shape, 3), dtype=np.uint8)
    anomaly_vis[anomaly_pred == 1] = [255, 0, 0]  # Red for detected anomalies
    anomaly_vis[anomaly_pred == 0] = [0, 255, 0]  # Green for normal
    axes[2, 1].imshow(anomaly_vis)
    detected_pixels = anomaly_pred.sum()
    total_pixels = anomaly_pred.size
    axes[2, 1].set_title(f'Detected Anomalies ({detected_pixels:,} / {total_pixels:,} px = {100*detected_pixels/total_pixels:.2f}%)',
                         fontsize=12, fontweight='bold')
    axes[2, 1].axis('off')

    # Column 3: Detection Performance (TP, FP, FN, TN)
    detection_vis = np.zeros((*anomaly_pred.shape, 3), dtype=np.uint8)
    tp = (anomaly_pred == 1) & (gt_anomaly == 1)  # True Positive: red
    fp = (anomaly_pred == 1) & (gt_anomaly == 0)  # False Positive: orange
    fn = (anomaly_pred == 0) & (gt_anomaly == 1)  # False Negative: blue
    tn = (anomaly_pred == 0) & (gt_anomaly == 0)  # True Negative: green

    detection_vis[tp] = [255, 0, 0]      # Red: correctly detected anomaly
    detection_vis[fp] = [255, 165, 0]    # Orange: false alarm
    detection_vis[fn] = [0, 0, 255]      # Blue: missed anomaly
    detection_vis[tn] = [0, 128, 0]      # Dark green: correctly normal

    # Blend with image for better context
    detection_overlay = (image * 0.6 + detection_vis * 0.4).astype(np.uint8)
    axes[2, 2].imshow(detection_overlay)

    # Compute metrics
    tp_count = tp.sum()
    fp_count = fp.sum()
    fn_count = fn.sum()
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    axes[2, 2].set_title(f'Detection Performance (P={precision:.2f}, R={recall:.2f}, F1={f1:.2f})',
                         fontsize=12, fontweight='bold')
    axes[2, 2].axis('off')

    # Add text summary
    summary_text = f"Sample {sample_idx} ({split_name})\n"
    summary_text += f"Mean IoU: {mean_iou:.4f}\n"
    summary_text += f"Pixel Accuracy: {accuracy:.4f}\n"
    summary_text += f"Avg Confidence: {confidence.mean():.3f}\n"

    # Add anomaly detection stats
    if has_anomaly:
        summary_text += f"\n⚠️ Anomaly Present in GT\n"
        anomaly_pixels_gt = (gt_mask == ANOMALY_CLASS_IDX).sum()
        summary_text += f"GT Anomaly: {anomaly_pixels_gt:,} ({100*anomaly_pixels_gt/total_pixels:.2f}%)\n"
        summary_text += f"Detected: {detected_pixels:,} ({100*detected_pixels/total_pixels:.2f}%)\n"
        summary_text += f"\nDetection Metrics:\n"
        summary_text += f"  Precision: {precision:.3f}\n"
        summary_text += f"  Recall: {recall:.3f}\n"
        summary_text += f"  F1 Score: {f1:.3f}\n"
        summary_text += f"  TP: {tp_count:,} | FP: {fp_count:,}\n"
        summary_text += f"  FN: {fn_count:,}\n"
    else:
        summary_text += f"\n✓ No Anomalies in GT\n"
        summary_text += f"False Alarms: {detected_pixels:,} ({100*detected_pixels/total_pixels:.2f}%)\n"

    # Top-3 classes by IoU
    if iou_per_class:
        summary_text += "\nTop-3 Classes by IoU:\n"
        sorted_classes = sorted(iou_per_class.items(), key=lambda x: x[1], reverse=True)[:3]
        for cls_id, iou in sorted_classes:
            summary_text += f"  {CLASS_NAMES[cls_id]}: {iou:.3f}\n"

    fig.text(0.02, 0.5, summary_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    save_path = save_dir / f'{split_name}_sample_{sample_idx:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

# -----------------------------
# MAIN EVALUATION
# -----------------------------
def evaluate_qualitative(model, dataset, split_name, num_samples, save_dir, device):
    """
    Evaluate model qualitatively on a dataset split.

    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        split_name: Name of the split (for naming)
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
        device: Device to run on
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {split_name} set ({num_samples} samples)")
    print(f"{'='*60}")

    # Create save directory
    split_dir = save_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Sample indices (evenly spaced + random)
    if num_samples <= len(dataset):
        # Take evenly spaced samples
        step = len(dataset) // num_samples
        indices = list(range(0, len(dataset), step))[:num_samples]
    else:
        indices = list(range(len(dataset)))

    # Compute metrics for all samples
    all_ious = []

    for i, idx in enumerate(indices, 1):
        # Get sample
        image, mask, img_path = dataset[idx]

        # Run inference with anomaly detection
        pred_mask, logits, anomaly_scores, anomaly_pred = predict(model, image, device, ANOMALY_THRESHOLD)

        # Compute IoU
        _, mean_iou = compute_iou_per_class(pred_mask, mask.numpy())
        all_ious.append(mean_iou)

        # Visualize
        print(f"  [{i}/{len(indices)}] Processing sample {idx}... mIoU: {mean_iou:.4f}")
        visualize_prediction(image, mask.numpy(), pred_mask, logits, anomaly_scores, anomaly_pred,
                           idx, split_name, split_dir, ANOMALY_THRESHOLD)

    # Print summary
    print(f"\n{split_name.upper()} Summary:")
    print(f"  Mean IoU across {len(indices)} samples: {np.mean(all_ious):.4f} ± {np.std(all_ious):.4f}")
    print(f"  Min IoU: {np.min(all_ious):.4f}")
    print(f"  Max IoU: {np.max(all_ious):.4f}")

def create_comparison_grid(save_dir, split_name, num_samples=5):
    """Create a grid comparing multiple samples side-by-side."""
    split_dir = save_dir / split_name

    # Find all saved images
    image_paths = sorted(list(split_dir.glob(f'{split_name}_sample_*.png')))[:num_samples]

    if not image_paths:
        print(f"No images found in {split_dir}")
        return

    # Load images
    images = [plt.imread(str(p)) for p in image_paths]

    # Create grid
    fig, axes = plt.subplots(len(images), 1, figsize=(18, 10 * len(images)))
    if len(images) == 1:
        axes = [axes]

    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()

    grid_path = save_dir / f'{split_name}_comparison_grid.png'
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison grid saved: {grid_path}")
    plt.close()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("="*60)
    print("QUALITATIVE EVALUATION OF DEEPLABV3+ MODEL")
    print("="*60)

    # Load model
    model = load_model(MODEL_PATH, DEVICE)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\nLoading datasets...")
    val_transform, val_mask_transform = get_transforms(IMAGE_SIZE, is_training=False)

    val_dataset = StreetHazardsDataset(
        root_dir=TRAIN_ROOT,
        split='validation',
        transform=val_transform,
        mask_transform=val_mask_transform
    )

    test_dataset = StreetHazardsDataset(
        root_dir=TEST_ROOT,
        split='test',
        transform=val_transform,
        mask_transform=val_mask_transform
    )

    # Evaluate on validation set
    evaluate_qualitative(model, val_dataset, 'validation', NUM_SAMPLES, OUTPUT_DIR, DEVICE)

    # Evaluate on test set (with anomalies!)
    evaluate_qualitative(model, test_dataset, 'test', NUM_SAMPLES, OUTPUT_DIR, DEVICE)

    # Create comparison grids
    print(f"\n{'='*60}")
    print("Creating comparison grids...")
    print(f"{'='*60}")
    create_comparison_grid(OUTPUT_DIR, 'validation', num_samples=5)
    create_comparison_grid(OUTPUT_DIR, 'test', num_samples=5)

    print(f"\n{'='*60}")
    print("✅ QUALITATIVE EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - validation/: {NUM_SAMPLES} validation samples")
    print(f"  - test/: {NUM_SAMPLES} test samples (with anomalies)")
    print(f"  - *_comparison_grid.png: Side-by-side comparisons")
    print("\nEach visualization shows:")
    print("  Row 1: Input | Ground Truth | Prediction")
    print("  Row 2: Overlay | Confidence Map | Error Map")
    print("  Row 3: Anomaly Scores | Detected Anomalies | Detection Performance")
    print(f"\nAnomaly Detection:")
    print(f"  Method: Simple Max Logits")
    print(f"  Threshold: {ANOMALY_THRESHOLD:.4f}")
    print(f"  Colors: Red=TP, Orange=FP, Blue=FN, Green=TN")
