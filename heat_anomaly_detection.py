"""
HEAT: Hybrid Energy-Adaptive Thresholding for Anomaly Detection

Combines three complementary anomaly scores:
1. Energy Score (logit-space) - Robust baseline
2. Mahalanobis Distance (feature-space) - Semantic outliers
3. Spatial Consistency (context) - Scene coherence

Plus test-time adaptive normalization to avoid domain shift failure.

Expected Performance: 10-12% AUPR (vs current 8.43% Max Logits)
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
import logging
import sys
from datetime import datetime

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
    TRAIN_ROOT,
    TEST_ROOT
)

# Setup logging to both file and console
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_path = OUTPUT_DIR / f'heat_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("HEAT: HYBRID ENERGY-ADAPTIVE THRESHOLDING")
logger.info("="*80)
logger.info(f"Device: {DEVICE}")
logger.info(f"Model: {MODEL_PATH}")
logger.info(f"Log file: {log_path}")

print("="*80)
print("HEAT: HYBRID ENERGY-ADAPTIVE THRESHOLDING")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Log file: {log_path}")

# ============================================================================
# COMPONENT 1: ENERGY SCORE
# ============================================================================
def compute_energy_score(logits, temperature=1.0):
    """
    Compute energy score for OOD detection.

    Energy = -T * log(sum(exp(z_c / T))) = -T * LogSumExp(z / T)

    Args:
        logits: (B, C, H, W) - model logits
        temperature: float - temperature parameter

    Returns:
        energy: (B, H, W) - energy scores (lower = more in-distribution)
    """
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return energy


# ============================================================================
# COMPONENT 2: FEATURE EXTRACTION & STATISTICS
# ============================================================================
class FeatureExtractor:
    """Extract intermediate features from model."""

    def __init__(self, model, layer_name='backbone.layer3'):
        self.features = None
        self.layer_name = layer_name

        # Register hook
        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.features = output

    def get_features(self):
        return self.features


@torch.no_grad()
def compute_class_statistics(model, train_loader, device, num_classes=13, layer_name='backbone.layer3'):
    """
    Compute mean and tied covariance for each class.

    Uses tied covariance (shared across classes) for memory efficiency.

    Returns:
        class_means: dict mapping class_id -> mean vector (D,)
        tied_cov: covariance matrix (D, D) shared across all classes
    """
    logger.info("="*80)
    logger.info("COMPUTING CLASS STATISTICS FOR MAHALANOBIS DISTANCE")
    logger.info("="*80)
    logger.info(f"Layer: {layer_name}")

    feature_extractor = FeatureExtractor(model, layer_name=layer_name)

    # Collect features per class
    class_features = {c: [] for c in range(num_classes)}

    model.eval()
    for images, labels, _ in tqdm(train_loader, desc="Extracting features"):
        try:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            _ = model(images)
            features = feature_extractor.get_features()  # (B, D, H_feat, W_feat)
        except Exception as e:
            logger.warning(f"Skipping batch due to error: {e}")
            continue

        B, D, H_feat, W_feat = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H_feat*W_feat, D)

        # Downsample labels to match feature spatial dimensions
        labels_downsampled = F.interpolate(
            labels.unsqueeze(1).float(),
            size=(H_feat, W_feat),
            mode='nearest'
        ).squeeze(1).long()
        labels_flat = labels_downsampled.reshape(-1)  # (B*H_feat*W_feat,)

        # Group by class (subsample to avoid memory issues)
        for c in range(num_classes):
            mask = (labels_flat == c)
            if mask.sum() > 0:
                class_feats = features[mask]
                # Subsample to avoid OOM (per class per batch)
                if class_feats.shape[0] > 10000:
                    idx = torch.randperm(class_feats.shape[0])[:10000]
                    class_feats = class_feats[idx]
                class_features[c].append(class_feats.cpu())

    # Compute class means
    logger.info("Computing class means...")
    class_means = {}
    for c in tqdm(range(num_classes), desc="Class means"):
        if len(class_features[c]) > 0:
            feats = torch.cat(class_features[c], dim=0)  # (N, D)
            class_means[c] = feats.mean(dim=0)  # (D,)
            logger.info(f"  Class {c}: {feats.shape[0]:,} samples")
        else:
            logger.warning(f"  Class {c}: No samples found!")

    # Compute tied covariance
    logger.info("Computing tied covariance matrix...")
    all_features_centered = []

    for c in tqdm(range(num_classes), desc="Centering features"):
        if c in class_means and len(class_features[c]) > 0:
            feats = torch.cat(class_features[c], dim=0)
            feats_centered = feats - class_means[c]
            all_features_centered.append(feats_centered)

    if len(all_features_centered) == 0:
        raise ValueError("No features extracted! Check that training data loaded correctly.")

    all_features_centered = torch.cat(all_features_centered, dim=0)
    N, D = all_features_centered.shape
    logger.info(f"Total centered features: {N:,} samples, {D} dimensions")

    # Compute covariance: Σ = (X^T X) / (N - 1)
    logger.info("Computing covariance matrix (this may take a moment)...")
    cov = (all_features_centered.T @ all_features_centered) / (N - 1)

    # Add regularization to avoid singularity
    cov = cov + 1e-4 * torch.eye(D)

    logger.info(f"Covariance matrix shape: {cov.shape}")
    logger.info(f"Covariance matrix memory: {cov.element_size() * cov.nelement() / 1e6:.2f} MB")

    return class_means, cov


# ============================================================================
# COMPONENT 3: MAHALANOBIS DISTANCE
# ============================================================================
def compute_mahalanobis_distance(features, class_means, cov_inv, device):
    """
    Compute minimum Mahalanobis distance to class prototypes.

    Args:
        features: (N, D) - feature vectors
        class_means: dict - class mean vectors
        cov_inv: (D, D) - inverse covariance matrix (tied)

    Returns:
        distances: (N,) - minimum Mahalanobis distance across classes
    """
    num_classes = len(class_means)
    N, D = features.shape

    min_distances = torch.full((N,), float('inf'), device=device)

    cov_inv = cov_inv.to(device)

    for c in range(num_classes):
        if c not in class_means:
            continue

        # Centered features
        diff = features - class_means[c].to(device)  # (N, D)

        # Mahalanobis distance: sqrt((x - μ)^T Σ^-1 (x - μ))
        mahal = torch.sum(diff @ cov_inv * diff, dim=1)  # (N,)
        mahal = torch.sqrt(mahal + 1e-8)

        # Keep minimum distance
        min_distances = torch.minimum(min_distances, mahal)

    return min_distances


# ============================================================================
# COMPONENT 4: SPATIAL CONSISTENCY
# ============================================================================
def compute_spatial_consistency(softmax_probs, kernel_size=3):
    """
    Compute spatial consistency score using KL divergence.

    Measures KL divergence between pixel prediction and neighborhood average.

    Args:
        softmax_probs: (B, C, H, W) - softmax probabilities
        kernel_size: int - neighborhood size

    Returns:
        consistency: (B, H, W) - consistency scores (higher = more consistent)
    """
    # Average pooling to get neighborhood distribution
    padding = kernel_size // 2
    neighbor_probs = F.avg_pool2d(
        softmax_probs,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    # KL divergence: D_KL(p || q) = sum(p * log(p / q))
    epsilon = 1e-8
    kl_div = (softmax_probs * torch.log(
        (softmax_probs + epsilon) / (neighbor_probs + epsilon)
    )).sum(dim=1)

    # Convert to consistency (negative KL)
    consistency = -kl_div

    return consistency


# ============================================================================
# COMPONENT 5: ADAPTIVE NORMALIZATION
# ============================================================================
class AdaptiveNormalizer:
    """Test-time adaptive normalization using EMA."""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.mean_ema = None
        self.std_ema = None

    def update(self, scores):
        """Update statistics with new batch."""
        batch_mean = scores.mean()
        batch_std = scores.std()

        if self.mean_ema is None:
            self.mean_ema = batch_mean
            self.std_ema = batch_std
        else:
            self.mean_ema = self.alpha * self.mean_ema + (1 - self.alpha) * batch_mean
            self.std_ema = self.alpha * self.std_ema + (1 - self.alpha) * batch_std

    def normalize(self, scores):
        """Normalize scores using EMA statistics."""
        if self.mean_ema is None:
            return scores

        normalized = (scores - self.mean_ema) / (self.std_ema + 1e-6)
        return normalized

    def reset(self):
        """Reset statistics."""
        self.mean_ema = None
        self.std_ema = None


# ============================================================================
# COMPONENT 6: SCORE COMBINATION WITH RELIABILITY WEIGHTING
# ============================================================================
def compute_reliability_weights(energy, mahalanobis, spatial_consistency, softmax_probs):
    """
    Compute reliability weights for each score component.

    Returns:
        weights: tuple of (w_energy, w_mahal, w_spatial) as tensors
    """
    # Energy reliability: inverse entropy
    entropy = -(softmax_probs * torch.log(softmax_probs + 1e-8)).sum(dim=1)
    w_energy = 1.0 / (entropy + 1e-6)

    # Mahalanobis reliability: constant weight
    w_mahal = torch.ones_like(w_energy)

    # Spatial reliability: inverse of local variance
    local_var = F.avg_pool2d(
        (softmax_probs - softmax_probs.mean(dim=1, keepdim=True)) ** 2,
        kernel_size=3, stride=1, padding=1
    ).sum(dim=1)
    w_spatial = 1.0 / (local_var + 1e-6)

    # Normalize weights
    total = w_energy + w_mahal + w_spatial
    w_energy = w_energy / total
    w_mahal = w_mahal / total
    w_spatial = w_spatial / total

    return w_energy, w_mahal, w_spatial


def combine_scores(energy, mahalanobis, spatial_consistency, softmax_probs):
    """
    Combine three scores with reliability weighting.

    Returns:
        combined_score: (B, H, W) - final anomaly score
    """
    # Compute weights
    w_e, w_m, w_s = compute_reliability_weights(
        energy, mahalanobis, spatial_consistency, softmax_probs
    )

    # Combine (higher = more anomalous)
    combined = (
        w_e * energy +           # Higher energy = more anomalous
        w_m * mahalanobis +      # Higher distance = more anomalous
        w_s * (-spatial_consistency)  # Lower consistency = more anomalous
    )

    return combined


# ============================================================================
# HEAT CLASS
# ============================================================================
class HEAT:
    """
    Hybrid Energy-Adaptive Thresholding for OOD detection.
    """

    def __init__(self, model, class_means, cov_inv, device,
                 temperature=1.0, alpha=0.9, kernel_size=3,
                 layer_name='backbone.layer3'):
        self.model = model
        self.class_means = class_means
        self.cov_inv = cov_inv
        self.device = device
        self.temperature = temperature
        self.kernel_size = kernel_size

        self.feature_extractor = FeatureExtractor(model, layer_name=layer_name)
        self.normalizer = AdaptiveNormalizer(alpha=alpha)

    @torch.no_grad()
    def forward(self, images):
        """
        Compute HEAT anomaly scores.

        Args:
            images: (B, 3, H, W)

        Returns:
            anomaly_scores: (B, H, W)
        """
        # Forward pass
        logits = self.model(images)['out']
        features = self.feature_extractor.get_features()

        # Softmax probabilities
        softmax_probs = F.softmax(logits, dim=1)

        # Component 1: Energy Score
        energy = compute_energy_score(logits, self.temperature)

        # Component 2: Mahalanobis Distance
        B, D, H_feat, W_feat = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, D)

        mahal_flat = compute_mahalanobis_distance(
            features_flat, self.class_means, self.cov_inv, self.device
        )
        mahal = mahal_flat.reshape(B, H_feat, W_feat)

        # Upsample to match logits size if needed
        if mahal.shape[1:] != energy.shape[1:]:
            mahal = F.interpolate(
                mahal.unsqueeze(1),
                size=energy.shape[1:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        # Component 3: Spatial Consistency
        spatial_cons = compute_spatial_consistency(softmax_probs, self.kernel_size)

        # Combine scores
        combined = combine_scores(energy, mahal, spatial_cons, softmax_probs)

        # Adaptive normalization
        self.normalizer.update(combined)
        normalized = self.normalizer.normalize(combined)

        return normalized

    def evaluate(self, test_loader):
        """
        Evaluate HEAT on test set.

        Returns:
            metrics: dict with AUPR, FPR95, AUROC
        """
        logger.info("="*80)
        logger.info("EVALUATING HEAT")
        logger.info("="*80)

        all_scores = []
        all_labels = []

        self.model.eval()
        self.normalizer.reset()  # Reset EMA statistics

        for images, labels, _ in tqdm(test_loader, desc="HEAT Evaluation"):
            images = images.to(self.device)

            scores = self.forward(images)

            all_scores.append(scores.cpu().numpy().astype(np.float16))
            all_labels.append(labels.numpy())

        # Concatenate all batches
        all_scores = np.concatenate([s.flatten() for s in all_scores])
        all_labels = np.concatenate([l.flatten() for l in all_labels])

        total_pixels = len(all_labels)
        logger.info(f"Total pixels: {total_pixels:,}")
        logger.info(f"Anomaly pixels: {(all_labels == ANOMALY_CLASS_IDX).sum():,} ({100*(all_labels == ANOMALY_CLASS_IDX).mean():.2f}%)")

        # Subsample if needed
        if total_pixels > MAX_PIXELS:
            logger.info(f"Subsampling to {MAX_PIXELS:,} pixels...")
            np.random.seed(RANDOM_SEED)
            idx = np.random.choice(total_pixels, MAX_PIXELS, replace=False)
            all_scores = all_scores[idx]
            all_labels = all_labels[idx]

        # Binary labels: 1 = anomaly, 0 = known
        binary_labels = (all_labels == ANOMALY_CLASS_IDX).astype(int)

        # Remove invalid values
        valid_mask = np.isfinite(all_scores)
        all_scores = all_scores[valid_mask]
        binary_labels = binary_labels[valid_mask]

        # Compute metrics
        auroc = roc_auc_score(binary_labels, all_scores)
        aupr = average_precision_score(binary_labels, all_scores)

        fpr, tpr, _ = roc_curve(binary_labels, all_scores)
        idx_tpr95 = np.argmin(np.abs(tpr - 0.95))
        fpr95 = fpr[idx_tpr95]

        logger.info("RESULTS:")
        logger.info(f"  AUROC: {auroc:.4f}")
        logger.info(f"  AUPR:  {aupr:.4f}")
        logger.info(f"  FPR95: {fpr95:.4f}")

        return {
            'auroc': auroc,
            'aupr': aupr,
            'fpr95': fpr95
        }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Paths for cached statistics
    stats_path = OUTPUT_DIR / 'feature_statistics.pkl'

    # Load model
    logger.info("Loading model...")
    model = load_model(MODEL_PATH, DEVICE)

    # Load datasets
    logger.info("Loading datasets...")
    train_t, train_mask_t = get_transforms(IMAGE_SIZE, is_training=False)  # No augmentation for statistics
    test_t, test_mask_t = get_transforms(IMAGE_SIZE, is_training=False)

    # Use validation set (1031 samples) instead of training (5125) to avoid OOM
    train_dataset = StreetHazardsDataset(TRAIN_ROOT, "validation", train_t, train_mask_t)
    test_dataset = StreetHazardsDataset(TEST_ROOT, "test", test_t, test_mask_t)

    # Use num_workers=0 to avoid multiprocessing issues with corrupted images
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=0, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=0, shuffle=False)

    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Test: {len(test_dataset)} samples")

    # Compute or load feature statistics
    if stats_path.exists():
        logger.info(f"Loading cached statistics from {stats_path}...")
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            class_means = stats['class_means']
            cov = stats['cov']
        logger.info("✅ Statistics loaded from cache")
    else:
        logger.info("Computing feature statistics (this will take several minutes)...")
        class_means, cov = compute_class_statistics(
            model, train_loader, DEVICE, num_classes=NUM_CLASSES
        )

        # Save statistics
        logger.info(f"Saving statistics to {stats_path}...")
        with open(stats_path, 'wb') as f:
            pickle.dump({'class_means': class_means, 'cov': cov}, f)
        logger.info("✅ Statistics saved")

    # Compute covariance inverse
    logger.info("Computing covariance inverse...")
    cov_inv = torch.inverse(cov)
    logger.info(f"Covariance inverse shape: {cov_inv.shape}")

    # Initialize HEAT
    logger.info("Initializing HEAT...")
    heat = HEAT(
        model=model,
        class_means=class_means,
        cov_inv=cov_inv,
        device=DEVICE,
        temperature=1.0,
        alpha=0.9,
        kernel_size=3
    )

    # Evaluate
    results = heat.evaluate(test_loader)

    # Save results
    summary_path = OUTPUT_DIR / 'heat_results.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HEAT: HYBRID ENERGY-ADAPTIVE THRESHOLDING RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Temperature: 1.0\n")
        f.write(f"EMA alpha: 0.9\n")
        f.write(f"Spatial kernel: 3x3\n")
        f.write(f"Feature layer: backbone.layer3\n\n")

        f.write("RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"AUROC: {results['auroc']:.4f} ({results['auroc']*100:.2f}%)\n")
        f.write(f"AUPR:  {results['aupr']:.4f} ({results['aupr']*100:.2f}%)\n")
        f.write(f"FPR95: {results['fpr95']:.4f} ({results['fpr95']*100:.2f}%)\n\n")

        f.write("BASELINE COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"Simple Max Logits:  AUPR = 8.43%, AUROC = 90.50%\n")
        f.write(f"HEAT:               AUPR = {results['aupr']*100:.2f}%, AUROC = {results['auroc']*100:.2f}%\n")
        f.write(f"Improvement:        AUPR = {(results['aupr']*100 - 8.43):+.2f}%, AUROC = {(results['auroc']*100 - 90.50):+.2f}%\n\n")

        f.write("="*80 + "\n")

    logger.info(f"✅ Results saved to {summary_path}")
    logger.info("="*80)
    logger.info("✅ HEAT EVALUATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Log file saved: {log_path}")

    print(f"\n✅ Results saved to {summary_path}")
    print(f"\n{'='*80}")
    print("✅ HEAT EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Log file: {log_path}")
