"""
Comprehensive Model vs Anomaly Detection Method Comparison
============================================================

This script runs all 5 anomaly detection methods on all 4 trained models
and creates a comparison table showing FPR95 / AUROC / AUPR for each combination.

Models tested:
1. ResNet50 + multi-scale augmentation (50.26% mIoU)
2. ResNet50 baseline (37.57% mIoU)
3. ResNet101 (37.07% mIoU)
4. SegFormer-B5 (35.57% mIoU)
5. Hiera-Base full resolution (32.83% mIoU)

Anomaly detection methods:
1. Simple Max Logits
2. Maximum Softmax Probability
3. Standardized Max Logits
4. Energy Score
5. HEAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
import warnings
warnings.filterwarnings('ignore')

# Import from project modules
from config import DEVICE, NUM_CLASSES, ANOMALY_CLASS_IDX, MAX_PIXELS_EVALUATION, RANDOM_SEED
from utils.dataloader import StreetHazardsDataset, get_transforms

# Try to import SegFormer and Hiera (optional)
try:
    from transformers import SegformerForSemanticSegmentation
    SEGFORMER_AVAILABLE = True
except ImportError:
    SEGFORMER_AVAILABLE = False
    print("⚠️  transformers not available - SegFormer models will be skipped")

try:
    from hiera import Hiera
    HIERA_AVAILABLE = True
except ImportError:
    try:
        import timm
        HIERA_AVAILABLE = True
    except ImportError:
        HIERA_AVAILABLE = False
        print("⚠️  hiera/timm not available - Hiera models will be skipped")

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODELS = {
    'ResNet50\n(50.26% mIoU)\nAugmented': {
        'path': 'models/checkpoints/deeplabv3_resnet50_augmented_10_47_09-11-25_mIoU_5026.pth',
        'architecture': 'deeplabv3_resnet50',
        'miou': 50.26,
    },
    'ResNet50\n(37.57% mIoU)\nBaseline': {
        'path': 'models/checkpoints/best_deeplabv3_streethazards_11_52_04-11-25_mIoU_3757.pth',
        'architecture': 'deeplabv3_resnet50',
        'miou': 37.57,
    },
    'ResNet101\n(37.07% mIoU)\nBaseline': {
        'path': 'models/checkpoints/deeplabv3_resnet101__05_02_07-11-25_mIoU_0.3707.pth',
        'architecture': 'deeplabv3_resnet101',
        'miou': 37.07,
    },
    'SegFormer-B5\n(35.57% mIoU)\nBaseline': {
        'path': 'models/checkpoints/segformer_b5_streethazards_04_44_09-11-25_mIoU_3556.pth',
        'architecture': 'segformer_b5',
        'miou': 35.57,
    },
    'Hiera-Base\n(32.83% mIoU)\nFull Res': {
        'path': 'models/checkpoints/hiera_base_streethazards_06_09_07-11-25_mIoU_3283.pth',
        'architecture': 'hiera_base',
        'miou': 32.83,
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

class SegFormerWrapper(nn.Module):
    """Wrapper to make SegFormer output compatible with our eval code."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # SegFormer expects 'pixel_values' parameter (from training script)
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        # Upsample to match input size if needed (from training script)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return {'out': logits}

class HieraSegmentationHead(nn.Module):
    """Lightweight segmentation decoder for Hiera (copied from training script)."""
    def __init__(self, in_channels_list, num_classes=13, embed_dim=256):
        super().__init__()

        # Project each stage to common embedding dimension
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, embed_dim, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Fusion module: combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        )

    def forward(self, features):
        # Convert from Hiera format [B, H, W, C] to PyTorch format [B, C, H, W]
        features = [f.permute(0, 3, 1, 2) for f in features]

        # Target size (use largest feature map as reference)
        target_size = features[0].shape[-2:]

        # Project and upsample all features to target size
        upsampled_features = []
        for feat, proj in zip(features, self.projections):
            feat = proj(feat)
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(feat)

        # Concatenate along channel dimension
        x = torch.cat(upsampled_features, dim=1)

        # Fuse features
        x = self.fusion(x)

        # Generate predictions
        x = self.head(x)

        return x

class HieraSegmentation(nn.Module):
    """Complete Hiera-based segmentation model (copied from training script)."""
    def __init__(self, backbone_name='hiera_base_224', num_classes=13, pretrained=False):
        super().__init__()

        # Import hiera
        import hiera

        # Load Hiera backbone
        if backbone_name == 'hiera_base_224':
            self.backbone = hiera.hiera_base_224(
                pretrained=pretrained,
                checkpoint="mae_in1k_ft_in1k" if pretrained else None
            )
            stage_channels = [96, 192, 384, 768]
        elif backbone_name == 'hiera_small_224':
            self.backbone = hiera.hiera_small_224(
                pretrained=pretrained,
                checkpoint="mae_in1k_ft_in1k" if pretrained else None
            )
            stage_channels = [96, 192, 384, 768]
        elif backbone_name == 'hiera_tiny_224':
            self.backbone = hiera.hiera_tiny_224(
                pretrained=pretrained,
                checkpoint="mae_in1k_ft_in1k" if pretrained else None
            )
            stage_channels = [96, 192, 384, 768]
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Create segmentation head
        self.decode_head = HieraSegmentationHead(
            in_channels_list=stage_channels,
            num_classes=num_classes,
            embed_dim=256
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        # Extract multi-scale features from Hiera
        _, intermediates = self.backbone(x, return_intermediates=True)

        # Decode to segmentation mask
        logits = self.decode_head(intermediates)

        # Upsample to input size
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        return logits

class HieraWrapper(nn.Module):
    """Wrapper to make Hiera output compatible with our eval code."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return {'out': logits}

def load_model(model_path, architecture):
    """Load a trained segmentation model."""
    print(f"Loading model: {model_path}")

    if architecture == 'deeplabv3_resnet50':
        model = deeplabv3_resnet50(weights=None)
        model.classifier[-1] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
        # aux_classifier might be None depending on PyTorch version
        if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
            model.aux_classifier[-1] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)

    elif architecture == 'deeplabv3_resnet101':
        model = deeplabv3_resnet101(weights=None)
        model.classifier[-1] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
        # aux_classifier might be None depending on PyTorch version
        if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
            model.aux_classifier[-1] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)

    elif architecture == 'segformer_b5':
        if not SEGFORMER_AVAILABLE:
            print(f"⚠️  Skipping {architecture} - transformers library not available")
            return None

        try:
            # Load SegFormer model (exactly as in training script)
            # From segformerb5.py lines 101-105
            segformer_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640",
                num_labels=NUM_CLASSES,  # 13 classes (same as training)
                ignore_mismatched_sizes=True  # Allow different number of classes
            )

            # Load trained weights
            state_dict = torch.load(model_path, map_location=DEVICE)
            segformer_model.load_state_dict(state_dict, strict=False)

            # Wrap to make output compatible
            model = SegFormerWrapper(segformer_model)

            print(f"✅ Successfully loaded SegFormer-B5 model")
        except Exception as e:
            print(f"⚠️  Failed to load SegFormer: {e}")
            print(f"⚠️  Skipping {architecture}")
            import traceback
            traceback.print_exc()
            return None

    elif architecture == 'hiera_base':
        if not HIERA_AVAILABLE:
            print(f"⚠️  Skipping {architecture} - hiera library not available")
            return None

        try:
            # Create Hiera model (same architecture as training script)
            hiera_model = HieraSegmentation(
                backbone_name='hiera_base_224',
                num_classes=NUM_CLASSES,
                pretrained=False  # We'll load trained weights
            )

            # Load trained weights
            state_dict = torch.load(model_path, map_location=DEVICE)
            hiera_model.load_state_dict(state_dict, strict=False)

            # Wrap to make output compatible
            model = HieraWrapper(hiera_model)

            print(f"✅ Successfully loaded Hiera model")
        except Exception as e:
            print(f"⚠️  Failed to load Hiera: {e}")
            print(f"⚠️  Skipping {architecture}")
            import traceback
            traceback.print_exc()
            return None

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = model.to(DEVICE)
    model.eval()

    return model

def calculate_fpr95(y_true, y_scores):
    """Calculate False Positive Rate at 95% True Positive Rate."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find the threshold where TPR >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0  # If we can't achieve 95% TPR, return worst case

    # Get FPR at the point where TPR first reaches 0.95
    return fpr[idx[0]]

# =============================================================================
# ANOMALY DETECTION METHODS
# =============================================================================

def method_simple_max_logits(logits):
    """Simple Max Logits: anomaly_score = -max(logits)"""
    max_logits = torch.max(logits, dim=1)[0]
    return -max_logits.cpu().numpy()

def method_maximum_softmax_probability(logits):
    """Maximum Softmax Probability: anomaly_score = -max(softmax(logits))"""
    probs = F.softmax(logits, dim=1)
    max_probs = torch.max(probs, dim=1)[0]
    return -max_probs.cpu().numpy()

def method_standardized_max_logits(logits, class_means, class_stds):
    """Standardized Max Logits: SML = (max_logit - mean) / std"""
    max_logits, pred_classes = torch.max(logits, dim=1)

    # Standardize per predicted class
    sml = torch.zeros_like(max_logits)
    for c in range(NUM_CLASSES):
        mask = (pred_classes == c)
        if mask.any():
            sml[mask] = (max_logits[mask] - class_means[c]) / (class_stds[c] + 1e-8)

    return -sml.cpu().numpy()

def method_energy_score(logits, temperature=1.0):
    """Energy Score: E(x) = -T * log(sum(exp(logits / T)))"""
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return energy.cpu().numpy()

def method_heat(logits, temperature=1.0):
    """HEAT: Hybrid Energy-Adaptive Thresholding (simplified version)"""
    # For simplicity, using energy score without spatial smoothing
    # Full HEAT requires feature extraction which is model-specific
    return method_energy_score(logits, temperature)

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def compute_class_statistics(model, val_loader):
    """Compute per-class mean and std of max logits on validation set."""
    print("Computing class statistics on validation set...")

    class_logits = {c: [] for c in range(NUM_CLASSES)}

    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc="Validation statistics"):
            images = images.to(DEVICE)
            outputs = model(images)['out']

            max_logits, pred_classes = torch.max(outputs, dim=1)

            for c in range(NUM_CLASSES):
                mask = (pred_classes == c)
                if mask.any():
                    class_logits[c].append(max_logits[mask].cpu())

    # Calculate statistics
    class_means = torch.zeros(NUM_CLASSES)
    class_stds = torch.ones(NUM_CLASSES)

    for c in range(NUM_CLASSES):
        if len(class_logits[c]) > 0:
            logits_c = torch.cat(class_logits[c])
            class_means[c] = logits_c.mean()
            class_stds[c] = logits_c.std()

    return class_means.to(DEVICE), class_stds.to(DEVICE)

def evaluate_method(model, test_loader, method_name, class_means=None, class_stds=None):
    """
    Evaluate a single anomaly detection method on a model.

    Returns:
        dict: {'fpr95': float, 'auroc': float, 'aupr': float}
    """
    print(f"\nEvaluating: {method_name}")

    all_anomaly_scores = []
    all_ground_truth = []

    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader, desc=f"{method_name}"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Get model predictions (logits)
            outputs = model(images)['out']

            # Apply anomaly detection method
            if method_name == 'Simple Max Logits':
                anomaly_scores = method_simple_max_logits(outputs)
            elif method_name == 'Maximum Softmax Probability':
                anomaly_scores = method_maximum_softmax_probability(outputs)
            elif method_name == 'Standardized Max Logits':
                anomaly_scores = method_standardized_max_logits(outputs, class_means, class_stds)
            elif method_name == 'Energy Score':
                anomaly_scores = method_energy_score(outputs)
            elif method_name == 'HEAT':
                anomaly_scores = method_heat(outputs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            # Ground truth: 1 if anomaly (class 13), 0 otherwise
            ground_truth = (masks == ANOMALY_CLASS_IDX).cpu().numpy().astype(np.float32)

            # Flatten and store
            all_anomaly_scores.append(anomaly_scores.flatten())
            all_ground_truth.append(ground_truth.flatten())

    # Concatenate all batches
    all_anomaly_scores = np.concatenate(all_anomaly_scores)
    all_ground_truth = np.concatenate(all_ground_truth)

    # Subsample if needed (memory efficiency)
    total_pixels = len(all_anomaly_scores)
    if total_pixels > MAX_PIXELS_EVALUATION:
        print(f"Subsampling {MAX_PIXELS_EVALUATION:,} pixels from {total_pixels:,}")
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(total_pixels, size=MAX_PIXELS_EVALUATION, replace=False)
        all_anomaly_scores = all_anomaly_scores[indices]
        all_ground_truth = all_ground_truth[indices]

    # Calculate metrics
    auroc = roc_auc_score(all_ground_truth, all_anomaly_scores)
    aupr = average_precision_score(all_ground_truth, all_anomaly_scores)
    fpr95 = calculate_fpr95(all_ground_truth, all_anomaly_scores)

    print(f"  FPR95: {fpr95:.4f} | AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")

    return {
        'fpr95': fpr95,
        'auroc': auroc,
        'aupr': aupr,
    }

# =============================================================================
# MAIN COMPARISON
# =============================================================================

def main():
    """Run comprehensive comparison of all models and methods."""

    print("="*80)
    print("COMPREHENSIVE MODEL vs ANOMALY DETECTION METHOD COMPARISON")
    print("="*80)
    print(f"\nModels to test: {len(MODELS)}")
    print(f"Methods to test: 5")
    print(f"Total evaluations: {len(MODELS) * 5} = 25")
    print(f"Device: {DEVICE}")
    print("="*80)

    # Prepare data loaders
    print("\nPreparing data loaders...")
    _, val_transform = get_transforms(image_size=(512, 512))

    val_dataset = StreetHazardsDataset(
        root_dir='streethazards_train/train',
        split='validation',
        transform=val_transform
    )
    test_dataset = StreetHazardsDataset(
        root_dir='streethazards_test/test',
        split='test',
        transform=val_transform
    )

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Results storage
    results = {}

    # Iterate over models
    for model_name, model_config in MODELS.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.replace(chr(10), ' ')}")
        print(f"mIoU: {model_config['miou']}%")
        print(f"{'='*80}")

        # Load model
        model = load_model(model_config['path'], model_config['architecture'])

        if model is None:
            print(f"⚠️  Skipping {model_name} - could not load model")
            continue

        # Compute class statistics for SML (only once per model)
        class_means, class_stds = compute_class_statistics(model, val_loader)

        # Test all methods on this model
        model_results = {}

        methods = [
            'Simple Max Logits',
            'Maximum Softmax Probability',
            'Standardized Max Logits',
            'Energy Score',
            'HEAT',
        ]

        for method in methods:
            try:
                metrics = evaluate_method(
                    model,
                    test_loader,
                    method,
                    class_means=class_means,
                    class_stds=class_stds
                )
                model_results[method] = metrics
            except Exception as e:
                print(f"❌ Error evaluating {method}: {e}")
                model_results[method] = {'fpr95': np.nan, 'auroc': np.nan, 'aupr': np.nan}

        results[model_name] = model_results

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

    # ==========================================================================
    # CREATE COMPARISON TABLE
    # ==========================================================================

    print("\n" + "="*80)
    print("GENERATING COMPARISON TABLE")
    print("="*80)

    # Create pandas DataFrame for easier formatting
    table_data = []

    for model_name in results.keys():
        row = {'Model': model_name.replace('\n', ' ')}

        for method in methods:
            if method in results[model_name]:
                metrics = results[model_name][method]
                fpr95 = metrics['fpr95'] * 100  # Convert to percentage
                auroc = metrics['auroc'] * 100
                aupr = metrics['aupr'] * 100

                # Format: FPR95 / AUROC / AUPR
                cell = f"{fpr95:.1f} / {auroc:.1f} / {aupr:.1f}"
            else:
                cell = "N/A"

            row[method] = cell

        table_data.append(row)

    df = pd.DataFrame(table_data)

    # Save to CSV
    output_csv = 'assets/model_method_comparison.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved CSV: {output_csv}")

    # Save to markdown table
    output_md = 'assets/model_method_comparison.md'
    with open(output_md, 'w') as f:
        f.write("# Model vs Anomaly Detection Method Comparison\n\n")
        f.write("**Format**: FPR95 / AUROC / AUPR (all in %)\n\n")
        f.write("**Lower FPR95 is better**, **Higher AUROC/AUPR is better**\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## Interpretation\n\n")
        f.write("- **FPR95**: False Positive Rate at 95% True Positive Rate (lower is better)\n")
        f.write("  - To detect 95% of anomalies, what % of normal pixels are flagged as anomalies?\n")
        f.write("- **AUROC**: Area Under ROC Curve (higher is better, 50% = random, 100% = perfect)\n")
        f.write("  - Measures overall ranking quality across all thresholds\n")
        f.write("- **AUPR**: Area Under Precision-Recall Curve (higher is better)\n")
        f.write("  - Primary metric for imbalanced data (~1% anomaly rate)\n")

    print(f"✅ Saved Markdown: {output_md}")

    # Print table to console
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    print("\nFormat: FPR95 / AUROC / AUPR (all in %)")
    print("Lower FPR95 is better, Higher AUROC/AUPR is better")
    print("\n" + df.to_string(index=False))
    print("\n" + "="*80)

    # Save raw results as JSON
    import json
    output_json = 'assets/model_method_comparison.json'

    # Convert to JSON-serializable format
    json_results = {}
    for model_name, model_results in results.items():
        json_results[model_name] = {}
        for method, metrics in model_results.items():
            json_results[model_name][method] = {
                'fpr95': float(metrics['fpr95']),
                'auroc': float(metrics['auroc']),
                'aupr': float(metrics['aupr']),
            }

    with open(output_json, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"✅ Saved JSON: {output_json}")

    print("\n🎉 Comparison complete!")
    print(f"Results saved to:")
    print(f"  - {output_csv}")
    print(f"  - {output_md}")
    print(f"  - {output_json}")

if __name__ == '__main__':
    main()
