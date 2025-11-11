"""
Comprehensive Test Suite for Standardized Max Logits (SML) Debugging

This script performs rigorous failure testing to identify why SML underperforms Simple Max Logits.

Test Suite:
1. Validation Statistics Sanity Checks
2. Logit Distribution Comparison (Val vs Test)
3. Standardization Effect Analysis
4. Numerical Stability Tests
5. Score Direction Verification
6. Side-by-Side Comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import sys

from dataloader import StreetHazardsDataset, get_transforms, CLASS_NAMES
from utils.model_utils import load_model
from config import (
    DEVICE, MODEL_PATH, NUM_CLASSES, ANOMALY_CLASS_IDX,
    IMAGE_SIZE, TRAIN_ROOT, TEST_ROOT, RANDOM_SEED,
    OUTPUT_DIR_ANOMALY as OUTPUT_DIR
)

# ============================================================================
# TEST 1: Compute and Validate Statistics
# ============================================================================
@torch.no_grad()
def test_1_compute_statistics(model, val_loader, device):
    """Test 1: Compute validation statistics and check for issues."""
    print("\n" + "="*80)
    print("TEST 1: VALIDATION STATISTICS SANITY CHECK")
    print("="*80)

    model.eval()

    # Using Welford's online algorithm for numerical stability
    class_count = {c: 0 for c in range(NUM_CLASSES)}
    class_mean = {c: 0.0 for c in range(NUM_CLASSES)}
    class_m2 = {c: 0.0 for c in range(NUM_CLASSES)}

    for images, _, _ in tqdm(val_loader, desc="Computing statistics"):
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

            # Welford's algorithm for online mean and variance
            n_old = class_count[c]
            n_new = n_old + n_pixels
            delta = values - class_mean[c]
            class_mean[c] += np.sum(delta) / n_new
            delta2 = values - class_mean[c]
            class_m2[c] += np.sum(delta * delta2)
            class_count[c] = n_new

    # Finalize statistics
    class_means, class_stds = {}, {}
    issues_found = False

    print(f"\n{'Class':<20} {'Count':<15} {'Mean':<12} {'Std':<12} {'Status'}")
    print("-"*75)

    for c in range(NUM_CLASSES):
        if class_count[c] > 1:
            mean = class_mean[c]
            std = np.sqrt(class_m2[c] / class_count[c])  # Population std
            class_means[c] = mean
            class_stds[c] = std
        else:
            mean = class_mean[c] if class_count[c] == 1 else 0.0
            std = 1.0
            class_means[c] = mean
            class_stds[c] = std

        # Check for issues
        status = "✓ OK"
        if class_count[c] == 0:
            status = "⚠ NO PIXELS"
            issues_found = True
        elif class_count[c] < 100:
            status = "⚠ LOW COUNT"
            issues_found = True
        elif np.isnan(mean) or np.isnan(std):
            status = "❌ NaN"
            issues_found = True
        elif np.isinf(mean) or np.isinf(std):
            status = "❌ Inf"
            issues_found = True
        elif std < 0.01:
            status = "⚠ LOW STD"
            issues_found = True
        elif std > 100:
            status = "⚠ HIGH STD"
            issues_found = True

        class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        print(f"{class_name:<20} {class_count[c]:<15,} {mean:<12.4f} {std:<12.4f} {status}")

    if issues_found:
        print("\n❌ TEST 1 FAILED: Issues detected in validation statistics")
    else:
        print("\n✅ TEST 1 PASSED: All validation statistics look reasonable")

    return class_means, class_stds, not issues_found


# ============================================================================
# TEST 2: Compare Logit Distributions (Validation vs Test)
# ============================================================================
@torch.no_grad()
def test_2_distribution_comparison(model, val_loader, test_loader, class_means, class_stds, device):
    """Test 2: Check if test set has similar logit distributions to validation."""
    print("\n" + "="*80)
    print("TEST 2: LOGIT DISTRIBUTION COMPARISON (VALIDATION vs TEST)")
    print("="*80)

    model.eval()

    # Sample validation set
    print("\nSampling validation set...")
    val_logits = {c: [] for c in range(NUM_CLASSES)}
    val_samples = 0
    for images, _, _ in tqdm(val_loader, desc="Validation", total=min(50, len(val_loader))):
        if val_samples >= 50:
            break
        images = images.to(device)
        output = model(images)['out']
        max_logits, pred_classes = output.max(dim=1)
        max_logits = max_logits.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()

        for c in range(NUM_CLASSES):
            mask = (pred_classes == c)
            if mask.any():
                val_logits[c].extend(max_logits[mask].flatten()[:1000])  # Sample up to 1000 pixels per class
        val_samples += 1

    # Sample test set
    print("Sampling test set...")
    test_logits = {c: [] for c in range(NUM_CLASSES)}
    test_samples = 0
    for images, _, _ in tqdm(test_loader, desc="Test", total=min(50, len(test_loader))):
        if test_samples >= 50:
            break
        images = images.to(device)
        output = model(images)['out']
        max_logits, pred_classes = output.max(dim=1)
        max_logits = max_logits.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()

        for c in range(NUM_CLASSES):
            mask = (pred_classes == c)
            if mask.any():
                test_logits[c].extend(max_logits[mask].flatten()[:1000])
        test_samples += 1

    # Compare distributions
    print(f"\n{'Class':<20} {'Val Mean':<12} {'Test Mean':<12} {'Mean Diff':<12} {'Status'}")
    print("-"*75)

    issues_found = False
    for c in range(NUM_CLASSES):
        if len(val_logits[c]) == 0 or len(test_logits[c]) == 0:
            continue

        val_mean = np.mean(val_logits[c])
        test_mean = np.mean(test_logits[c])
        diff = test_mean - val_mean
        diff_pct = 100 * diff / (abs(val_mean) + 1e-8)

        status = "✓ OK"
        if abs(diff_pct) > 20:  # More than 20% difference
            status = f"⚠ {diff_pct:+.1f}%"
            issues_found = True

        class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        print(f"{class_name:<20} {val_mean:<12.4f} {test_mean:<12.4f} {diff:<+12.4f} {status}")

    if issues_found:
        print("\n⚠ TEST 2 WARNING: Significant distribution shift detected")
        print("   → SML relies on validation statistics, may not generalize to test set")
    else:
        print("\n✅ TEST 2 PASSED: Test set distributions similar to validation")

    return not issues_found


# ============================================================================
# TEST 3: Standardization Effect Analysis
# ============================================================================
@torch.no_grad()
def test_3_standardization_effect(model, test_loader, class_means, class_stds, device):
    """Test 3: Check if standardization helps or hurts anomaly detection."""
    print("\n" + "="*80)
    print("TEST 3: STANDARDIZATION EFFECT ANALYSIS")
    print("="*80)

    model.eval()

    # Collect subset of test data
    np.random.seed(RANDOM_SEED)
    max_pixels = 100_000  # Sample for faster testing

    all_max_logits = []
    all_sml_scores = []
    all_simple_scores = []
    all_gt = []

    print("\nCollecting test samples...")
    for images, masks, _ in tqdm(test_loader, desc="Test sampling", total=min(20, len(test_loader))):
        if len(all_gt) > 0 and np.concatenate(all_gt).size > max_pixels:
            break

        images = images.to(device)
        masks = masks.numpy()

        output = model(images)['out']
        max_logits, pred_classes = output.max(dim=1)
        max_logits = max_logits.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()

        # Compute SML
        sml = np.zeros_like(max_logits)
        for c in range(NUM_CLASSES):
            mask = (pred_classes == c)
            if mask.any():
                sml[mask] = (max_logits[mask] - class_means[c]) / (class_stds[c] + 1e-8)

        # Compute anomaly scores
        simple_scores = -max_logits
        sml_scores = -sml

        gt = (masks == ANOMALY_CLASS_IDX).astype(int)

        all_max_logits.append(max_logits.flatten())
        all_simple_scores.append(simple_scores.flatten())
        all_sml_scores.append(sml_scores.flatten())
        all_gt.append(gt.flatten())

    # Concatenate
    all_max_logits = np.concatenate(all_max_logits)
    all_simple_scores = np.concatenate(all_simple_scores)
    all_sml_scores = np.concatenate(all_sml_scores)
    all_gt = np.concatenate(all_gt)

    # Subsample if needed
    if len(all_gt) > max_pixels:
        idx = np.random.choice(len(all_gt), max_pixels, replace=False)
        all_max_logits = all_max_logits[idx]
        all_simple_scores = all_simple_scores[idx]
        all_sml_scores = all_sml_scores[idx]
        all_gt = all_gt[idx]

    print(f"\nAnalyzing {len(all_gt):,} pixels ({all_gt.sum():,} anomalies, {100*all_gt.mean():.2f}%)")

    # Compute metrics for both methods
    auroc_simple = roc_auc_score(all_gt, all_simple_scores)
    aupr_simple = average_precision_score(all_gt, all_simple_scores)

    auroc_sml = roc_auc_score(all_gt, all_sml_scores)
    aupr_sml = average_precision_score(all_gt, all_sml_scores)

    print(f"\n{'Method':<25} {'AUROC':<12} {'AUPR':<12} {'Status'}")
    print("-"*55)
    print(f"{'Simple Max Logits':<25} {auroc_simple:<12.4f} {aupr_simple:<12.4f} {'baseline'}")
    print(f"{'Standardized (SML)':<25} {auroc_sml:<12.4f} {aupr_sml:<12.4f} {'test'}")

    auroc_diff = auroc_sml - auroc_simple
    aupr_diff = aupr_sml - aupr_simple

    print(f"\n{'Difference (SML - Simple)':<25} {auroc_diff:<+12.4f} {aupr_diff:<+12.4f}")

    if aupr_sml < aupr_simple:
        print("\n❌ TEST 3 FAILED: Standardization HURTS performance")
        print("   → SML AUPR is lower than Simple Max Logits")

        # Additional diagnostics
        print("\n   Diagnostics:")
        anomaly_mask = all_gt == 1
        normal_mask = all_gt == 0

        # Check score distributions
        simple_anom_mean = all_simple_scores[anomaly_mask].mean()
        simple_norm_mean = all_simple_scores[normal_mask].mean()
        sml_anom_mean = all_sml_scores[anomaly_mask].mean()
        sml_norm_mean = all_sml_scores[normal_mask].mean()

        print(f"   Simple Max Logits:")
        print(f"      Normal pixels:  mean score = {simple_norm_mean:.4f}")
        print(f"      Anomaly pixels: mean score = {simple_anom_mean:.4f}")
        print(f"      Separation: {simple_anom_mean - simple_norm_mean:+.4f}")

        print(f"   Standardized (SML):")
        print(f"      Normal pixels:  mean score = {sml_norm_mean:.4f}")
        print(f"      Anomaly pixels: mean score = {sml_anom_mean:.4f}")
        print(f"      Separation: {sml_anom_mean - sml_norm_mean:+.4f}")

        if (sml_anom_mean - sml_norm_mean) < (simple_anom_mean - simple_norm_mean):
            print("\n   → SML reduces class separation (worse discrimination)")

        return False
    else:
        print("\n✅ TEST 3 PASSED: Standardization helps performance")
        return True


# ============================================================================
# TEST 4: Numerical Stability
# ============================================================================
@torch.no_grad()
def test_4_numerical_stability(model, test_loader, class_means, class_stds, device):
    """Test 4: Check for NaN, Inf, or extreme values."""
    print("\n" + "="*80)
    print("TEST 4: NUMERICAL STABILITY CHECK")
    print("="*80)

    model.eval()

    nan_count = 0
    inf_count = 0
    extreme_count = 0
    total_pixels = 0

    print("\nChecking for numerical issues...")
    for images, _, _ in tqdm(test_loader, desc="Numerical check", total=min(20, len(test_loader))):
        if total_pixels > 100_000:
            break

        images = images.to(device)
        output = model(images)['out']
        max_logits, pred_classes = output.max(dim=1)
        max_logits = max_logits.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()

        # Compute SML
        sml = np.zeros_like(max_logits)
        for c in range(NUM_CLASSES):
            mask = (pred_classes == c)
            if mask.any():
                sml[mask] = (max_logits[mask] - class_means[c]) / (class_stds[c] + 1e-8)

        # Check for issues
        nan_count += np.isnan(sml).sum()
        inf_count += np.isinf(sml).sum()
        extreme_count += (np.abs(sml) > 100).sum()
        total_pixels += sml.size

    print(f"\nResults from {total_pixels:,} pixels:")
    print(f"  NaN values:     {nan_count:>10,} ({100*nan_count/total_pixels:.4f}%)")
    print(f"  Inf values:     {inf_count:>10,} ({100*inf_count/total_pixels:.4f}%)")
    print(f"  Extreme values: {extreme_count:>10,} ({100*extreme_count/total_pixels:.4f}%)")

    issues = nan_count + inf_count
    if issues > 0:
        print(f"\n❌ TEST 4 FAILED: {issues:,} numerical issues detected")
        return False
    elif extreme_count > total_pixels * 0.01:  # More than 1% extreme values
        print(f"\n⚠ TEST 4 WARNING: Many extreme values (|SML| > 100)")
        print("   → May indicate poor standardization or outliers")
        return False
    else:
        print("\n✅ TEST 4 PASSED: No numerical stability issues")
        return True


# ============================================================================
# TEST 5: Score Direction Verification
# ============================================================================
@torch.no_grad()
def test_5_score_direction(model, test_loader, class_means, class_stds, device):
    """Test 5: Verify that higher scores indicate anomalies."""
    print("\n" + "="*80)
    print("TEST 5: SCORE DIRECTION VERIFICATION")
    print("="*80)

    model.eval()

    # Collect samples with known anomalies
    anomaly_scores_sml = []
    normal_scores_sml = []

    print("\nCollecting samples with known anomalies...")
    samples_collected = 0
    for images, masks, _ in tqdm(test_loader, desc="Collecting", total=len(test_loader)):
        if samples_collected >= 100:  # Sample first 100 batches
            break

        images = images.to(device)
        masks = masks.numpy()

        # Check if batch has anomalies
        has_anomaly = (masks == ANOMALY_CLASS_IDX).any()
        if not has_anomaly:
            continue

        output = model(images)['out']
        max_logits, pred_classes = output.max(dim=1)
        max_logits = max_logits.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()

        # Compute SML
        sml = np.zeros_like(max_logits)
        for c in range(NUM_CLASSES):
            mask = (pred_classes == c)
            if mask.any():
                sml[mask] = (max_logits[mask] - class_means[c]) / (class_stds[c] + 1e-8)

        sml_scores = -sml  # Anomaly scores

        # Separate anomaly and normal pixels
        anomaly_mask = (masks == ANOMALY_CLASS_IDX)
        normal_mask = ~anomaly_mask

        anomaly_scores_sml.extend(sml_scores[anomaly_mask].flatten()[:1000])
        normal_scores_sml.extend(sml_scores[normal_mask].flatten()[:1000])

        samples_collected += 1

    anomaly_scores_sml = np.array(anomaly_scores_sml)
    normal_scores_sml = np.array(normal_scores_sml)

    print(f"\nCollected {len(anomaly_scores_sml):,} anomaly pixels, {len(normal_scores_sml):,} normal pixels")

    anom_mean = anomaly_scores_sml.mean()
    norm_mean = normal_scores_sml.mean()

    print(f"\nMean anomaly score:")
    print(f"  Anomaly pixels: {anom_mean:.4f}")
    print(f"  Normal pixels:  {norm_mean:.4f}")
    print(f"  Difference:     {anom_mean - norm_mean:+.4f}")

    if anom_mean > norm_mean:
        print("\n✅ TEST 5 PASSED: Anomaly pixels have higher scores (correct direction)")
        return True
    else:
        print("\n❌ TEST 5 FAILED: Anomaly pixels have LOWER scores (wrong direction!)")
        print("   → Bug in score computation or sign error")
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================
def run_all_tests():
    """Run all diagnostic tests."""
    print("="*80)
    print("COMPREHENSIVE SML DIAGNOSTIC TEST SUITE")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    # Load model and data
    print("\nLoading model...")
    model = load_model(MODEL_PATH, DEVICE)

    print("Loading datasets...")
    val_t, val_mask_t = get_transforms(IMAGE_SIZE, is_training=False)
    val_dataset = StreetHazardsDataset(TRAIN_ROOT, "validation", val_t, val_mask_t)
    test_dataset = StreetHazardsDataset(TEST_ROOT, "test", val_t, val_mask_t)

    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4, shuffle=False)

    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    # Run tests
    test_results = {}

    # Test 1: Validation statistics
    class_means, class_stds, test_results['test_1'] = test_1_compute_statistics(model, val_loader, DEVICE)

    # Test 2: Distribution comparison
    test_results['test_2'] = test_2_distribution_comparison(model, val_loader, test_loader, class_means, class_stds, DEVICE)

    # Test 3: Standardization effect
    test_results['test_3'] = test_3_standardization_effect(model, test_loader, class_means, class_stds, DEVICE)

    # Test 4: Numerical stability
    test_results['test_4'] = test_4_numerical_stability(model, test_loader, class_means, class_stds, DEVICE)

    # Test 5: Score direction
    test_results['test_5'] = test_5_score_direction(model, test_loader, class_means, class_stds, DEVICE)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.upper()}: {status}")

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n✅ ALL TESTS PASSED - SML implementation appears correct")
        print("   → Underperformance may be due to dataset characteristics")
    else:
        print(f"\n❌ {total_tests - passed_tests} TEST(S) FAILED")
        print("   → Review failed tests above for root cause")

    return test_results


if __name__ == "__main__":
    run_all_tests()
