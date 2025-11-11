# HEAT Implementation Guide

**Hybrid Energy-Adaptive Thresholding for Domain-Robust Anomaly Detection**

This document provides a practical step-by-step guide for implementing the proposed HEAT method.

---

## Quick Overview

HEAT combines three complementary anomaly scores:
1. **Energy Score** (logit-space) - Robust baseline
2. **Mahalanobis Distance** (feature-space) - Semantic outliers
3. **Spatial Consistency** (context) - Scene coherence

Plus **test-time adaptive normalization** to avoid SML's domain shift failure.

**Expected Performance:** 8-12% AUPR (vs. current 6.19%)

---

## Phase 1: Energy Score Baseline (Week 1)

### Goal
Implement energy-based OOD detection and validate it beats our current Max Logits baseline.

### Implementation Steps

#### Step 1.1: Energy Score Computation
```python
import torch
import torch.nn.functional as F

def compute_energy_score(logits, temperature=1.0):
    """
    Compute energy score for OOD detection.

    Args:
        logits: (B, C, H, W) - model logits
        temperature: float - temperature parameter (default: 1.0)

    Returns:
        energy: (B, H, W) - energy scores (lower = more in-distribution)
    """
    # Energy = -T * log(sum(exp(z_c / T)))
    # Equivalent to: -T * LogSumExp(z / T)
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return energy
```

#### Step 1.2: Integration with Existing Model
```python
# In your evaluation script

def evaluate_energy_baseline(model, test_loader, threshold):
    """Evaluate energy-based OOD detection."""
    all_scores = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()

            # Forward pass
            logits = model(images)

            # Compute energy scores
            energy = compute_energy_score(logits, temperature=1.0)

            # Anomaly detection
            # Higher energy = more OOD, so we negate for consistency
            anomaly_scores = -energy  # Now higher = more anomalous

            all_scores.append(anomaly_scores.cpu())
            all_labels.append(labels)

    # Compute metrics
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    # Binary labels: 1 = anomaly (class 13), 0 = known
    binary_labels = (labels == 13).long()

    return compute_metrics(scores.flatten(), binary_labels.flatten())
```

#### Step 1.3: Threshold Selection
```python
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(scores, labels):
    """Find threshold maximizing F1 or AUPR."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # F1-optimal threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold, f1_scores[optimal_idx]
```

### Expected Results
- **Target:** AUPR = 7-8% (conservative estimate)
- **Comparison:** Should beat Max Logits (6.19%) and significantly beat SML (3.70%)

### Validation Checklist
- [ ] Energy scores computed correctly (verify shape and range)
- [ ] Metrics match expected format (AUPR, FPR95, AUROC)
- [ ] Threshold selection on validation set
- [ ] Results better than Max Logits baseline

---

## Phase 2: Feature Statistics Extraction (Week 2)

### Goal
Extract and save class-wise feature statistics (mean, covariance) from training data.

### Implementation Steps

#### Step 2.1: Feature Extraction Hook
```python
class FeatureExtractor:
    """Extract intermediate features from model."""

    def __init__(self, model, layer_name='layer3'):
        self.features = None
        self.layer_name = layer_name

        # Register hook
        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.features = output

    def get_features(self):
        return self.features
```

#### Step 2.2: Compute Class Statistics
```python
def compute_class_statistics(model, train_loader, num_classes=13):
    """
    Compute mean and covariance for each class.

    Returns:
        class_means: dict mapping class_id -> mean vector (D,)
        class_covs: dict mapping class_id -> covariance matrix (D, D)
    """
    feature_extractor = FeatureExtractor(model, layer_name='layer3')

    # Collect features per class
    class_features = {c: [] for c in range(num_classes)}

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Extracting features"):
            images = images.cuda()
            labels = labels.cuda()

            # Forward pass
            _ = model(images)
            features = feature_extractor.get_features()  # (B, D, H, W)

            B, D, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
            labels_flat = labels.reshape(-1)  # (B*H*W,)

            # Group by class
            for c in range(num_classes):
                mask = (labels_flat == c)
                if mask.sum() > 0:
                    class_features[c].append(features[mask].cpu())

    # Compute statistics
    class_means = {}
    class_covs = {}

    for c in range(num_classes):
        if len(class_features[c]) > 0:
            feats = torch.cat(class_features[c], dim=0)  # (N, D)

            # Mean
            class_means[c] = feats.mean(dim=0)  # (D,)

            # Covariance (with regularization)
            feats_centered = feats - class_means[c]
            cov = (feats_centered.T @ feats_centered) / (feats.shape[0] - 1)

            # Add regularization to avoid singularity
            cov = cov + 1e-4 * torch.eye(cov.shape[0])
            class_covs[c] = cov  # (D, D)

    return class_means, class_covs
```

#### Step 2.3: Save Statistics
```python
import pickle

def save_statistics(class_means, class_covs, save_path):
    """Save feature statistics."""
    statistics = {
        'class_means': class_means,
        'class_covs': class_covs
    }
    with open(save_path, 'wb') as f:
        pickle.dump(statistics, f)
    print(f"Statistics saved to {save_path}")

# Usage
class_means, class_covs = compute_class_statistics(model, train_loader)
save_statistics(class_means, class_covs, 'feature_statistics.pkl')
```

### Memory Optimization

For large feature dimensions, use **tied covariance** (shared across classes):

```python
def compute_tied_covariance(model, train_loader, class_means, num_classes=13):
    """Compute single tied covariance matrix."""
    feature_extractor = FeatureExtractor(model, layer_name='layer3')

    all_features_centered = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            _ = model(images)
            features = feature_extractor.get_features()

            B, D, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, D)
            labels_flat = labels.reshape(-1)

            # Center by class mean
            for c in range(num_classes):
                mask = (labels_flat == c)
                if mask.sum() > 0:
                    features_c = features[mask] - class_means[c].cuda()
                    all_features_centered.append(features_c.cpu())

    # Compute tied covariance
    all_features_centered = torch.cat(all_features_centered, dim=0)
    N, D = all_features_centered.shape
    cov = (all_features_centered.T @ all_features_centered) / (N - 1)
    cov = cov + 1e-4 * torch.eye(D)

    return cov
```

### Validation Checklist
- [ ] Features extracted from correct layer (verify dimensions)
- [ ] Class means computed for all 13 classes
- [ ] Covariance matrices are positive definite
- [ ] Statistics saved successfully
- [ ] Memory usage acceptable (consider tied covariance if needed)

---

## Phase 3: Mahalanobis Distance (Week 2)

### Goal
Implement Mahalanobis distance computation for feature-space anomaly detection.

### Implementation Steps

#### Step 3.1: Mahalanobis Distance Computation
```python
def compute_mahalanobis_distance(features, class_means, class_covs, use_tied=False):
    """
    Compute minimum Mahalanobis distance to class prototypes.

    Args:
        features: (N, D) - feature vectors
        class_means: dict - class mean vectors
        class_covs: dict or tensor - class covariances (or tied)
        use_tied: bool - whether to use tied covariance

    Returns:
        distances: (N,) - minimum Mahalanobis distance across classes
    """
    num_classes = len(class_means)
    N, D = features.shape

    min_distances = torch.full((N,), float('inf'))

    for c in range(num_classes):
        if c not in class_means:
            continue

        # Centered features
        diff = features - class_means[c].to(features.device)  # (N, D)

        # Covariance matrix
        if use_tied:
            cov_inv = torch.inverse(class_covs).to(features.device)  # Shared cov
        else:
            cov_inv = torch.inverse(class_covs[c]).to(features.device)

        # Mahalanobis distance: sqrt((x - μ)^T Σ^-1 (x - μ))
        mahal = torch.sum(diff @ cov_inv * diff, dim=1)  # (N,)
        mahal = torch.sqrt(mahal + 1e-8)

        # Keep minimum distance
        min_distances = torch.minimum(min_distances, mahal)

    return min_distances
```

#### Step 3.2: Efficient Batch Computation
```python
def compute_mahalanobis_batch(model, images, class_means, class_covs):
    """Compute Mahalanobis distances for a batch."""
    feature_extractor = FeatureExtractor(model, layer_name='layer3')

    with torch.no_grad():
        _ = model(images)
        features = feature_extractor.get_features()  # (B, D, H, W)

    B, D, H, W = features.shape
    features_flat = features.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)

    # Compute distances
    distances = compute_mahalanobis_distance(
        features_flat, class_means, class_covs, use_tied=True
    )

    # Reshape back
    distances = distances.reshape(B, H, W)

    return distances
```

### Memory Optimization with Low-Rank Approximation

For very high-dimensional features:

```python
def compute_low_rank_mahalanobis(features, class_means, class_covs, rank=256):
    """
    Compute approximate Mahalanobis using low-rank covariance.

    Uses eigendecomposition: Σ ≈ U Λ U^T where U is (D, rank)
    """
    # Eigendecomposition of covariance
    eigenvalues, eigenvectors = torch.linalg.eigh(class_covs)

    # Keep top-k eigenvalues/vectors
    top_k_idx = torch.argsort(eigenvalues, descending=True)[:rank]
    eigenvalues_k = eigenvalues[top_k_idx]
    eigenvectors_k = eigenvectors[:, top_k_idx]

    # Approximate inverse: (U Λ U^T)^-1 ≈ U Λ^-1 U^T
    # For low-rank, this is much faster
    lambda_inv = 1.0 / (eigenvalues_k + 1e-6)

    # Compute distance
    for c in range(num_classes):
        diff = features - class_means[c]
        projected = diff @ eigenvectors_k  # (N, rank)
        mahal = torch.sum(projected ** 2 * lambda_inv, dim=1)
        # Continue as before...
```

### Validation Checklist
- [ ] Mahalanobis distances computed correctly
- [ ] Distances are non-negative (verify sqrt operation)
- [ ] Minimum distance across classes is selected
- [ ] Performance is acceptable (consider optimizations if slow)
- [ ] Memory usage manageable

---

## Phase 4: Spatial Consistency (Week 3)

### Goal
Implement spatial consistency scoring based on neighborhood agreement.

### Implementation Steps

#### Step 4.1: Spatial Consistency Computation
```python
def compute_spatial_consistency(softmax_probs, kernel_size=3):
    """
    Compute spatial consistency score.

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
```

#### Step 4.2: Alternative: Variance-Based Consistency
```python
def compute_variance_consistency(predictions, kernel_size=3):
    """
    Simpler alternative: measure prediction variance in neighborhood.

    Low variance = high consistency
    """
    B, C, H, W = predictions.shape

    # For each class, compute local variance
    padding = kernel_size // 2
    variances = []

    for c in range(C):
        class_pred = predictions[:, c:c+1, :, :]  # (B, 1, H, W)

        # Mean in neighborhood
        mean = F.avg_pool2d(class_pred, kernel_size, stride=1, padding=padding)

        # Variance = E[X^2] - E[X]^2
        mean_sq = F.avg_pool2d(class_pred ** 2, kernel_size, stride=1, padding=padding)
        var = mean_sq - mean ** 2
        variances.append(var)

    # Total variance
    total_var = torch.cat(variances, dim=1).sum(dim=1)  # (B, H, W)

    # Consistency = negative variance
    consistency = -total_var

    return consistency
```

### Validation Checklist
- [ ] Spatial consistency scores computed
- [ ] Anomalies have lower consistency (verify on examples)
- [ ] Edge effects handled (padding)
- [ ] Scores are in reasonable range

---

## Phase 5: Score Combination & Adaptive Normalization (Week 3)

### Goal
Combine the three scores with confidence weighting and adaptive normalization.

### Implementation Steps

#### Step 5.1: Reliability Weighting
```python
def compute_reliability_weights(energy, mahalanobis, spatial_consistency, softmax_probs):
    """
    Compute reliability weights for each score component.

    Returns:
        weights: tuple of (w_energy, w_mahal, w_spatial) as tensors
    """
    # Energy reliability: inverse entropy
    entropy = -(softmax_probs * torch.log(softmax_probs + 1e-8)).sum(dim=1)
    w_energy = 1.0 / (entropy + 1e-6)

    # Mahalanobis reliability: constant or inverse of feature variance
    # For simplicity, use constant weight
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
```

#### Step 5.2: Score Combination
```python
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

    # Combine
    combined = (
        w_e * (-energy) +           # Negative energy (higher = more anomalous)
        w_m * mahalanobis +         # Higher distance = more anomalous
        w_s * (-spatial_consistency) # Lower consistency = more anomalous
    )

    return combined
```

#### Step 5.3: Adaptive Normalization
```python
class AdaptiveNormalizer:
    """Test-time adaptive normalization using EMA."""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.mean_ema = None
        self.std_ema = None

    def update(self, scores):
        """
        Update statistics with new batch.

        Args:
            scores: (B, H, W) - anomaly scores
        """
        batch_mean = scores.mean()
        batch_std = scores.std()

        if self.mean_ema is None:
            self.mean_ema = batch_mean
            self.std_ema = batch_std
        else:
            self.mean_ema = self.alpha * self.mean_ema + (1 - self.alpha) * batch_mean
            self.std_ema = self.alpha * self.std_ema + (1 - self.alpha) * batch_std

    def normalize(self, scores):
        """
        Normalize scores using EMA statistics.

        Returns:
            normalized_scores: (B, H, W)
        """
        if self.mean_ema is None:
            return scores

        normalized = (scores - self.mean_ema) / (self.std_ema + 1e-6)
        return normalized

    def reset(self):
        """Reset statistics."""
        self.mean_ema = None
        self.std_ema = None
```

### Validation Checklist
- [ ] Weights sum to 1.0
- [ ] Combined scores in reasonable range
- [ ] Adaptive normalization updates correctly
- [ ] EMA statistics stabilize after few batches

---

## Phase 6: Full HEAT Integration (Week 4)

### Goal
Integrate all components into complete HEAT pipeline.

### Implementation

#### Step 6.1: HEAT Class
```python
class HEAT:
    """
    Hybrid Energy-Adaptive Thresholding for OOD detection.
    """

    def __init__(self, model, class_means, class_covs,
                 temperature=1.0, alpha=0.9, use_tied_cov=True):
        self.model = model
        self.class_means = class_means
        self.class_covs = class_covs
        self.temperature = temperature
        self.use_tied_cov = use_tied_cov

        self.feature_extractor = FeatureExtractor(model, layer_name='layer3')
        self.normalizer = AdaptiveNormalizer(alpha=alpha)

    def forward(self, images):
        """
        Compute HEAT anomaly scores.

        Args:
            images: (B, 3, H, W)

        Returns:
            anomaly_scores: (B, H, W)
        """
        with torch.no_grad():
            # Forward pass
            logits = self.model(images)
            features = self.feature_extractor.get_features()

            # Softmax probabilities
            softmax_probs = F.softmax(logits, dim=1)

            # Component 1: Energy Score
            energy = compute_energy_score(logits, self.temperature)

            # Component 2: Mahalanobis Distance
            B, D, H, W = features.shape
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, D)
            mahal_flat = compute_mahalanobis_distance(
                features_flat, self.class_means, self.class_covs,
                use_tied=self.use_tied_cov
            )
            mahal = mahal_flat.reshape(B, H, W)

            # Component 3: Spatial Consistency
            spatial_cons = compute_spatial_consistency(softmax_probs)

            # Combine scores
            combined = combine_scores(energy, mahal, spatial_cons, softmax_probs)

            # Adaptive normalization
            self.normalizer.update(combined)
            normalized = self.normalizer.normalize(combined)

        return normalized

    def evaluate(self, test_loader, threshold):
        """
        Evaluate HEAT on test set.

        Returns:
            metrics: dict with AUPR, FPR95, AUROC
        """
        all_scores = []
        all_labels = []

        self.model.eval()
        for images, labels in tqdm(test_loader, desc="Evaluating HEAT"):
            images = images.cuda()

            scores = self.forward(images)

            all_scores.append(scores.cpu())
            all_labels.append(labels)

        # Compute metrics
        scores = torch.cat(all_scores).flatten()
        labels = torch.cat(all_labels).flatten()
        binary_labels = (labels == 13).long()  # Anomaly class

        metrics = compute_metrics(scores, binary_labels)
        return metrics
```

#### Step 6.2: Usage Example
```python
# Load feature statistics
with open('feature_statistics.pkl', 'rb') as f:
    stats = pickle.load(f)
    class_means = stats['class_means']
    class_covs = stats['class_covs']

# Initialize HEAT
heat = HEAT(
    model=model,
    class_means=class_means,
    class_covs=class_covs,
    temperature=1.0,
    alpha=0.9,
    use_tied_cov=True
)

# Evaluate
metrics = heat.evaluate(test_loader, threshold=0.0)

print(f"AUPR: {metrics['aupr']:.4f}")
print(f"FPR95: {metrics['fpr95']:.4f}")
print(f"AUROC: {metrics['auroc']:.4f}")
```

### Validation Checklist
- [ ] All components integrated
- [ ] End-to-end pipeline works
- [ ] Metrics computed correctly
- [ ] Performance meets targets (8-12% AUPR)

---

## Hyperparameter Tuning

### Key Hyperparameters

```python
hyperparams = {
    'temperature': [0.5, 1.0, 1.5, 2.0],     # Energy score temperature
    'alpha': [0.8, 0.9, 0.95],                # EMA decay for normalization
    'kernel_size': [3, 5],                    # Spatial consistency neighborhood
    'use_tied_cov': [True, False],            # Tied vs. class-wise covariance
    'layer_name': ['layer2', 'layer3', 'layer4'],  # Feature extraction layer
}
```

### Grid Search
```python
from itertools import product

def grid_search(model, val_loader, hyperparams):
    """Find best hyperparameters on validation set."""
    best_aupr = 0
    best_params = None

    for temp, alpha, ks, tied, layer in product(*hyperparams.values()):
        # Compute statistics if layer changes
        if layer != 'layer3':
            class_means, class_covs = compute_class_statistics(
                model, train_loader, layer_name=layer
            )

        # Initialize HEAT
        heat = HEAT(model, class_means, class_covs,
                   temperature=temp, alpha=alpha, use_tied_cov=tied)

        # Modify spatial consistency kernel size
        # (requires minor code adjustment)

        # Evaluate
        metrics = heat.evaluate(val_loader, threshold=0.0)

        if metrics['aupr'] > best_aupr:
            best_aupr = metrics['aupr']
            best_params = {
                'temperature': temp,
                'alpha': alpha,
                'kernel_size': ks,
                'use_tied_cov': tied,
                'layer_name': layer
            }

    return best_params, best_aupr
```

---

## Ablation Studies

### Study 1: Individual Components
```python
def ablation_individual_components(model, test_loader):
    """Test each component individually."""
    results = {}

    # Energy only
    results['energy'] = evaluate_energy_baseline(model, test_loader, threshold=0.0)

    # Mahalanobis only
    results['mahalanobis'] = evaluate_mahalanobis_only(model, test_loader)

    # Spatial consistency only
    results['spatial'] = evaluate_spatial_only(model, test_loader)

    # HEAT (all combined)
    results['heat'] = heat.evaluate(test_loader, threshold=0.0)

    # Print comparison
    for name, metrics in results.items():
        print(f"{name:15s} - AUPR: {metrics['aupr']:.4f}")
```

### Study 2: Adaptive vs. Fixed Normalization
```python
def ablation_normalization(model, test_loader):
    """Compare adaptive vs. fixed normalization."""
    # With adaptive
    heat_adaptive = HEAT(model, class_means, class_covs, alpha=0.9)
    metrics_adaptive = heat_adaptive.evaluate(test_loader, threshold=0.0)

    # Without adaptive (alpha=0 means no adaptation)
    heat_fixed = HEAT(model, class_means, class_covs, alpha=0.0)
    metrics_fixed = heat_fixed.evaluate(test_loader, threshold=0.0)

    print(f"Adaptive: AUPR = {metrics_adaptive['aupr']:.4f}")
    print(f"Fixed:    AUPR = {metrics_fixed['aupr']:.4f}")
    print(f"Gain:     {(metrics_adaptive['aupr'] - metrics_fixed['aupr']):.4f}")
```

### Study 3: Feature Layer Selection
```python
def ablation_feature_layers(model, test_loader):
    """Compare different feature extraction layers."""
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    results = {}

    for layer in layers:
        # Extract features from this layer
        class_means, class_covs = compute_class_statistics(
            model, train_loader, layer_name=layer
        )

        # Evaluate
        heat = HEAT(model, class_means, class_covs, layer_name=layer)
        metrics = heat.evaluate(test_loader, threshold=0.0)
        results[layer] = metrics

        print(f"{layer:10s} - AUPR: {metrics['aupr']:.4f}")
```

---

## Troubleshooting

### Issue 1: Out of Memory
**Symptoms:** CUDA OOM error during Mahalanobis computation

**Solutions:**
1. Use tied covariance instead of class-wise
2. Reduce feature dimension via PCA or low-rank approximation
3. Process smaller batches
4. Use CPU for covariance operations

```python
# Solution: Low-rank approximation
def reduce_covariance_rank(cov, target_rank=256):
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    top_k = torch.argsort(eigenvalues, descending=True)[:target_rank]
    return eigenvalues[top_k], eigenvectors[:, top_k]
```

### Issue 2: Numerical Instability
**Symptoms:** NaN or Inf values in energy scores or Mahalanobis distances

**Solutions:**
1. Add epsilon to avoid log(0) and division by zero
2. Clamp extreme values
3. Use double precision for covariance inversion

```python
# Solution: Robust computation
energy = torch.clamp(-temperature * torch.logsumexp(logits / temperature, dim=1),
                     min=-100, max=100)
```

### Issue 3: Poor Performance
**Symptoms:** AUPR below expected range (< 7%)

**Debugging Steps:**
1. Verify each component individually (ablation)
2. Check feature extraction is from correct layer
3. Ensure statistics are computed correctly
4. Try different hyperparameters
5. Visualize anomaly score maps

```python
# Visualization
import matplotlib.pyplot as plt

def visualize_scores(images, scores, labels):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(images[0].permute(1, 2, 0).cpu())
    axes[0].set_title('Input Image')
    axes[1].imshow(scores[0].cpu(), cmap='hot')
    axes[1].set_title('HEAT Scores')
    axes[2].imshow(labels[0].cpu(), cmap='gray')
    axes[2].set_title('Ground Truth')
    plt.show()
```

---

## Performance Benchmarks

### Expected Timeline

| Phase | Duration | Expected AUPR | Status |
|-------|----------|---------------|--------|
| Energy Baseline | Week 1 | 7-8% | To do |
| + Mahalanobis | Week 2 | 8-9% | To do |
| + Spatial | Week 3 | 9-10% | To do |
| + Adaptive Norm | Week 3 | 10-11% | To do |
| Hyperparameter Tuning | Week 4 | 11-12% | To do |

### Success Criteria

**Minimum Viable Performance:**
- AUPR ≥ 8% (beating Max Logits by 2%)
- FPR95 ≤ 28%
- AUROC ≥ 89%

**Target Performance:**
- AUPR ≥ 10%
- FPR95 ≤ 25%
- AUROC ≥ 90%

**Stretch Goals:**
- AUPR ≥ 12%
- FPR95 ≤ 20%
- AUROC ≥ 92%

---

## Next Steps After HEAT

If HEAT meets performance targets, consider:

1. **ATTA Integration:** Add test-time batch norm adaptation for further improvement
2. **Synthetic Training:** Use VOS or AnoGen to generate training anomalies
3. **Ensemble:** Combine multiple HEAT instances with different hyperparameters
4. **Multi-Scale:** Apply HEAT at multiple feature pyramid levels
5. **Temporal:** Add temporal consistency for video sequences

---

## References

See `research.md` Section 7 and `key_papers.md` for complete references and paper links.

**Key Implementation References:**
- Energy Score: Liu et al., NeurIPS 2020
- Mahalanobis: Lee et al., NeurIPS 2018
- Adaptive Normalization: Inspired by ATTA (Gao et al., NeurIPS 2023)

---

**Document Version:** 1.0
**Last Updated:** November 6, 2025
**Status:** Ready for implementation
