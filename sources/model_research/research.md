# Semantic Segmentation & Anomaly Detection Research

**Research Date:** November 4, 2025
**Purpose:** Phase 2/3 literature review for ML4CV assignment on anomaly segmentation

---

## Table of Contents
1. [Paper 1: DeepLabV3+ (Architecture)](#paper-1-deeplabv3-architecture)
2. [Paper 2: U-Net (Architecture)](#paper-2-u-net-architecture)
3. [Paper 3: Standardized Max Logits (Anomaly Detection)](#paper-3-standardized-max-logits)
4. [Paper 4: Scaling OOD Detection (StreetHazards Dataset)](#paper-4-scaling-ood-detection)
5. [Paper 5: Deep Metric Learning (Advanced Anomaly Detection)](#paper-5-deep-metric-learning)
6. [Common Threads & Key Insights](#common-threads--key-insights)
7. [Recommendations for Implementation](#recommendations-for-implementation)

---

## Paper 1: DeepLabV3+ (Architecture)

### Citation
**Title:** Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
**Authors:** Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam
**Conference:** ECCV 2018
**Links:**
- arXiv: https://arxiv.org/abs/1802.02611
- PDF: https://arxiv.org/pdf/1802.02611
- **Local PDF:** `sources/deeplabv3plus.pdf`

### Core Contribution
DeepLabV3+ combines **spatial pyramid pooling** (for multi-scale context) with **encoder-decoder structure** (for sharp boundary refinement). This addresses the complementary requirements of:
1. **Global context understanding** (what is the object?)
2. **Precise localization** (where are its exact boundaries?)

### Architecture Details

#### 1. Atrous Spatial Pyramid Pooling (ASPP)
- Probes features at multiple dilation rates (6, 12, 18)
- Captures multi-scale contextual information
- Mathematical formulation: Atrous convolution with rate `r`:
  ```
  y[i] = Σ x[i + r·k] · w[k]
  ```
  where `r` controls the receptive field size

#### 2. Encoder-Decoder Structure
- **Encoder:** Extracts rich semantic features (via ResNet/Xception + ASPP)
- **Decoder:** Gradually recovers spatial information
  - Takes low-level features from encoder (skip connections)
  - Upsamples and refines boundaries
  - Uses 3x3 convolutions to smooth predictions

#### 3. Depthwise Separable Convolutions
- Replaces standard convolutions throughout
- **Benefit:** ~10x fewer parameters, faster inference
- **Formula:** Depthwise conv + 1x1 pointwise conv
  - Computational cost: `D_K × D_K × M × D_F × D_F + M × N × D_F × D_F`
  - vs standard: `D_K × D_K × M × N × D_F × D_F`

### Design Reasoning

**Why combine spatial pyramid + encoder-decoder?**
- Spatial pyramid alone lacks fine boundary details (outputs are 1/16 or 1/8 resolution)
- Encoder-decoder alone may lack rich semantic context
- **Together:** Best of both worlds

**Why Xception backbone?**
- Depthwise separable convolutions are computationally efficient
- Deeper networks extract better features
- Modified Xception (aligned with ResNet) combines efficiency + performance

### Performance

| Dataset | mIoU | Notes |
|---------|------|-------|
| PASCAL VOC 2012 | 89.0% | Without post-processing |
| Cityscapes | 82.1% | Test set |

### Key Takeaway for Our Project
✅ **Use DeepLabV3+ as baseline architecture**
- Proven performance on urban scenes (Cityscapes ~= StreetHazards setting)
- PyTorch has pretrained models available
- Good balance of accuracy and speed

---

## Paper 2: U-Net (Architecture)

### Citation
**Title:** U-Net: Convolutional Networks for Biomedical Image Segmentation
**Authors:** Olaf Ronneberger, Philipp Fischer, Thomas Brox
**Conference:** MICCAI 2015
**Links:**
- arXiv: https://arxiv.org/abs/1505.04597
- PDF: https://arxiv.org/pdf/1505.04597
- Project page: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
- **Local PDF:** `sources/unet.pdf`

### Core Contribution
U-Net introduces a **symmetric contracting-expanding architecture** optimized for:
1. **Limited training data** (crucial for medical imaging)
2. **Precise localization** (pixel-perfect segmentation)
3. **Fast inference** (<1 second per image)

### Architecture Details

#### Contracting Path (Encoder)
- Repeated: 3x3 conv → ReLU → 3x3 conv → ReLU → 2x2 max pool
- Doubles feature channels at each downsampling
- Captures **contextual information** at multiple scales

#### Expanding Path (Decoder)
- Repeated: 2x2 up-conv → concatenate with encoder features → 3x3 conv → ReLU
- **Skip connections** from encoder preserve spatial details
- Halves feature channels at each upsampling

#### Key Innovation: Skip Connections
```
Encoder features -----> Decoder features
     |                       ↑
     └───────(concat)────────┘
```
- Directly passes high-resolution features from encoder to decoder
- Enables precise localization while maintaining context

### Design Reasoning

**Why symmetric U-shape?**
- Encoder reduces spatial resolution → captures "what"
- Decoder increases spatial resolution → captures "where"
- Symmetry ensures balanced information flow

**Why strong data augmentation?**
- Medical images are expensive to annotate (limited data)
- Augmentations: elastic deformations, rotations, shifts, intensity variations
- **Result:** Network learns invariances, generalizes better with fewer samples

**Mathematical insight:** Overlap-tile strategy
- Input: larger patch than output
- Predicts center region, uses border as context
- Enables segmentation of arbitrarily large images

### Performance
- **ISBI cell tracking challenge 2015:** 1st place (92% average IOU)
- Fast: <1 second inference on GPU

### Key Takeaway for Our Project
⚠️ **Alternative to DeepLabV3+**
- Simpler architecture, easier to implement from scratch
- Better for limited data scenarios (but StreetHazards has 5k+ images)
- Skip connections are crucial for boundary refinement
- **Decision:** Use DeepLabV3+ (better for large datasets, pretrained models available)

---

## Paper 3: Standardized Max Logits (Anomaly Detection)

### Citation
**Title:** Standardized Max Logits: A Simple yet Effective Approach for Identifying Unexpected Road Obstacles in Urban-Scene Segmentation
**Authors:** Sanghun Jung, Jungsoo Lee, Daehoon Gwak, Sungha Choi, Jaegul Choo
**Conference:** ICCV 2021 (Oral)
**Links:**
- arXiv: https://arxiv.org/abs/2107.11264
- ICCV Open Access: https://openaccess.thecvf.com/content/ICCV2021/html/Jung_Standardized_Max_Logits_A_Simple_yet_Effective_Approach_for_Identifying_ICCV_2021_paper.html
- GitHub: https://github.com/shjung13/standardized-max-logits
- **Local PDF:** `sources/standardized_max_logits.pdf`

### Core Problem
Existing anomaly detection methods face a critical issue:
> "The distribution of max logits of each predicted class is significantly different from each other"

**Example:**
- "Road" class: max logits typically range [5, 10]
- "Pedestrian" class: max logits typically range [2, 7]
- **Issue:** Directly comparing max logits across classes is meaningless!

### Core Intuition
If neighboring pixels belong to the same class, their max logit distributions should align. Anomalies will have **abnormally low standardized scores** compared to their local neighborhood.

### Methodology

#### Step 1: Extract Max Logits
For each pixel `i` with predicted class `c`:
```
max_logit[i] = max(logits[i])  // Maximum unnormalized prediction score
```

#### Step 2: Standardize Per-Class
For each class `c`, compute statistics over all pixels predicted as `c`:
```
μ_c = mean(max_logits[class == c])
σ_c = std(max_logits[class == c])
```

#### Step 3: Compute Standardized Max Logit
```
SML[i] = (max_logit[i] - μ_c) / σ_c
```

where `c = argmax(logits[i])` (predicted class)

#### Step 4: Local Spatial Standardization
Consider two perspectives:
1. **Pixel-wise:** Standardize each pixel
2. **Region-wise:** Standardize within local windows (spatial smoothing)

**Formula (simplified):**
```
anomaly_score[i] = -SML[i]  // Lower SML → higher anomaly probability
```

### Why It Works

**Key Insight:** Standardization normalizes class-specific distributions
- In-distribution pixels: SML ≈ 0 (near mean)
- Out-of-distribution pixels: SML << 0 (far below mean)

**Benefits:**
1. ✅ **No retraining required** - works with any pretrained model
2. ✅ **No external data** - uses only model's own predictions
3. ✅ **Fast** - simple post-processing operation
4. ✅ **Effective** - SOTA on Fishyscapes Lost & Found leaderboard

### Mathematical Justification
From information theory perspective:
- High max logit = high confidence = low uncertainty
- Low max logit (after standardization) = model is uncertain → potential anomaly

### Key Takeaway for Our Project
🎯 **START HERE for Phase 4**
- Simplest effective method for anomaly detection
- Requires only trained segmentation model
- Literally 20 lines of code to implement
- Expected performance: 15-25% AUPR on StreetHazards

---

## Paper 4: Scaling OOD Detection (StreetHazards Dataset)

### Citation
**Title:** Scaling Out-of-Distribution Detection for Real-World Settings
**Authors:** Dan Hendrycks, Steven Basart, Mantas Mazeika, Andy Zou, Joe Kwon, Mohammadreza Mostajabi, Jacob Steinhardt, Dawn Song
**Conference:** ICML 2022
**Links:**
- arXiv: https://arxiv.org/abs/1911.11132
- ICML Proceedings: https://proceedings.mlr.press/v162/hendrycks22a.html
- PDF: https://proceedings.mlr.press/v162/hendrycks22a/hendrycks22a.pdf
- GitHub (Dataset + Code): https://github.com/hendrycks/anomaly-seg
- **Local PDF:** `sources/streethazards_hendrycks.pdf`

### Core Contribution
This paper **created the StreetHazards dataset** we're using! It scales OOD detection research from small toy datasets to real-world large-scale settings.

### StreetHazards Dataset

#### Creation Process
1. **Base:** CARLA simulator + Unreal Engine
2. **Method:** Insert foreign objects (250 types) into driving scenes
3. **Realism:** Physically-based rendering with proper lighting/shadows
4. **Scale:**
   - 5,125 training images (12 known classes)
   - 1,031 validation images (12 known classes)
   - 1,500 test images (12 known + anomaly class)
   - Resolution: 1280×720

#### 12 Known Classes
0. unlabeled, 1. building, 2. fence, 3. other, 4. pedestrian, 5. pole, 6. road line, 7. road, 8. sidewalk, 9. vegetation, 10. car, 11. wall, 12. traffic sign

#### 13th Class: Anomaly
- 250 different anomalous object types
- Examples: animals (cats, dogs), debris, unusual vehicles
- **Critical:** Only appears in test set (train/val have 0 anomalies)

### Key Findings

#### Simple Max Logit Baseline
The paper shows that **maximum logit** (unnormalized prediction score) outperforms complex methods:

```python
anomaly_score[i] = -max(logits[i])  # Simple but effective!
```

**Why max logit works:**
- In-distribution: Model confident → high max logit → low anomaly score
- Out-of-distribution: Model uncertain → low max logit → high anomaly score

#### Performance Benchmarks
- Established baseline metrics for StreetHazards
- Shows that scaling to realistic settings (high-res, many classes) changes what methods work best
- Simpler methods (max logit) often outperform complex ones at scale

### Combined Anomalous Object Segmentation (CAOS) Benchmark
- **StreetHazards** (synthetic)
- **BDD-Anomaly** (real-world)
- Tests both synthetic and real-world generalization

### Key Takeaway for Our Project
📊 **This is our benchmark!**
- Use official splits (don't mix train/test)
- Baseline to beat: max logit anomaly detection
- Focus on AUPR metric for anomaly detection
- Model must generalize to 250 unseen object types

---

## Paper 5: Deep Metric Learning (Advanced Anomaly Detection)

### Citation
**Title:** Deep Metric Learning for Open World Semantic Segmentation
**Authors:** Jun Cen, Peng Yun, Junhao Cai, Michael Yu Wang, Ming Liu
**Conference:** ICCV 2021
**Links:**
- arXiv: https://arxiv.org/abs/2108.04562
- ICCV Open Access: https://openaccess.thecvf.com/content/ICCV2021/html/Cen_Deep_Metric_Learning_for_Open_World_Semantic_Segmentation_ICCV_2021_paper.html
- GitHub: https://github.com/Jun-CEN/Open-World-Semantic-Segmentation
- **Local PDF:** `sources/deep_metric_learning.pdf`

### Core Contribution
**DMLNet** - A metric learning approach that learns to:
1. **Detect** out-of-distribution objects
2. **Incrementally learn** new object classes with few-shot learning

### Key Distinction from Pixel-wise Classification

#### Traditional Segmentation (Pixel-wise)
```
Input → CNN → Logits → Softmax → Class per pixel
```
- **Issue:** Closed set - can only predict known classes
- Anomalies get misclassified as most similar known class

#### Metric Learning Approach
```
Input → CNN → Feature embeddings → Distance to class prototypes
```
- Learns **feature space** where similar objects are close
- Can detect when object is far from all known prototypes
- **Formula:**
  ```
  distance[i, c] = ||embedding[i] - prototype[c]||
  ```

### Methodology

#### 1. Contrastive Clustering
- **Positive pairs:** Pixels from same class (pull together)
- **Negative pairs:** Pixels from different classes (push apart)
- **Loss function (simplified):**
  ```
  L = Σ [||f_i - f_j||² · (y_i == y_j) + max(0, m - ||f_i - f_j||²) · (y_i != y_j)]
  ```
  where `m` is margin, `f_i` are feature embeddings, `y_i` are labels

#### 2. Open-Set Detection
- Compute distances to all known class prototypes
- If `min(distances) > threshold` → classify as anomaly
- Threshold learned from validation set

#### 3. Incremental Few-Shot Learning
- When anomaly detected, human can label it
- Update model with few examples (1-5 shots)
- Add new prototype to embedding space

### Why It Works Better Than Max Logits

| Aspect | Max Logits | Metric Learning |
|--------|-----------|----------------|
| Training | Standard cross-entropy | Contrastive learning |
| Feature space | Implicit (final layer) | Explicit (embedding space) |
| Anomaly detection | Threshold on confidence | Distance to prototypes |
| Adaptability | Fixed classes | Incremental learning |
| Performance | Good | Better (SOTA) |

### Design Reasoning

**Philosophy:** Behave like humans
- Humans recognize unknown objects by their dissimilarity to known objects
- Embedding space explicitly models similarity/dissimilarity
- Natural extension to continual learning (add new classes over time)

### Performance
Achieves SOTA on three benchmarks:
- StreetHazards
- BDD-Anomaly
- Fishyscapes

**Without:**
- ❌ External OOD data
- ❌ Generative models
- ❌ Complex post-processing

### Key Takeaway for Our Project
🚀 **For Phase 5 (Advanced Method)**
- More sophisticated than max logits
- Requires modifying training (contrastive loss)
- Expected improvement: +3-5% AUPR over max logits
- Trade-off: More complex to implement

---

## Common Threads & Key Insights

### 1. Architecture Patterns

#### Encoder-Decoder is Standard
All modern segmentation architectures use encoder-decoder:
- **Encoder:** Capture context (what objects exist?)
- **Decoder:** Refine boundaries (where exactly?)
- **Skip connections:** Preserve spatial details

#### Multi-Scale Processing
Multiple papers emphasize multi-scale features:
- DeepLabV3+: Atrous spatial pyramid pooling
- U-Net: Multiple resolution paths
- **Reason:** Objects appear at different scales in images

### 2. Anomaly Detection Hierarchy

**Level 1 (Simplest):** Max Logit / Max Softmax Probability
- No retraining required
- Works surprisingly well
- Fast and simple

**Level 2 (Better):** Standardized Max Logits
- Still no retraining
- Normalizes class-specific distributions
- SOTA among training-free methods

**Level 3 (Best):** Metric Learning
- Requires retraining with contrastive loss
- Learns explicit embedding space
- Best performance, but more complex

### 3. The "Simplicity Paradox"

**Observation:** Simpler methods often work better at scale!

**Why?**
- Complex methods overfit to small benchmarks
- Simple methods are more robust
- Computational efficiency matters for real-world deployment

**Hendrycks et al.:** "A surprisingly simple detector based on the maximum logit outperforms prior methods"

### 4. Training Philosophy

#### Data Augmentation is Critical
- U-Net: Strong elastic deformations for limited data
- DeepLabV3+: Random crops, flips, color jitter
- **Takeaway:** Even with 5k images, augmentation helps

#### Pretrained Backbones
- All papers use ImageNet pretrained encoders
- Transfer learning from 1000-class classification
- **Benefit:** Better features, faster convergence

### 5. Evaluation Metrics

#### For Segmentation (Known Classes)
- **mIoU** (mean Intersection over Union)
- Standard metric across all papers
- Target: 40-60% on StreetHazards

#### For Anomaly Detection
- **AUPR** (Area Under Precision-Recall curve)
- Better than AUROC for imbalanced data (anomalies are rare)
- Target: 15-25% baseline, 25-30% advanced

---

## Recommendations for Implementation

### Phase 2: Baseline Model (Now - Next 8 hours)

#### Architecture Choice: DeepLabV3+
**Reasoning:**
1. ✅ Proven on Cityscapes (similar to StreetHazards)
2. ✅ Pretrained models available in PyTorch
3. ✅ Good balance of accuracy and speed
4. ✅ Widely used in research (easier to debug)

**Alternative:** U-Net if DeepLabV3+ has issues
- Simpler architecture
- Easier to implement from scratch
- Still competitive performance

#### Training Strategy
```python
# Pseudocode for baseline training
model = deeplabv3_resnet50(pretrained=True)
model.classifier[-1] = nn.Conv2d(256, 13, 1)  # 13 classes (0-12, no anomaly yet)

optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, patience=3)
loss_fn = CrossEntropyLoss(ignore_index=13)  # Ignore anomaly class in training

for epoch in range(30):
    train_one_epoch(model, train_loader, optimizer, loss_fn)
    miou = validate(model, val_loader)
    scheduler.step(miou)
    save_best_model(model, miou)
```

**Expected Results:**
- Validation mIoU: 40-50% (baseline)
- Training time: ~2-3 hours (depending on GPU)

### Phase 4: Simple Anomaly Detection (8 hours)

#### Method: Standardized Max Logits
**Reasoning:**
1. ✅ No retraining needed (use trained baseline)
2. ✅ Simple to implement (~20 lines)
3. ✅ SOTA among training-free methods
4. ✅ Fast inference

**Implementation:**
```python
def detect_anomalies(model, image):
    logits = model(image)  # Shape: [B, 13, H, W]

    # Step 1: Get max logits and predicted classes
    max_logits, pred_classes = logits.max(dim=1)  # [B, H, W]

    # Step 2: Compute per-class statistics (on validation set)
    class_means = {}  # μ_c for each class c
    class_stds = {}   # σ_c for each class c

    for c in range(13):
        mask = (pred_classes == c)
        class_means[c] = max_logits[mask].mean()
        class_stds[c] = max_logits[mask].std()

    # Step 3: Standardize max logits
    sml = torch.zeros_like(max_logits)
    for c in range(13):
        mask = (pred_classes == c)
        sml[mask] = (max_logits[mask] - class_means[c]) / class_stds[c]

    # Step 4: Anomaly score (lower SML = higher anomaly)
    anomaly_score = -sml

    return anomaly_score, pred_classes
```

**Expected Results:**
- Test AUPR: 15-25%
- No additional training time

### Phase 5: Advanced Method (10 hours)

#### Method: Metric Learning (DMLNet-inspired)
**Reasoning:**
1. ✅ Best performance (SOTA)
2. ✅ Principled approach (embedding space)
3. ✅ Potential +3-5% AUPR improvement

**Trade-offs:**
- ⚠️ Requires retraining with contrastive loss
- ⚠️ More complex implementation
- ⚠️ Longer training time

**Decision Point:**
- If time permits: Implement metric learning
- If running short: Stick with Standardized Max Logits + ablations

### Phase 6: Ablation Studies (8 hours)

#### Key Questions to Answer
1. **Backbone:** ResNet50 vs ResNet101?
2. **Pretraining:** ImageNet vs random init?
3. **Data augmentation:** With vs without?
4. **Anomaly threshold:** How to set optimally?
5. **Temperature scaling:** Does calibration help?

---

## Summary Table

| Paper | Type | Key Contribution | Relevance | Implementation Priority |
|-------|------|------------------|-----------|------------------------|
| DeepLabV3+ | Architecture | Encoder-decoder + ASPP | High - baseline model | 🔴 Phase 2 |
| U-Net | Architecture | Skip connections, limited data | Medium - alternative | ⚪ Backup |
| Standardized Max Logits | Anomaly | Class-normalized confidence | High - simple & effective | 🟡 Phase 4 |
| Scaling OOD (StreetHazards) | Dataset | Our benchmark dataset | Critical - evaluation | 🟢 Phase 1 (done) |
| Deep Metric Learning | Anomaly | Embedding space approach | High - advanced method | 🟠 Phase 5 |

---

## References

1. Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. In ECCV.

2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In MICCAI.

3. Jung, S., Lee, J., Gwak, D., Choi, S., & Choo, J. (2021). Standardized max logits: A simple yet effective approach for identifying unexpected road obstacles in urban-scene segmentation. In ICCV.

4. Hendrycks, D., Basart, S., Mazeika, M., Zou, A., Kwon, J., Mostajabi, M., ... & Song, D. (2022). Scaling out-of-distribution detection for real-world settings. In ICML.

5. Cen, J., Yun, P., Cai, J., Wang, M. Y., & Ming, L. (2021). Deep metric learning for open world semantic segmentation. In ICCV.

---

*Research compiled: 2025-11-04*
*Next update: After Phase 5 implementation*
