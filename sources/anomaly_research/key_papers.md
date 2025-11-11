# Key Papers for Anomaly Detection in Semantic Segmentation

This document provides direct links and summaries of the most important papers referenced in the research.

---

## Essential Papers (Top Priority)

### 1. ATTA: Anomaly-aware Test-Time Adaptation
**Authors:** Zhitong Gao, Shipeng Yan, Xuming He
**Venue:** NeurIPS 2023
**Paper:** https://arxiv.org/abs/2309.05994
**Code:** https://github.com/gaozhitong/ATTA

**Key Contribution:** Addresses domain shift and semantic shift jointly through test-time adaptation with batch normalization updates and anomaly-aware self-training.

**Why Important:** Directly addresses our domain shift problem where SML failed.

---

### 2. Residual Pattern Learning (RPL)
**Authors:** Yuyuan Liu et al.
**Venue:** ICCV 2023
**Paper:** https://arxiv.org/abs/2211.14512
**Code:** https://github.com/yyliu01/RPL

**Key Contribution:** Learns residual patterns between normal and anomalous pixels with context-robust contrastive learning.

**Performance:** ~10% FPR improvement, ~7% AUPR improvement on Fishyscapes, SMIYC, RoadAnomaly.

---

### 3. Meta-OOD: Entropy Maximization and Meta Classification
**Authors:** Robin Chan, Matthias Rottmann, Hanno Gottschalk
**Venue:** ICCV 2021
**Paper:** https://arxiv.org/abs/2012.06575
**Code:** https://github.com/robin-chan/meta-ood

**Key Contribution:** Two-stage approach using entropy maximization on OOD proxy data + meta-classifier on hand-crafted metrics.

**Performance:** 52% reduction in detection errors.

---

### 4. VOS: Virtual Outlier Synthesis
**Authors:** Xuefeng Du et al.
**Venue:** ICLR 2022
**Paper:** https://arxiv.org/abs/2202.01197
**Code:** https://github.com/deeplearning-wisc/vos

**Key Contribution:** Synthesizes virtual outliers from low-likelihood regions in feature space, no real OOD data needed.

**Performance:** 9.36% FPR95 reduction.

---

### 5. SegmentMeIfYouCan Benchmark
**Authors:** Hermann Blum et al.
**Venue:** NeurIPS 2021
**Paper:** https://arxiv.org/abs/2104.14812
**Website:** https://segmentmeifyoucan.com/
**Code:** https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark

**Key Contribution:** Standardized benchmark for anomaly segmentation in driving scenarios with 100 annotated images.

---

### 6. LogitNorm: Mitigating Neural Network Overconfidence
**Authors:** Hongxin Wei et al.
**Venue:** ICML 2022
**Paper:** https://arxiv.org/abs/2205.09310
**Code:** https://github.com/hongxin001/logitnorm_ood

**Key Contribution:** Enforces constant norm on logit vectors to reduce overconfidence.

**Performance:** 42.30% average FPR95 reduction.

---

## Important Baseline Papers

### 7. Maximum Softmax Probability (MSP)
**Authors:** Dan Hendrycks, Kevin Gimpel
**Venue:** ICLR 2017
**Paper:** https://arxiv.org/abs/1610.02136

**Key Contribution:** Original baseline for OOD detection using maximum softmax probability.

---

### 8. Energy-based OOD Detection
**Authors:** Weitang Liu et al.
**Venue:** NeurIPS 2020
**Paper:** https://arxiv.org/abs/2010.03759

**Key Contribution:** Uses logsumexp of logits as energy score for OOD detection.

---

### 9. Standardized Max Logits (SML)
**Authors:** Sanghun Jung et al.
**Venue:** ICCV 2021
**Paper:** (Search on IEEE Xplore or conference proceedings)

**Key Contribution:** Normalizes max logits using class-wise statistics for consistent thresholds.

**Our Finding:** Fails under domain shift (3.70% vs 6.19% AUPR).

---

### 10. ODIN: Enhanced OOD Detection
**Authors:** Shiyu Liang et al.
**Venue:** ICLR 2018
**Paper:** https://arxiv.org/abs/1706.02690

**Key Contribution:** Temperature scaling and input perturbation to improve OOD detection.

---

## Feature Space Methods

### 11. Mahalanobis Distance for OOD Detection
**Authors:** Kimin Lee et al.
**Venue:** NeurIPS 2018
**Paper:** https://arxiv.org/abs/1807.03888

**Key Contribution:** Uses Mahalanobis distance in feature space with class-conditional Gaussian assumption.

---

### 12. Prototypical Residual Networks (PRNet)
**Authors:** Various
**Venue:** CVPR 2023
**Paper:** https://arxiv.org/abs/2212.02031

**Key Contribution:** Multi-scale prototypes for anomaly detection and localization.

---

## Advanced Methods (2024-2025)

### 13. Diffusion Models Survey for Anomaly Detection
**Authors:** Various
**Year:** 2025
**Paper:** https://arxiv.org/abs/2501.11430

**Key Contribution:** Comprehensive survey of diffusion models applied to anomaly detection.

---

### 14. MC-Frequency Dropout
**Authors:** Various
**Year:** 2025
**Paper:** https://arxiv.org/abs/2501.11258

**Key Contribution:** Extends dropout to frequency domain for better uncertainty estimation in segmentation.

---

### 15. AnomalyControl: Cross-modal Anomaly Synthesis
**Authors:** Various
**Year:** 2024
**Paper:** https://arxiv.org/abs/2412.06510

**Key Contribution:** Uses cross-modal semantic features for controllable anomaly generation.

---

### 16. PEBAL: Pixel-wise Energy-Biased Abstention Learning
**Authors:** Yu Tian et al.
**Venue:** ECCV 2022
**Paper:** (Check ECCV proceedings or author pages)

**Key Contribution:** Energy-based abstention learning for road anomaly detection.

**Performance:** Best FPR95 on Lost and Found, only 0.4% AUC drop under domain shift.

---

## Ensemble & Uncertainty Methods

### 17. Deep Ensembles for Uncertainty
**Authors:** Balaji Lakshminarayanan et al.
**Venue:** NeurIPS 2017
**Paper:** https://arxiv.org/abs/1612.01474

**Key Contribution:** Simple and scalable uncertainty estimation using ensembles.

---

### 18. Scalable Ensemble Diversification (SED)
**Authors:** Various
**Year:** 2024
**Paper:** https://arxiv.org/abs/2409.16797

**Key Contribution:** Training diverse ensembles for better OOD detection without OOD samples.

---

### 19. MC Dropout (Bayesian Deep Learning)
**Authors:** Yarin Gal, Zoubin Ghahramani
**Venue:** ICML 2016
**Paper:** https://arxiv.org/abs/1506.02142

**Key Contribution:** Dropout as Bayesian approximation for uncertainty estimation.

---

## Synthetic Anomaly Generation

### 20. Few-Shot Anomaly Generation (AnoGen)
**Venue:** ECCV 2024
**Paper:** (Check ECCV 2024 proceedings)

**Key Contribution:** Generates realistic anomalies with few examples using diffusion models.

**Performance:** 5.8% AUPR improvement with DRAEM.

---

### 21. Anomaly-Aware Semantic Segmentation
**Authors:** Various
**Year:** 2021
**Paper:** https://arxiv.org/abs/2111.14343

**Key Contribution:** Leveraging synthetic-unknown data for anomaly-aware segmentation.

---

## Benchmark & Datasets

### 22. StreetHazards Dataset
**Authors:** Dan Hendrycks et al.
**Year:** 2021
**Paper:** https://arxiv.org/abs/1911.11132
**Website:** https://github.com/hendrycks/anomaly-seg

**Key Contribution:** Synthetic dataset with diverse anomalous objects in driving scenes.

---

### 23. Lost and Found Dataset
**Website:** http://lmb.informatik.uni-freiburg.de/resources/datasets/laf.en.html

**Key Contribution:** Real-world dataset of obstacles and hazards on roads.

---

### 24. Fishyscapes
**Website:** https://fishyscapes.com/

**Key Contribution:** Benchmark for pixel-level OOD detection in autonomous driving.

---

## Surveys & Reviews

### 25. Generalized OOD Detection Survey
**Authors:** Jingkang Yang et al.
**Year:** 2021
**Paper:** https://arxiv.org/abs/2110.11334

**Key Contribution:** Comprehensive survey of OOD detection methods and taxonomy.

---

### 26. Recent Advances in OOD Detection
**Year:** 2024
**Paper:** https://arxiv.org/abs/2409.11884

**Key Contribution:** Recent advances covering 2020-2024 methods and approaches.

---

### 27. Anomaly Detection in Deep Learning (2019-2023)
**Authors:** Kumari et al.
**Venue:** IET Information Security 2024

**Key Contribution:** Comprehensive investigation of anomaly detection methods in ML/DL.

---

## Additional Important Papers

### 28. Normalizing Flows for OOD Detection
**Year:** 2024
**Paper:** https://arxiv.org/abs/2402.06537

**Key Contribution:** Feature density estimation via normalizing flows without OOD data.

---

### 29. Balanced Energy Regularization
**Venue:** CVPR 2023
**Paper:** (Check CVPR 2023 proceedings)

**Key Contribution:** Balanced energy loss for improved OOD detection in segmentation.

---

### 30. ReAct: Rectified Activations
**Authors:** Yiyou Sun et al.
**Venue:** NeurIPS 2021
**Paper:** (Check NeurIPS proceedings)

**Key Contribution:** Activation clipping to enhance energy-based OOD detection.

---

## How to Access Papers

### ArXiv Papers (Free Access)
Most papers listed above are available on arXiv. Simply visit the provided links.

### Conference Papers
Papers from ICCV, CVPR, ECCV, NeurIPS are available on:
- **CVPR/ICCV:** https://openaccess.thecvf.com/
- **NeurIPS:** https://proceedings.neurips.cc/
- **ECCV:** SpringerLink (may require institutional access)

### Code Repositories
Most recent papers provide GitHub repositories with implementations. Links are provided where available.

---

## Reading Priority

**For Our StreetHazards Problem:**

**High Priority (Read First):**
1. ATTA (addresses domain shift directly)
2. PEBAL (domain-robust energy method)
3. Meta-OOD (practical two-stage approach)
4. Energy-based OOD (foundational)
5. SegmentMeIfYouCan (benchmark understanding)

**Medium Priority:**
6. RPL (state-of-the-art method)
7. VOS (synthetic outlier generation)
8. LogitNorm (overconfidence mitigation)
9. Mahalanobis Distance (feature-space baseline)

**Lower Priority (Background):**
10. MSP, ODIN (historical baselines)
11. Surveys (comprehensive understanding)
12. Diffusion models (emerging trend)

---

**Last Updated:** November 6, 2025
