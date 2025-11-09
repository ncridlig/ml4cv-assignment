# Multi-Scale Augmentation Implementation

## Summary of Changes

### Problem Identified
All previous models showed severe overfitting:
- SegFormer-B5: Train 65% IoU → Val 31% mIoU (34% gap)
- Hiera-Base: Train 65% IoU → Val 28% mIoU (37% gap)
- Previous augmentation was too weak (only flip + basic color jitter)

### Solution: Multi-Scale Training + Stronger Augmentation

Based on DeepLabV3+ paper and semantic segmentation best practices.

## What Changed

### 1. `dataloader.py` - New Augmentation Pipeline

**New Custom Transform Classes:**
- `JointRandomHorizontalFlip`: Synchronized flip for image and mask
- `JointRandomRotation`: ±10° rotation for both image and mask
- `JointRandomScaleCrop`: **Multi-scale training (0.5-2.0x)** - KEY IMPROVEMENT

**Augmentation Pipeline (Training Only):**
1. **Multi-scale random crop with variable crop sizes** (0.5-2.0x scale range)
   - Following DeepLabV3+ paper literally
   - Scale 0.5x: crop 256×256 → resize to 512×512 (zooms in, fine details)
   - Scale 1.0x: crop 512×512 → resize to 512×512 (normal view)
   - Scale 2.0x: crop 1024×1024 → resize to 512×512 (zooms out, context)
   - **No padding needed** - crop size adapts to scale!
   - Model sees objects at different scales
   - Reduces overfitting dramatically
2. **Random rotation** (±10°)
   - Roads can appear at slight angles
3. **Random horizontal flip** (50%)
   - Standard augmentation
4. **Color jitter** (brightness, contrast, saturation, **+ hue**)
   - Added hue variation (0.1) for more diversity
5. **Gaussian blur** (50% probability)
   - Simulates motion/focus blur in driving

**Validation/Test:** No augmentation (same as before)

### 2. `config.py` - Training Hyperparameters

**Changed:**
- `EPOCHS`: 15 → **40** (stronger augmentation needs more epochs)
- `BATCH_SIZE`: Already at 4 (good)
- Other params unchanged

### 3. `train_augmented_resnet50.py` - New Training Script

**Features:**
- Uses DeepLabV3+ ResNet50 (best baseline model)
- Comprehensive logging with training history
- Automatic comparison with baseline (37.57%)
- Saves detailed summary to `assets/resnet50_augmented_training_summary.txt`

## Expected Results

### Baseline (Previous Best)
- **Model**: DeepLabV3+ ResNet50 @ 512×512
- **Augmentation**: Weak (flip + color jitter)
- **Result**: 37.57% mIoU

### Target (This Run)
- **Model**: DeepLabV3+ ResNet50 @ 512×512
- **Augmentation**: Strong (multi-scale + rotation + blur)
- **Expected**: **40-42% mIoU** (+2-5% improvement)

### Why This Will Work

1. **Addresses overfitting directly** - more diverse training samples
2. **Proven technique** - DeepLabV3+ paper uses random scaling 0.5-2.0×
3. **Only major untried improvement** - we tested bigger models ❌, higher resolution ❌, transformers ❌
4. **Literature support** - multi-scale training is standard in semantic segmentation

## How to Run

### Step 1: Test Augmentations (Recommended)
```bash
.venv/bin/python3 test_augmented_dataloader.py
```
- Loads 3 samples with different augmentations
- Saves visualization to `assets/augmented_dataloader_test.png`
- Verifies no errors in dataloader

### Step 2: Start Training
```bash
.venv/bin/python3 train_augmented_resnet50.py
```

**Training Details:**
- Duration: ~4-5 hours (40 epochs × 7 min/epoch)
- GPU Usage: Same as before (~6-8GB VRAM)
- Outputs:
  - Best model: `models/deeplabv3_resnet50_augmented_*.pth`
  - TensorBoard: `runs/resnet50_augmented/`
  - Summary: `assets/resnet50_augmented_training_summary.txt`

### Step 3: Monitor Training
```bash
tensorboard --logdir runs/resnet50_augmented/
```
- Open http://localhost:6006
- Watch train/val IoU curves
- Check if train-val gap is smaller (sign of reduced overfitting)

## What to Watch For

### Good Signs (Expected)
- ✅ Train IoU increases slower than before (augmentation working)
- ✅ Validation mIoU is more stable
- ✅ Smaller train-val gap (< 15% gap is good)
- ✅ Best model around epoch 20-30

### Warning Signs (Unlikely)
- ⚠️ Validation mIoU plateaus below 38% → augmentation might be too strong
- ⚠️ Training very slow (>10 min/epoch) → dataloader issue
- ⚠️ Crash/errors → run test script first

## Files Modified/Created

### Modified
- `dataloader.py` - Added joint transforms and augmentation pipeline
- `config.py` - Increased EPOCHS to 40

### Created
- `train_augmented_resnet50.py` - New training script with enhanced logging
- `test_augmented_dataloader.py` - Verification script
- `AUGMENTATION_CHANGES.md` - This file

## Rollback (If Needed)

If results are worse than baseline:
1. Revert to weak augmentation by changing dataloader.py line 139:
   ```python
   scale_crop = JointRandomScaleCrop(target_size=IMAGE_SIZE, scale_range=(0.8, 1.2))
   # Reduced from (0.5, 2.0) to (0.8, 1.2)
   ```
2. Or disable specific augmentations by commenting them out in `dataloader.py`

## Next Steps After Training

1. Compare results with baseline
2. If improvement ≥ 2%: Use this model for final anomaly detection
3. If improvement < 2%: Stick with baseline (37.57%)
4. Document results in log.md
5. Focus remaining time on ablation studies

---

**Good luck with training! Expected improvement: +2-5% mIoU** 🚀
