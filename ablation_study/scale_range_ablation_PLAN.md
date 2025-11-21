# Scale Range Ablation Study Plan (Revised)

## Objective
Systematically evaluate the impact of different multi-scale augmentation ranges on model performance by training models with various `scale_range` parameters.

## Motivation
Our augmentation ablation study showed that multi-scale augmentation is the single most important factor (+12.69% mIoU). The current implementation uses `scale_range=(0.5, 2.0)` following the DeepLabV3+ paper, but we want to verify if this is optimal for anomaly detection in street scenes.

**Key Questions:**
1. Is (0.5, 2.0) optimal or can we do better?
2. Do narrower ranges improve training efficiency?
3. Do wider ranges improve robustness?
4. Are asymmetric ranges beneficial for anomaly detection?

---

## Implementation: Single Minimal Script

### File: `ablation_study/scale_range_ablation.py`

**Strategy:** Keep it simple - reuse existing `train_config()` function from `augmentation_ablation.py`, just modify the scale_range parameter.

**Key Changes from `augmentation_ablation.py`:**
1. Define scale range configurations instead of augmentation type configurations
2. Pass `scale_range` parameter in aug_config
3. Update checkpoint naming to reflect scale range
4. Everything else stays the same (training loop, early stopping, etc.)

**Pseudo-code:**
```python
#!/usr/bin/env python3
"""
Scale Range Ablation Study
Tests different scale_range values for multi-scale augmentation
"""

from ablation_study.augmentation_ablation import train_config
import json
from pathlib import Path

def main():
    """Run scale range ablation study"""

    # Define scale range configurations to test
    configs = {
        # Narrow ranges (less augmentation)
        'ScaleRange_0.9_1.1': {
            'scale': True,
            'scale_range': (0.9, 1.1)
        },
        'ScaleRange_0.75_1.25': {
            'scale': True,
            'scale_range': (0.75, 1.25)
        },

        # Current baseline
        'ScaleRange_0.5_2.0': {
            'scale': True,
            'scale_range': (0.5, 2.0)
        },

        # Wide ranges (more augmentation)
        'ScaleRange_0.4_2.5': {
            'scale': True,
            'scale_range': (0.4, 2.5)
        },
        'ScaleRange_0.3_3.0': {
            'scale': True,
            'scale_range': (0.3, 3.0)
        },

        # Asymmetric ranges
        'ScaleRange_0.5_1.5': {
            'scale': True,
            'scale_range': (0.5, 1.5)
        },  # Zoom-in emphasis
        'ScaleRange_0.7_2.0': {
            'scale': True,
            'scale_range': (0.7, 2.0)
        },  # Zoom-out emphasis
    }

    results = {}
    Path('ablation_study/results').mkdir(parents=True, exist_ok=True)

    # Train each configuration
    for name, aug_config in configs.items():
        print(f"\n{'='*80}")
        print(f"Configuration: {name}")
        print(f"Scale Range: {aug_config['scale_range']}")
        print(f"{'='*80}")

        result = train_config(
            aug_config=aug_config,
            config_name=name,
            max_epochs=20,  # Reduced from 40 for efficiency
            patience=3
        )
        results[name] = result

        # Save incremental results
        with open('ablation_study/results/scale_range_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("SCALE RANGE ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"\n{'Configuration':<25} {'Scale Range':<15} {'Val mIoU':<12} {'Best Epoch'}")
    print("-"*80)
    for name, result in sorted(results.items(),
                               key=lambda x: x[1]['best_val_miou'],
                               reverse=True):
        scale_range = configs[name]['scale_range']
        print(f"{name:<25} {str(scale_range):<15} "
              f"{result['best_val_miou']:.4f}       {result['best_epoch']}")

    print(f"\n✅ Results saved to ablation_study/results/scale_range_results.json")

if __name__ == '__main__':
    main()
```

**That's it!** This is the only new file needed. Everything else uses existing infrastructure.

---

## Required Code Changes

### 1. Update `utils/dataloader.py` - Add scale_range parameter support

**Current code (line ~143-144):**
```python
scale_crop = JointRandomScaleCrop(output_size=self.image_size, scale_range=(0.5, 2.0), base_crop_size=512)
```

**Change to:**
```python
# Get scale_range from aug_config, default to (0.5, 2.0)
scale_range = aug_config.get('scale_range', (0.5, 2.0))
scale_crop = JointRandomScaleCrop(output_size=self.image_size, scale_range=scale_range, base_crop_size=512)
```

This allows passing custom scale_range through aug_config.

---

## Evaluation (Use Existing Tools)

### After Training Completes:

**Step 1: Update `visualizations/create_comparison_table.py`**

Add a new dictionary `SCALE_RANGE_MODELS` similar to `ABLATION_MODELS`:

```python
SCALE_RANGE_MODELS = {
    'Scale 0.9-1.1': {
        'path': 'ablation_study/checkpoints/ScaleRange_0.9_1.1_*.pth',
        'architecture': 'deeplabv3_resnet50',
        'miou': <from results.json>,
        'image_size': (512, 512),
    },
    'Scale 0.75-1.25': {
        'path': 'ablation_study/checkpoints/ScaleRange_0.75_1.25_*.pth',
        ...
    },
    # ... etc
}
```

**Step 2: Run anomaly detection evaluation:**
```bash
# Modify create_comparison_table.py to use SCALE_RANGE_MODELS instead of DEV_MODELS
.venv/bin/python3 visualizations/create_comparison_table.py
```

This generates:
- `assets/scale_range_method_comparison.md` (results table)
- `assets/model_method_comparison.json` (JSON results)

**Step 3: Visualize qualitative results (optional):**
```bash
.venv/bin/python3 evaluate_qualitative.py --model <path> --split test
```

**Step 4: Create comparison plots (optional):**
```bash
.venv/bin/python3 visualizations/create_comparison_plots.py
```

---

## Scale Range Configurations to Test

### Category A: Narrow Ranges (Less Augmentation)
1. **(0.9, 1.1)** - Minimal: ±10% scale variation
2. **(0.75, 1.25)** - Conservative: ±25% scale variation

### Category B: Current Baseline
3. **(0.5, 2.0)** - DeepLabV3+ paper default (current best)

### Category C: Wide Ranges (More Augmentation)
4. **(0.4, 2.5)** - Extended: 0.4× to 2.5×
5. **(0.3, 3.0)** - Aggressive: 0.3× to 3×

### Category D: Asymmetric Ranges
6. **(0.5, 1.5)** - Zoom-in emphasis (fine details)
7. **(0.7, 2.0)** - Zoom-out emphasis (context)

**Total: 7 configurations** (reduced from 9 for efficiency)

---

## Computational Budget

**Per Configuration:**
- Training: ~12 epochs average (early stopping) × 6 min/epoch = ~1.2 hours
- Anomaly evaluation: ~10 minutes (via create_comparison_table.py)

**Total Estimate:**
- 7 configurations × 1.2 hours = **~8.5 hours training**
- Evaluation: **~1 hour** (one batch run)
- **Total: ~9.5 hours on RTX 4080 Super**

---

## Expected Outputs

### 1. Training Phase
- `ablation_study/results/scale_range_results.json` - Training results (mIoU, epochs)
- `ablation_study/checkpoints/ScaleRange_X.X_Y.Y_*.pth` - Model checkpoints (7 files)

### 2. Evaluation Phase (using existing tools)
- `assets/scale_range_method_comparison.md` - Formatted comparison table
- `assets/scale_range_method_comparison.json` - JSON with all metrics

### 3. Analysis Phase
- Add section to `log.md` documenting findings
- Add section to `main.ipynb` (optional, if findings are significant)

---

## Hypotheses to Test

1. **H1: Current (0.5, 2.0) is near-optimal**
   - Expect narrow ranges (0.9-1.1) to underperform
   - Expect very wide ranges (0.3-3.0) to be noisy

2. **H2: Wider ranges help anomaly detection more than segmentation**
   - Anomaly detection benefits from scale robustness
   - Segmentation benefits from scale diversity but with diminishing returns

3. **H3: Asymmetric ranges may offer specialized trade-offs**
   - Zoom-in (0.5-1.5): Better small object segmentation, worse anomaly detection
   - Zoom-out (0.7-2.0): Better context/anomalies, worse fine details

---

## Questions for User

1. **Training Duration:**
   - 20 epochs max, early stopping patience=3 OK?
   - This averages to ~12 epochs per config

2. **Configuration Selection:**
   - Are these 7 configurations sufficient?
   - Want to add/remove any specific ranges?

3. **Evaluation Scope:**
   - Run Simple Max Logits only, or also HEAT?
   - Evaluate on test set or validation set only?

4. **Baseline:**
   - Should we re-train (0.5, 2.0) baseline or use existing checkpoint?
   - Existing: `ablation_study/checkpoints/+Scale__20_52_19-11-25_mIoU_0.5176_size_512x512.pth`

---

## Next Steps

Once approved:

1. **Implement** (10 minutes):
   - [ ] Update `utils/dataloader.py` to accept `scale_range` in `aug_config`
   - [ ] Create `ablation_study/scale_range_ablation.py` (~50 lines)

2. **Run Training** (~8.5 hours):
   - [ ] Execute: `.venv/bin/python3 ablation_study/scale_range_ablation.py`
   - [ ] Monitor training progress

3. **Evaluation** (~1 hour):
   - [ ] Update `visualizations/create_comparison_table.py` with `SCALE_RANGE_MODELS`
   - [ ] Run anomaly detection evaluation
   - [ ] Generate comparison tables/plots

4. **Analysis** (30 minutes):
   - [ ] Analyze results
   - [ ] Update `log.md` with findings
   - [ ] Decide if notebook update is warranted

**Total Time: ~10 hours** (mostly unattended GPU training)
