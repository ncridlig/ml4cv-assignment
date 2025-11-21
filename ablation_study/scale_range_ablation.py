#!/usr/bin/env python3
"""
Scale Range Ablation Study

Tests different scale_range values for multi-scale augmentation to find optimal range
for both closed-set segmentation and anomaly detection.

Reuses training infrastructure from augmentation_ablation.py.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

from ablation_study.augmentation_ablation import train_config


# Scale range configurations organized by category
SCALE_CONFIGS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Category A: Narrow ranges (less augmentation)
    'ScaleRange_0.9_1.1': {
        'scale': True,
        'scale_range': (0.9, 1.1),
    },
    'ScaleRange_0.75_1.25': {
        'scale': True,
        'scale_range': (0.75, 1.25),
    },

    # Category B: Current baseline (DeepLabV3+ paper default)
    'ScaleRange_0.5_2.0': {
        'scale': True,
        'scale_range': (0.5, 2.0),
    },

    # Category C: Wide ranges (more augmentation)
    'ScaleRange_0.4_2.5': {
        'scale': True,
        'scale_range': (0.4, 2.5),
    },
    'ScaleRange_0.3_3.0': {
        'scale': True,
        'scale_range': (0.3, 3.0),
    },

    # Category D: Asymmetric ranges
    'ScaleRange_0.5_1.5': {
        'scale': True,
        'scale_range': (0.5, 1.5),
    },  # Zoom-in emphasis (fine details)
    'ScaleRange_0.7_2.0': {
        'scale': True,
        'scale_range': (0.7, 2.0),
    },  # Zoom-out emphasis (context)
}


def print_summary(results: Dict) -> None:
    """Print formatted summary table of results."""
    print("\n" + "="*90)
    print("SCALE RANGE ABLATION STUDY SUMMARY")
    print("="*90)
    print(f"\n{'Configuration':<25} {'Scale Range':<15} {'Val mIoU':<12} "
          f"{'Best Epoch':<12} {'Stopped Early'}")
    print("-"*90)

    # Sort by val mIoU descending
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['best_val_miou'],
        reverse=True
    )

    for name, result in sorted_results:
        scale_range = SCALE_CONFIGS[name]['scale_range']
        stopped = "Yes" if result['stopped_early'] else "No"
        print(f"{name:<25} {str(scale_range):<15} "
              f"{result['best_val_miou']:.4f}       "
              f"{result['best_epoch']:<12} {stopped}")

    print("="*90)


def save_results(results: Dict, results_path: Path) -> None:
    """Save results to JSON file."""
    # Add scale_range to each result for convenience
    for name, result in results.items():
        result['scale_range'] = SCALE_CONFIGS[name]['scale_range']

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {results_path}")


def main():
    """Run scale range ablation study."""
    print("\n" + "="*90)
    print("SCALE RANGE ABLATION STUDY")
    print("="*90)
    print(f"\nTesting {len(SCALE_CONFIGS)} scale range configurations")
    print("Training: 20 epochs max, early stopping patience=3")
    print("Augmentation: Multi-scale only (no flip/color/rotate)")
    print("="*90)

    # Setup output directories
    results_dir = Path('ablation_study/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'scale_range_results.json'

    # Train each configuration
    results = {}
    for i, (config_name, aug_config) in enumerate(SCALE_CONFIGS.items(), 1):
        print(f"\n[{i}/{len(SCALE_CONFIGS)}] Configuration: {config_name}")
        print(f"Scale Range: {aug_config['scale_range']}")

        result = train_config(
            aug_config=aug_config,
            config_name=config_name,
            max_epochs=20,
            patience=3
        )
        results[config_name] = result

        # Save incremental results after each training
        save_results(results, results_path)

        print(f"✓ {config_name} complete: {result['best_val_miou']:.4f} mIoU "
              f"(epoch {result['best_epoch']})")

    # Print final summary
    print_summary(results)

    # Print next steps
    print("\n" + "="*90)
    print("NEXT STEPS")
    print("="*90)
    print("\n1. Update visualizations/create_comparison_table.py:")
    print("   - Add SCALE_RANGE_MODELS dictionary with checkpoint paths")
    print("   - Use scale_range_results.json to get mIoU values")
    print("\n2. Run anomaly detection evaluation:")
    print("   .venv/bin/python3 visualizations/create_comparison_table.py")
    print("\n3. Generate comparison plots (optional):")
    print("   .venv/bin/python3 visualizations/create_comparison_plots.py")
    print("\n4. Analyze results and update log.md")
    print("="*90)


if __name__ == '__main__':
    main()
