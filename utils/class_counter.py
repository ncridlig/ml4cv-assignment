"""
Utility script to count class distribution across train, validation, and test splits.
Analyzes pixel counts for each class in the StreetHazards dataset.
"""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import dataloader
sys.path.append(str(Path(__file__).parent.parent))

from dataloader import (
    StreetHazardsDataset,
    get_transforms,
    CLASS_NAMES,
    NUM_CLASSES,
    ANOMALY_CLASS_IDX
)


def count_class_pixels(dataset, dataset_name="Dataset"):
    """
    Count the number of pixels for each class in a dataset.

    Args:
        dataset: StreetHazardsDataset instance
        dataset_name: Name of the dataset for display

    Returns:
        class_counts: numpy array of pixel counts per class
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name} ({len(dataset)} images)")
    print(f"{'='*60}")

    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_pixels = 0

    # Iterate through all images with progress bar
    for idx in tqdm(range(len(dataset)), desc=f"Processing {dataset_name}"):
        _, mask, _ = dataset.get_raw_item(idx)

        # Count pixels for each class
        unique, counts = np.unique(mask, return_counts=True)
        for class_id, count in zip(unique, counts):
            if 0 <= class_id < NUM_CLASSES:
                class_counts[class_id] += count
                total_pixels += count

    return class_counts, total_pixels


def print_class_statistics(class_counts, total_pixels, dataset_name):
    """
    Print formatted statistics for class distribution.

    Args:
        class_counts: Array of pixel counts per class
        total_pixels: Total number of pixels
        dataset_name: Name of the dataset
    """
    print(f"\n{dataset_name} Class Distribution:")
    print(f"{'='*80}")
    print(f"{'Class ID':<10} {'Class Name':<20} {'Pixel Count':<15} {'Percentage':<10}")
    print(f"{'-'*80}")

    for class_id, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
        if count > 0:
            percentage = (count / total_pixels) * 100
            print(f"{class_id:<10} {name:<20} {count:<15,} {percentage:>6.2f}%")

    print(f"{'-'*80}")
    print(f"{'TOTAL':<10} {'':<20} {total_pixels:<15,} {100.0:>6.2f}%")
    print(f"{'='*80}")

    # Print summary statistics
    non_zero_classes = np.sum(class_counts > 0)
    print(f"\nSummary:")
    print(f"  - Classes present: {non_zero_classes}/{NUM_CLASSES}")
    print(f"  - Total pixels: {total_pixels:,}")
    print(f"  - Average pixels per class: {total_pixels / non_zero_classes:,.0f}")


def compare_splits(train_counts, val_counts, test_counts):
    """
    Compare class distributions across splits.

    Args:
        train_counts: Pixel counts for training set
        val_counts: Pixel counts for validation set
        test_counts: Pixel counts for test set
    """
    print(f"\n{'='*80}")
    print("Class Presence Comparison Across Splits")
    print(f"{'='*80}")
    print(f"{'Class ID':<10} {'Class Name':<20} {'Train':<10} {'Val':<10} {'Test':<10}")
    print(f"{'-'*80}")

    for class_id, name in enumerate(CLASS_NAMES):
        train_present = "✓" if train_counts[class_id] > 0 else "✗"
        val_present = "✓" if val_counts[class_id] > 0 else "✗"
        test_present = "✓" if test_counts[class_id] > 0 else "✗"

        # Highlight anomaly class
        marker = " ← ANOMALY (test only)" if class_id == ANOMALY_CLASS_IDX else ""

        print(f"{class_id:<10} {name:<20} {train_present:<10} {val_present:<10} {test_present:<10}{marker}")

    print(f"{'='*80}")


def main():
    """Main function to analyze all dataset splits."""
    print("=" * 80)
    print("StreetHazards Dataset Class Counter")
    print("=" * 80)

    # Create datasets (no transforms needed for counting)
    _, mask_transform = get_transforms(512, is_training=False)

    print("\nLoading datasets...")

    train_dataset = StreetHazardsDataset(
        root_dir='streethazards_train/train',
        split='training',
        transform=None,
        mask_transform=None
    )

    val_dataset = StreetHazardsDataset(
        root_dir='streethazards_train/train',
        split='validation',
        transform=None,
        mask_transform=None
    )

    test_dataset = StreetHazardsDataset(
        root_dir='streethazards_test/test',
        split='test',
        transform=None,
        mask_transform=None
    )

    # Count pixels for each split
    train_counts, train_total = count_class_pixels(train_dataset, "Training Set")
    val_counts, val_total = count_class_pixels(val_dataset, "Validation Set")
    test_counts, test_total = count_class_pixels(test_dataset, "Test Set")

    # Print statistics for each split
    print_class_statistics(train_counts, train_total, "Training Set")
    print_class_statistics(val_counts, val_total, "Validation Set")
    print_class_statistics(test_counts, test_total, "Test Set")

    # Compare splits
    compare_splits(train_counts, val_counts, test_counts)

    # Save results to file
    output_file = 'assets/class_distribution.txt'
    os.makedirs('assets', exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("StreetHazards Dataset Class Distribution\n")
        f.write("=" * 80 + "\n\n")

        for split_name, counts, total in [
            ("Training", train_counts, train_total),
            ("Validation", val_counts, val_total),
            ("Test", test_counts, test_total)
        ]:
            f.write(f"\n{split_name} Set:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Class ID':<10} {'Class Name':<20} {'Pixel Count':<15} {'Percentage':<10}\n")
            f.write("-" * 80 + "\n")

            for class_id, (name, count) in enumerate(zip(CLASS_NAMES, counts)):
                if count > 0:
                    percentage = (count / total) * 100
                    f.write(f"{class_id:<10} {name:<20} {count:<15,} {percentage:>6.2f}%\n")

            f.write("-" * 80 + "\n")
            f.write(f"Total: {total:,} pixels\n\n")

    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
