"""
Test and visualize scale augmentation from a dataloader.

Usage:
    # As a module:
    from utils.test_augmented_dataloader import visualize_scale_augmentation
    visualize_scale_augmentation(train_dataset, val_dataset, image_idx=100, num_samples=4)

    # As a script:
    python -m utils.test_augmented_dataloader --num_samples 5 --image_idx 100
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.dataloader import denormalize_image, mask_to_rgb


def visualize_scale_augmentation(train_dataset, val_dataset=None, image_idx=101,
                                  num_samples=4, save_path=None, show=True):
    """
    Visualize scale augmentation effects on a single image.

    Args:
        train_dataset: Dataset with augmentations enabled
        val_dataset: Dataset without augmentations (for original). If None, uses train_dataset for all.
        image_idx: Index of image to visualize
        num_samples: Number of rows (1 original + N-1 augmented)
        save_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        fig: matplotlib Figure object
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'Scale Augmentation Visualization: Image Index {image_idx}', fontsize=16)

    for i in range(num_samples):
        if i == 0 and val_dataset is not None:
            # First row: Original image without augmentations
            image, mask, path = val_dataset[image_idx]
            row_title = "Original (No Augmentation)"
        else:
            # Subsequent rows: Load with augmentations
            image, mask, path = train_dataset[image_idx]
            row_title = f"Scale Augmented #{i}" if i > 0 else "Augmented #0"

        # Denormalize image
        img_np = denormalize_image(image)

        # Convert mask to RGB
        mask_np = mask.numpy()
        mask_rgb = mask_to_rgb(mask_np)

        # Create overlay
        overlay = (img_np * 0.6 + mask_rgb * 0.4).astype(np.uint8)

        # Display
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"{row_title} - Image")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_rgb)
        axes[i, 1].set_title(f"{row_title} - Mask")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"{row_title} - Overlay")
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    if show:
        plt.show()

    return fig


def visualize_from_dataloader(dataset, num_samples=4, save_path=None, show=True, image_idx=0):
    """
    Visualize scale augmentation on a single image.

    First row shows original, remaining rows show augmented versions.

    Args:
        dataset: StreetHazardsDataset with augmentations enabled
        num_samples: Number of rows (1 original + N-1 augmented)
        save_path: Path to save figure (optional)
        show: Whether to display the figure
        image_idx: Index of image to visualize

    Returns:
        fig: matplotlib Figure object
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'Scale Augmentation - Image #{image_idx}', fontsize=16)

    for i in range(num_samples):
        if i == 0:
            # First row: original image (no augmentation)
            img_np, mask_np, path = dataset.get_raw_item(image_idx)
            row_label = "Original"
        else:
            # Augmented versions
            image, mask, path = dataset[image_idx]
            img_np = denormalize_image(image)
            mask_np = mask.numpy() if hasattr(mask, 'numpy') else np.array(mask)
            row_label = f"Augmented #{i}"

        mask_rgb = mask_to_rgb(mask_np)

        # Create overlay
        overlay = (img_np * 0.6 + mask_rgb * 0.4).astype(np.uint8)

        # Display
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"{row_label} - Image")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_rgb)
        axes[i, 1].set_title(f"{row_label} - Mask")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"{row_label} - Overlay")
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    import argparse
    from utils.dataloader import StreetHazardsDataset, get_transforms
    from config import IMAGE_SIZE, TRAIN_ROOT

    parser = argparse.ArgumentParser(description='Test scale augmentation visualization')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to display (default: 4)')
    parser.add_argument('--image_idx', type=int, default=101,
                        help='Index of image to test (default: 101)')
    args = parser.parse_args()

    print("="*80)
    print("SCALE AUGMENTATION TEST")
    print("="*80)

    # Create datasets with scale-only augmentation
    train_transform, train_mask_transform = get_transforms(IMAGE_SIZE, is_training=True)
    train_dataset = StreetHazardsDataset(
        root_dir=TRAIN_ROOT,
        split='training',
        transform=train_transform,
        mask_transform=train_mask_transform
    )

    val_transform, val_mask_transform = get_transforms(IMAGE_SIZE, is_training=False)
    val_dataset = StreetHazardsDataset(
        root_dir=TRAIN_ROOT,
        split='training',
        transform=val_transform,
        mask_transform=val_mask_transform
    )

    print(f"Dataset loaded: {len(train_dataset)} samples")

    # Visualize
    visualize_scale_augmentation(
        train_dataset,
        val_dataset,
        image_idx=args.image_idx,
        num_samples=args.num_samples,
        save_path='assets/augmented_dataloader_test.png',
        show=False
    )

    print("="*80)
    print("TEST COMPLETE!")
    print("="*80)
