"""
Quick test to verify that the augmented dataloader works correctly.
Loads a few samples and displays them to check augmentations.

Usage:
    python test_augmented_dataloader.py --num_samples 5 --image_idx 100
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
from dataloader import StreetHazardsDataset, get_transforms, denormalize_image, mask_to_rgb, CLASS_NAMES
from config import IMAGE_SIZE, TRAIN_ROOT

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Test augmented dataloader with visualization')
parser.add_argument('--num_samples', type=int, default=4,
                    help='Number of samples to display (default: 4, includes 1 original + N augmented)')
parser.add_argument('--image_idx', type=int, default=101,
                    help='Index of image to test (default: 101)')
args = parser.parse_args()

print("="*80)
print("TESTING AUGMENTED DATALOADER")
print("="*80)
print(f"\nConfiguration:")
print(f"  - Number of samples: {args.num_samples} (1 original + {args.num_samples-1} augmented)")
print(f"  - Image index: {args.image_idx}")

# Create datasets
print("\nCreating datasets...")

# Training dataset with augmentations
train_transform, train_mask_transform = get_transforms(IMAGE_SIZE, is_training=True)
train_dataset = StreetHazardsDataset(
    root_dir=TRAIN_ROOT,
    split='training',
    transform=train_transform,
    mask_transform=train_mask_transform
)

# Validation dataset without augmentations (for original image)
val_transform, val_mask_transform = get_transforms(IMAGE_SIZE, is_training=False)
val_dataset = StreetHazardsDataset(
    root_dir=TRAIN_ROOT,
    split='training',
    transform=val_transform,
    mask_transform=val_mask_transform
)

print(f"✅ Dataset loaded: {len(train_dataset)} training samples")
print(f"✅ Augmentations enabled:")
print("   - Multi-scale random crop (0.5-2.0x) with variable crop sizes")
print("   - Random horizontal flip")
print("   - Color jitter (including hue)")
print("   - Gaussian blur (50%)")
print("   - NO rotation (commented out to avoid black edges)")

# Test loading samples
print(f"\nGenerating visualization with {args.num_samples} rows...")
print(f"  - Row 1: Original image (no augmentation)")
print(f"  - Rows 2-{args.num_samples}: Same image with different random augmentations")

fig, axes = plt.subplots(args.num_samples, 3, figsize=(15, 4 * args.num_samples))
if args.num_samples == 1:
    axes = axes.reshape(1, -1)  # Ensure 2D array for single row
fig.suptitle(f'Augmentation Test: Image Index {args.image_idx}', fontsize=16)

idx = args.image_idx

for i in range(args.num_samples):
    try:
        if i == 0:
            # First row: Load original image without augmentations
            image, mask, path = val_dataset[idx]
            row_title = "Original (No Augmentation)"
        else:
            # Subsequent rows: Load with augmentations
            image, mask, path = train_dataset[idx]
            row_title = f"Augmented #{i}"

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

        print(f"✅ Row {i+1}/{args.num_samples}: {row_title} - Image shape {image.shape}, Mask shape {mask.shape}")

    except Exception as e:
        print(f"❌ Error loading row {i+1}: {e}")
        import traceback
        traceback.print_exc()

plt.tight_layout()
output_path = 'assets/augmented_dataloader_test.png'
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"\n✅ Test visualization saved to: {output_path}")
print(f"   Figure size: {args.num_samples} rows × 3 columns")
print(f"   Row 1: Original image without augmentations")
print(f"   Rows 2-{args.num_samples}: Same image with random augmentations")

# Test batch loading
print("\nTesting batch loading...")
from torch.utils.data import DataLoader

loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

try:
    batch_images, batch_masks, batch_paths = next(iter(loader))
    print(f"✅ Batch loaded successfully!")
    print(f"   Images shape: {batch_images.shape}")
    print(f"   Masks shape: {batch_masks.shape}")
    print(f"   Image dtype: {batch_images.dtype}")
    print(f"   Mask dtype: {batch_masks.dtype}")
    print(f"   Image range: [{batch_images.min():.2f}, {batch_images.max():.2f}]")
    print(f"   Mask range: [{batch_masks.min()}, {batch_masks.max()}]")
except Exception as e:
    print(f"❌ Error loading batch: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DATALOADER TEST COMPLETE!")
print("="*80)
print("\nIf you see this message and the visualization was saved,")
print("the augmented dataloader is working correctly!")
print("\nYou can now run: .venv/bin/python3 train_augmented_resnet50.py")
print("="*80 + "\n")
