"""
Test script to verify that data augmentations are working.
Shows side-by-side comparison of original vs augmented images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataloader import StreetHazardsDataset, get_transforms, denormalize_image, mask_to_rgb
from config import TRAIN_ROOT

print("="*60)
print("TESTING DATA AUGMENTATIONS")
print("="*60)

# Create two versions of the same dataset
print("\n1. Creating training dataset (WITH augmentation)...")
train_transform, train_mask_transform = get_transforms(image_size=512, is_training=True)
train_dataset = StreetHazardsDataset(
    root_dir=TRAIN_ROOT,
    split='training',
    transform=train_transform,
    mask_transform=train_mask_transform
)

print("2. Creating validation dataset (WITHOUT augmentation)...")
val_transform, val_mask_transform = get_transforms(image_size=512, is_training=False)
val_dataset = StreetHazardsDataset(
    root_dir=TRAIN_ROOT,
    split='training',  # Same split, different transforms!
    transform=val_transform,
    mask_transform=val_mask_transform
)

# Test on same image multiple times
test_idx = 100
num_samples = 6

print(f"\n3. Loading image #{test_idx} multiple times to see random augmentations...")

# Get original image
orig_img, orig_mask, path = val_dataset.get_raw_item(test_idx)
print(f"   Original image size: {orig_img.shape}")
print(f"   Path: {path}")

# Create figure
fig, axes = plt.subplots(3, num_samples, figsize=(20, 10))

# First row: Original (repeated)
for col in range(num_samples):
    # Get without augmentation
    img_no_aug, mask_no_aug, _ = val_dataset[test_idx]
    img_vis = denormalize_image(img_no_aug)

    axes[0, col].imshow(img_vis)
    axes[0, col].set_title('No Augmentation\n(Validation Transform)', fontsize=9)
    axes[0, col].axis('off')

# Second row: With augmentation (should vary!)
for col in range(num_samples):
    # Get with augmentation - should be different each time!
    img_aug, mask_aug, _ = train_dataset[test_idx]
    img_vis = denormalize_image(img_aug)

    axes[1, col].imshow(img_vis)
    axes[1, col].set_title(f'WITH Augmentation\nSample {col+1}', fontsize=9)
    axes[1, col].axis('off')

# Third row: Raw original for reference
for col in range(num_samples):
    axes[2, col].imshow(orig_img)
    axes[2, col].set_title('Original Raw\n(Before Resize)', fontsize=9)
    axes[2, col].axis('off')

plt.suptitle(f'Augmentation Test - Image #{test_idx}\n' +
             'Row 1: Without augmentation (should all look identical)\n' +
             'Row 2: With augmentation (should vary - flips, color changes)\n' +
             'Row 3: Original image for reference',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('assets/augmentation_test.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved comparison to: assets/augmentation_test.png")
plt.show()

# Detailed analysis
print("\n" + "="*60)
print("DETAILED ANALYSIS")
print("="*60)

# Sample multiple times and check for differences
print("\n4. Checking if augmentations are actually random...")
images_aug = []
for i in range(10):
    img, _, _ = train_dataset[test_idx]
    images_aug.append(img)

# Check if images are different
all_same = True
for i in range(1, len(images_aug)):
    if not torch.equal(images_aug[0], images_aug[i]):
        all_same = False
        break

if all_same:
    print("   ❌ ERROR: All augmented images are IDENTICAL!")
    print("   ❌ Augmentations are NOT working!")
    print("\n   Possible causes:")
    print("   1. Random seed is set somewhere (check for torch.manual_seed or np.random.seed)")
    print("   2. Transforms are not being applied")
    print("   3. Dataset is using wrong transform")
else:
    print("   ✅ SUCCESS: Augmented images are DIFFERENT!")
    print("   ✅ Augmentations are working correctly!")

    # Calculate difference
    max_diff = 0
    for i in range(1, len(images_aug)):
        diff = (images_aug[0] - images_aug[i]).abs().max().item()
        max_diff = max(max_diff, diff)

    print(f"   ✅ Max pixel difference between samples: {max_diff:.4f}")

# Test horizontal flip specifically
print("\n5. Testing horizontal flip specifically...")
flips_detected = 0
for i in range(20):
    img, _, _ = train_dataset[test_idx]
    # Check if image is flipped by comparing left vs right edges
    left_edge = img[:, :, :10].mean()
    right_edge = img[:, :, -10:].mean()

    # Get reference (no flip)
    img_ref, _, _ = val_dataset[test_idx]
    left_edge_ref = img_ref[:, :, :10].mean()
    right_edge_ref = img_ref[:, :, -10:].mean()

    # If edges are swapped, likely flipped
    if abs(left_edge - right_edge_ref) < abs(left_edge - left_edge_ref):
        flips_detected += 1

print(f"   Flips detected in 20 samples: {flips_detected}/20")
print(f"   Expected: ~10 (50% flip probability)")

if flips_detected < 3:
    print("   ⚠️  WARNING: Very few flips detected! Check RandomHorizontalFlip")
elif flips_detected > 17:
    print("   ⚠️  WARNING: Too many flips detected! Probability might be wrong")
else:
    print("   ✅ Flip rate looks reasonable")

# Test color jitter
print("\n6. Testing color jitter...")
colors_no_aug = []
colors_aug = []

for i in range(10):
    img_no_aug, _, _ = val_dataset[test_idx]
    img_aug, _, _ = train_dataset[test_idx]

    colors_no_aug.append(img_no_aug.mean().item())
    colors_aug.append(img_aug.mean().item())

color_var_no_aug = np.var(colors_no_aug)
color_var_aug = np.var(colors_aug)

print(f"   Variance without augmentation: {color_var_no_aug:.6f}")
print(f"   Variance with augmentation: {color_var_aug:.6f}")

if color_var_aug > color_var_no_aug * 2:
    print("   ✅ Color jitter is working (augmented has higher variance)")
else:
    print("   ⚠️  Color jitter effect is weak or not working")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nCheck assets/augmentation_test.png to visually confirm augmentations!")
print("If row 2 images look identical, augmentations are NOT working.")
print("If row 2 images vary (flips, colors), augmentations ARE working.")
print("="*60)
