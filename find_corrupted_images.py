"""
Script to identify corrupted image files in the StreetHazards dataset.
Checks all images to find which ones cannot be loaded by PIL.
"""

import glob
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def check_image(image_path):
    """
    Try to open and convert an image to RGB.
    Returns True if successful, False if corrupted.
    """
    try:
        img = Image.open(image_path)
        img.convert('RGB')  # This is where the error occurred in training
        img.close()
        return True
    except Exception as e:
        return False, str(e)

def find_corrupted_images(root_dir, split='training'):
    """
    Find all corrupted images in a dataset split.

    Args:
        root_dir: Path to dataset root (e.g., 'streethazards_train/train')
        split: 'training', 'validation', or 'test'
    """
    print(f"\n{'='*60}")
    print(f"Checking {split} split for corrupted images...")
    print(f"{'='*60}")

    # Find all image paths
    image_dir = Path(root_dir) / 'images' / split
    image_paths = sorted(glob.glob(str(image_dir / '**' / '*.png'), recursive=True))

    print(f"Total images to check: {len(image_paths)}")
    print(f"Checking images...\n")

    corrupted_images = []

    # Check each image
    for img_path in tqdm(image_paths, desc=f"Checking {split}"):
        result = check_image(img_path)
        if result != True:
            corrupted_images.append((img_path, result[1]))

    # Report results
    print(f"\n{'='*60}")
    print(f"RESULTS: {split} split")
    print(f"{'='*60}")
    print(f"Total images checked: {len(image_paths)}")
    print(f"Corrupted images found: {len(corrupted_images)}")

    if corrupted_images:
        print(f"\n⚠️  CORRUPTED FILES:")
        for path, error in corrupted_images:
            print(f"  - {path}")
            print(f"    Error: {error}")
    else:
        print(f"✅ All images are valid!")

    print(f"{'='*60}\n")

    return corrupted_images

if __name__ == "__main__":
    print("\nStreetHazards Image Corruption Checker")
    print("="*60)

    # Check training split (where the crash occurred)
    train_corrupted = find_corrupted_images('streethazards_train/train', 'training')

    # Also check validation split for completeness
    val_corrupted = find_corrupted_images('streethazards_train/train', 'validation')

    # Summary
    total_corrupted = len(train_corrupted) + len(val_corrupted)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total corrupted images: {total_corrupted}")
    print(f"  - Training: {len(train_corrupted)}")
    print(f"  - Validation: {len(val_corrupted)}")

    if total_corrupted > 0:
        print("\n⚠️  ACTION REQUIRED:")
        print("  1. Re-download the corrupted files")
        print("  2. Or remove them from the dataset")
        print("  3. Or add error handling in dataloader to skip them")
    else:
        print("\n✅ No corrupted images found!")
        print("   The crash may have been a temporary disk/memory issue.")

    print("="*60 + "\n")
