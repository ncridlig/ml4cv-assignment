import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from config import (
    DEVICE,
    NUM_CLASSES,
    IGNORE_INDEX,
    BATCH_SIZE,
    LEARNING_RATE as LR,
    EPOCHS,
    PRINT_FREQ,
    NUM_WORKERS,
    IMAGE_SIZE,
    TRAIN_ROOT
)

# -----------------------------
# Class definitions for StreetHazards
# -----------------------------
# Official mapping from StreetHazards dataset
# Original indices 1-14 → After -1 remapping → 0-13
CLASS_NAMES = [
    'unlabeled',      # 0 (originally 1)
    'building',       # 1 (originally 2)
    'fence',          # 2 (originally 3)
    'other',          # 3 (originally 4)
    'pedestrian',     # 4 (originally 5)
    'pole',           # 5 (originally 6)
    'road line',      # 6 (originally 7)
    'road',           # 7 (originally 8)
    'sidewalk',       # 8 (originally 9)
    'vegetation',     # 9 (originally 10)
    'car',            # 10 (originally 11)
    'wall',           # 11 (originally 12)
    'traffic sign',   # 12 (originally 13)
    'anomaly'         # 13 (originally 14) - ONLY in test set
]

# Color map for visualization (RGB) - Official StreetHazards colors
CLASS_COLORS = np.array([
    [0, 0, 0],         # 0: unlabeled - black
    [70, 70, 70],      # 1: building - dark gray
    [190, 153, 153],   # 2: fence - light gray
    [250, 170, 160],   # 3: other - pink
    [220, 20, 60],     # 4: pedestrian - red
    [153, 153, 153],   # 5: pole - gray
    [157, 234, 50],    # 6: road line - lime green
    [128, 64, 128],    # 7: road - purple
    [244, 35, 232],    # 8: sidewalk - magenta
    [107, 142, 35],    # 9: vegetation - olive green
    [0, 0, 142],       # 10: car - dark blue
    [102, 102, 156],   # 11: wall - blue-gray
    [220, 220, 0],     # 12: traffic sign - yellow
    [60, 250, 240]     # 13: anomaly - cyan/turquoise - ONLY in test set
], dtype=np.uint8)

# Constants
NUM_KNOWN_CLASSES = 13  # Known classes in training (0-12)
ANOMALY_CLASS_IDX = 13  # Index of anomaly class (ONLY in test)

# -----------------------------
# Custom Dataset Class
# -----------------------------
class StreetHazardsDataset(Dataset):
    """
    Custom dataset for StreetHazards segmentation.
    Loads image-mask pairs from the dataset folder structure.
    """
    def __init__(self, root_dir, split='training', transform=None, mask_transform=None, image_size=None, augconfig=None):
        """
        Args:
            root_dir (str): Path to dataset root (e.g., 'streethazards_train/train')
            split (str): 'training', 'validation', or 'test'
            transform: Transformations for images
            mask_transform: Transformations for masks
            image_size: Target image size (H, W). If None, uses IMAGE_SIZE from config.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_size = image_size if image_size is not None else IMAGE_SIZE
        self.augconfig = augconfig

        # Find all image paths
        if split == 'test':
            self.image_dir = self.root_dir / 'images' / 'test'
            self.mask_dir = self.root_dir / 'annotations' / 'test'
        else:
            self.image_dir = self.root_dir / 'images' / split
            self.mask_dir = self.root_dir / 'annotations' / split

        # Collect all image files recursively
        self.image_paths = sorted(glob.glob(str(self.image_dir / '**' / '*.png'), recursive=True))

        # Generate corresponding mask paths
        self.mask_paths = []
        for img_path in self.image_paths:
            # Replace 'images' with 'annotations' in the path
            mask_path = img_path.replace('/images/', '/annotations/')
            self.mask_paths.append(mask_path)

        # Verify all masks exist
        for mask_path in self.mask_paths:
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

        print(f"Loaded {len(self.image_paths)} {split} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor
            mask: Mask tensor with 0-based class indices (H, W), range 0-13
            image_path: Path to the image file
        """
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # Convert mask to numpy and remap from 1-indexed to 0-indexed
        mask_array = np.array(mask, dtype=np.int64) - 1
        mask_pil = Image.fromarray(mask_array.astype(np.uint8))

        # Apply transformations
        if self.transform is None and self.mask_transform is None:
            # Training mode: apply joint transforms (synchronized augmentations)
            if self.augconfig is None:

                # 1. Multi-scale random crop (KEY AUGMENTATION)
                # Following DeepLabV3+ paper: variable crop sizes with scale range [0.5, 2.0]
                # Default scale_range can be overridden via augconfig
                default_scale_range = (0.5, 2.0)
                scale_crop = JointRandomScaleCrop(output_size=self.image_size, scale_range=default_scale_range, base_crop_size=512)
                image, mask_pil = scale_crop(image, mask_pil)

                # 2. Random rotation (±10 degrees) introduces black edges
                # rotation = JointRandomRotation(degrees=10)
                # image, mask_pil = rotation(image, mask_pil)

                # 3. Random horizontal flip
                flip = JointRandomHorizontalFlip(p=0.5)
                image, mask_pil = flip(image, mask_pil)

                # 4. Image-only augmentations
                # Color jitter (image only)
                image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)(image)

                # 5. Gaussian blur (image only, simulate motion/focus blur)
                if np.random.random() < 0.5:
                    image = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(image)
            
            else:
                # Apply augmentations based on augconfig
                if self.augconfig.get('scale', False):
                    # Get scale_range from augconfig, default to (0.5, 2.0)
                    scale_range = self.augconfig.get('scale_range', (0.5, 2.0))
                    scale_crop = JointRandomScaleCrop(output_size=self.image_size, scale_range=scale_range, base_crop_size=512)
                    image, mask_pil = scale_crop(image, mask_pil)

                if self.augconfig.get('rotate', False):
                    rotation = JointRandomRotation(degrees=10)
                    image, mask_pil = rotation(image, mask_pil)

                if self.augconfig.get('flip', False):
                    flip = JointRandomHorizontalFlip(p=0.5)
                    image, mask_pil = flip(image, mask_pil)

                if self.augconfig.get('color', False):
                    image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)(image)

            # Convert to tensor and normalize
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

            # Convert mask to tensor
            mask = torch.from_numpy(np.array(mask_pil, dtype=np.int64))

        else:
            # Validation/test mode: use standard transforms
            if self.transform:
                image = self.transform(image)

            if self.mask_transform:
                mask = self.mask_transform(mask_pil)
            else:
                mask = torch.from_numpy(mask_array)

        return image, mask, img_path

    def get_raw_item(self, idx):
        """
        Returns raw image and mask without transformations (for visualization).
        Mask is 0-indexed (0-13).
        """
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = np.array(Image.open(img_path).convert('RGB'))
        # Remap mask from 1-indexed (file) to 0-indexed (our classes)
        mask = np.array(Image.open(mask_path), dtype=np.int64) - 1

        return image, mask, img_path


# -----------------------------
# Custom Synchronized Transforms
# -----------------------------
class JointRandomHorizontalFlip:
    """Apply random horizontal flip to both image and mask."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() < self.p:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        return image, mask


class JointRandomRotation:
    """Apply random rotation to both image and mask."""
    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, image, mask):
        angle = np.random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = transforms.functional.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        return image, mask


class JointRandomScaleCrop:
    """
    Multi-scale training following DeepLabV3+ paper literally.

    Paper approach:
    1. Random scale in range [0.5, 2.0]
    2. Random crop with size proportional to scale (variable crop size)
    3. Resize crop to fixed output_size for batching

    This avoids padding:
    - Scale 0.5x: crop ~256x256, resize to 512x512 (zooms in, sees fine details)
    - Scale 1.0x: crop ~512x512, resize to 512x512 (normal view)
    - Scale 2.0x: crop ~1024x1024, resize to 512x512 (zooms out, sees context)

    No black padding needed because crop size adapts to scale!
    """
    def __init__(self, output_size=(512, 512), scale_range=(0.5, 2.0), base_crop_size=512):
        self.output_size = output_size  # Fixed size for batching (512x512)
        self.scale_range = scale_range
        self.base_crop_size = base_crop_size  # Base crop size at scale=1.0

    def __call__(self, image, mask):
        # Original size
        w, h = image.size  # StreetHazards: 1280×720

        # Random scale factor
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Scaled image size
        scaled_h = int(h * scale)
        scaled_w = int(w * scale)

        # Resize to scaled size
        image = transforms.functional.resize(image, (scaled_h, scaled_w),
                                            interpolation=transforms.InterpolationMode.BILINEAR)
        mask = transforms.functional.resize(mask, (scaled_h, scaled_w),
                                           interpolation=transforms.InterpolationMode.NEAREST)

        # Variable crop size: proportional to scale
        # At scale=0.5: crop_size = 256
        # At scale=1.0: crop_size = 512
        # At scale=2.0: crop_size = 1024
        crop_size = int(self.base_crop_size * scale)
        crop_size = min(crop_size, scaled_h, scaled_w)  # Don't exceed image bounds

        # Random crop of variable size
        if scaled_h > crop_size:
            top = np.random.randint(0, scaled_h - crop_size + 1)
        else:
            top = 0

        if scaled_w > crop_size:
            left = np.random.randint(0, scaled_w - crop_size + 1)
        else:
            left = 0

        image = transforms.functional.crop(image, top, left, crop_size, crop_size)
        mask = transforms.functional.crop(mask, top, left, crop_size, crop_size)

        # Resize crop to fixed output size for batching
        image = transforms.functional.resize(image, self.output_size,
                                            interpolation=transforms.InterpolationMode.BILINEAR)
        mask = transforms.functional.resize(mask, self.output_size,
                                           interpolation=transforms.InterpolationMode.NEAREST)

        return image, mask


# -----------------------------
# Transformations
# -----------------------------
def get_transforms(image_size=(512, 512), is_training=True):
    """
    Get image and mask transforms with multi-scale training support.

    Args:
        image_size: Target size for images (default: 512x512)
        is_training: If True, apply strong data augmentation including multi-scale

    Returns:
        image_transform: Transform for images
        mask_transform: Transform for masks

    Note: For training, we use custom joint transforms that apply the same
          random transformations to both image and mask (flip, rotation, crop).
    """
    if is_training:
        # IMPORTANT: We return None for transforms and handle augmentation in __getitem__
        # This allows synchronized transforms between image and mask
        return None, None
    else:
        # Validation/test transforms (no augmentation)
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64)))
        ])

        return image_transform, mask_transform


# -----------------------------
# Visualization Functions
# -----------------------------
def denormalize_image(img_tensor):
    """
    Denormalize image tensor for visualization.

    Args:
        img_tensor: Normalized image tensor (C, H, W)
    Returns:
        Denormalized image as numpy array (H, W, C)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).numpy()

    return img


def mask_to_rgb(mask, class_colors=CLASS_COLORS):
    """
    Convert segmentation mask to RGB image.

    Args:
        mask: Mask array (H, W) with class indices
        class_colors: Color map (num_classes, 3)
    Returns:
        RGB image (H, W, 3)
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(len(class_colors)):
        rgb[mask == class_id] = class_colors[class_id]

    return rgb


def plot_samples(dataset, indices=None, num_samples=5, figsize=(20, 12), save_path='assets/sample.png'):
    """
    Plot image-mask pairs from dataset.

    Args:
        dataset: StreetHazardsDataset instance
        indices: List of indices to plot. If None, random samples are chosen.
        num_samples: Number of samples to plot
        figsize: Figure size
    """
    if indices is None:
        indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    else:
        num_samples = len(indices)

    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        # Get raw data (without transforms)
        image, mask, img_path = dataset.get_raw_item(idx)

        # Convert mask to RGB
        mask_rgb = mask_to_rgb(mask)

        # Count unique classes in this mask
        unique_classes = np.unique(mask)
        class_names_in_mask = [CLASS_NAMES[c] for c in unique_classes if c < len(CLASS_NAMES)]

        # Plot image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Image {idx}\n{Path(img_path).name}', fontsize=10)
        axes[i, 0].axis('off')

        # Plot mask
        axes[i, 1].imshow(mask_rgb)
        axes[i, 1].set_title(f'Segmentation Mask', fontsize=10)
        axes[i, 1].axis('off')

        # Plot overlay
        overlay = (image * 0.6 + mask_rgb * 0.4).astype(np.uint8)
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'Overlay\nClasses: {", ".join(class_names_in_mask[:5])}{"..." if len(class_names_in_mask) > 5 else ""}',
                            fontsize=9)
        axes[i, 2].axis('off')

    plt.tight_layout()

    # Save the figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample plots saved to: {save_path}")

    plt.show()


def show_legend(save_path='assets/class_color_map.png'):
    """Display color legend for all classes and save to assets folder."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create legend patches
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        rect = plt.Rectangle((0, len(CLASS_NAMES) - 1 - i), 1, 0.8,
                            facecolor=color/255.0, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(1.2, len(CLASS_NAMES) - i - 0.6, f'{i}: {name}',
               va='center', fontsize=11, fontweight='bold')

    ax.set_xlim(0, 5)
    ax.set_ylim(0, len(CLASS_NAMES))
    ax.axis('off')
    ax.set_title('StreetHazards Class Color Legend', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save the figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Color map saved to: {save_path}")

    plt.show()


# -----------------------------
# DataLoader Factory Functions
# -----------------------------
def get_dataloaders(batch_size=8, num_workers=4, image_size=IMAGE_SIZE):
    """
    Create train, validation, and test dataloaders.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        image_size: Target image size

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform, train_mask_transform = get_transforms(image_size, is_training=True)
    val_transform, val_mask_transform = get_transforms(image_size, is_training=False)

    # Create datasets
    train_dataset = StreetHazardsDataset(
        root_dir='streethazards_train/train',
        split='training',
        transform=train_transform,
        mask_transform=train_mask_transform
    )

    val_dataset = StreetHazardsDataset(
        root_dir='streethazards_train/train',
        split='validation',
        transform=val_transform,
        mask_transform=val_mask_transform
    )

    test_dataset = StreetHazardsDataset(
        root_dir='streethazards_test/test',
        split='test',
        transform=val_transform,
        mask_transform=val_mask_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("StreetHazards DataLoader Test")
    print("=" * 60)

    # Create datasets
    train_transform, train_mask_transform = get_transforms(IMAGE_SIZE, is_training=True)
    val_test_transform, val_test_mask_transform = get_transforms(IMAGE_SIZE, is_training=False)

    train_dataset = StreetHazardsDataset(
        root_dir='streethazards_train/train',
        split='training',
        transform=train_transform,
        mask_transform=train_mask_transform
    )

    val_dataset = StreetHazardsDataset(
        root_dir='streethazards_train/train',
        split='validation',
        transform=val_test_transform,
        mask_transform=val_test_mask_transform
    )

    test_dataset = StreetHazardsDataset(
        root_dir='streethazards_test/test',
        split='test',
        transform=val_test_transform,
        mask_transform=val_test_mask_transform
    )

    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    # Verify remapping
    print("\n" + "=" * 60)
    print("Verifying class remapping...")
    print("=" * 60)

    # Check training classes
    train_classes = set()
    for i in range(min(100, len(train_dataset))):
        _, mask, _ = train_dataset.get_raw_item(i)
        train_classes.update(np.unique(mask))
    print(f"Training classes (remapped): {sorted(train_classes)}")
    print(f"Expected: 0-12 (13 known classes: unlabeled + 12 semantic classes)")
    print(f"Class 0 (unlabeled) in training: {0 in train_classes}")
    print(f"Class 13 (anomaly) in training: {13 in train_classes} (should be False)")

    # Check test classes
    test_classes = set()
    for i in range(min(100, len(test_dataset))):
        _, mask, _ = test_dataset.get_raw_item(i)
        test_classes.update(np.unique(mask))
    print(f"\nTest classes (remapped): {sorted(test_classes)}")
    print(f"Expected: 0-12 (known) + 13 (anomaly)")
    print(f"Class 13 (anomaly) present in test: {ANOMALY_CLASS_IDX in test_classes} (should be True)")

    print("\n" + "=" * 60)
    print("Displaying class legend...")
    print("=" * 60)
    show_legend()

    print("\n" + "=" * 60)
    print("Plotting training samples...")
    print("=" * 60)
    plot_samples(train_dataset, num_samples=5, save_path='assets/train_samples.png')

    print("\n" + "=" * 60)
    print("Plotting validation samples...")
    print("=" * 60)
    plot_samples(val_dataset, num_samples=5, save_path='assets/val_samples.png')

    print("\n" + "=" * 60)
    print("Plotting test samples (with anomalies)...")
    print("=" * 60)
    plot_samples(test_dataset, num_samples=5, save_path='assets/test_samples.png')

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
