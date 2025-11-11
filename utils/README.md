# Utility Functions

This directory contains utility modules for data loading, model management, visualization, and dataset analysis.

## Modules

### 1. `dataloader.py`

**Purpose**: Dataset loading, augmentation, and preprocessing for StreetHazards semantic segmentation.

**Key Components**:

#### Dataset Class
```python
class StreetHazardsDataset(Dataset):
    """
    Dataset for StreetHazards semantic segmentation.
    Handles train, validation, and test splits.
    """
```

**Features**:
- Automatic split handling (training/validation/test)
- Label remapping (original 1-14 → 0-13)
- Ignore index handling for anomaly class (13)
- PIL image loading with error handling
- Transform pipeline application

#### Transform Functions
```python
def get_transforms(split='train', image_size=(512, 512)):
    """
    Get transforms for different dataset splits.

    Args:
        split: 'train', 'validation', or 'test'
        image_size: Tuple of (height, width)

    Returns:
        img_transform, mask_transform
    """
```

**Training Augmentations** (Multi-scale strategy):
- Random horizontal flip (50%)
- Random scaling (0.5× to 2.0×)
- Random crop to target size
- ColorJitter (brightness, contrast, saturation, hue)
- Normalization (ImageNet stats)

**Validation/Test Augmentations**:
- Resize to target size
- Normalization only (no augmentation)

#### Utility Functions
```python
denormalize_image(tensor)  # Reverse ImageNet normalization for visualization
mask_to_rgb(mask)          # Convert class mask to RGB visualization
```

#### Constants
```python
CLASS_NAMES = [...]        # 14 class names (0-13)
CLASS_COLORS = [...]       # RGB colors for visualization
NUM_CLASSES = 13           # Training classes (0-12)
ANOMALY_CLASS_IDX = 13     # Anomaly class for test set
```

**Usage**:
```python
from utils.dataloader import StreetHazardsDataset, get_transforms

# Create dataset
img_t, mask_t = get_transforms(split='train', image_size=(512, 512))
dataset = StreetHazardsDataset(
    root='streethazards_train/train',
    split='training',
    image_transform=img_t,
    mask_transform=mask_t
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
```

---

### 2. `model_utils.py`

**Purpose**: Loading and managing segmentation models.

**Key Functions**:

```python
def load_model(model_path, device, num_classes=13, architecture='deeplabv3_resnet50'):
    """
    Load trained segmentation model.

    Args:
        model_path: Path to checkpoint (.pth file)
        device: torch.device (cuda/cpu)
        num_classes: Number of output classes (default: 13)
        architecture: Model architecture
            - 'deeplabv3_resnet50' (default)
            - 'deeplabv3_resnet101'
            - 'fcn_resnet50'

    Returns:
        model: Loaded model in eval mode
    """
```

**Supported Architectures**:
- **DeepLabV3+ ResNet50**: 50.26% mIoU (best model)
- **DeepLabV3+ ResNet101**: 37% mIoU
- **FCN ResNet50**: Alternative architecture

**Features**:
- Automatic classifier head replacement
- Strict=False loading (ignores aux_classifier)
- Device-agnostic loading
- Eval mode by default

**Usage**:
```python
from utils.model_utils import load_model
from config import MODEL_PATH, DEVICE, NUM_CLASSES, MODEL_ARCHITECTURE

model = load_model(MODEL_PATH, DEVICE, NUM_CLASSES, MODEL_ARCHITECTURE)
```

---

### 3. `class_counter.py`

**Purpose**: Dataset analysis - count class distribution across splits.

**Key Functions**:

```python
def count_class_pixels(dataset, dataset_name="Dataset"):
    """
    Count pixel distribution for each class.

    Args:
        dataset: StreetHazardsDataset instance
        dataset_name: Name for display

    Returns:
        class_counts: numpy array of pixel counts per class
    """
```

**Features**:
- Pixel-level class counting
- Percentage calculation
- Progress bar with tqdm
- Supports all splits (train/val/test)
- Handles anomaly class separately

**Usage**:
```bash
# Run from project root
.venv/bin/python3 utils/class_counter.py
```

**Output Example**:
```
============================================================
Analyzing Training Set (5125 images)
============================================================
unlabeled      :   234,567,890 pixels (12.34%)
building       :   123,456,789 pixels (6.54%)
...
```

---

### 4. `visualize.py`

**Purpose**: Visualization utilities for segmentation masks.

**Key Components**:

```python
COLORS = np.array([...])  # 14 RGB colors for classes 0-13

def color(annot_path: str, colors: np.ndarray) -> Image.Image:
    """
    Convert grayscale annotation to RGB visualization.

    Args:
        annot_path: Path to grayscale annotation mask
        colors: Array of RGB colors (14 x 3)

    Returns:
        RGB PIL Image
    """
```

**Color Mapping**:
- Unlabeled: Black [0, 0, 0]
- Building: Gray [70, 70, 70]
- Road: Purple [128, 64, 128]
- Car: Blue [0, 0, 142]
- Anomaly: Cyan [60, 250, 240]
- ... (14 total classes)

**Source**:
Based on StreetHazards official color scheme
https://github.com/hendrycks/anomaly-seg/issues/15

**Usage**:
```python
from utils.visualize import color, COLORS

# Convert annotation to colored mask
segm_map = color('path/to/annotation.png', COLORS)
segm_map.save('output/colored_mask.png')
```

---

## File Structure

```
utils/
├── README.md              # This file
├── __init__.py            # Package initialization
├── dataloader.py          # Dataset loading and augmentation
├── model_utils.py         # Model loading utilities
├── class_counter.py       # Dataset analysis tool
└── visualize.py           # Visualization utilities
```

---

## Import Examples

### From Root Directory Scripts
```python
# Import dataset utilities
from utils.dataloader import StreetHazardsDataset, get_transforms, CLASS_NAMES

# Import model utilities
from utils.model_utils import load_model

# Import visualization
from utils.visualize import color, COLORS
```

### From Subdirectories (anomaly_detection/, models/)
```python
# Same imports work due to Python path
from utils.dataloader import StreetHazardsDataset, get_transforms
from utils.model_utils import load_model
```

---

## Configuration Integration

All utilities are designed to work seamlessly with `config.py`:

```python
from config import (
    DEVICE,
    MODEL_PATH,
    MODEL_ARCHITECTURE,
    NUM_CLASSES,
    ANOMALY_CLASS_IDX,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    TRAIN_ROOT,
    TEST_ROOT
)
from utils.dataloader import StreetHazardsDataset, get_transforms
from utils.model_utils import load_model

# Load model
model = load_model(MODEL_PATH, DEVICE, NUM_CLASSES, MODEL_ARCHITECTURE)

# Create dataset
img_t, mask_t = get_transforms('train', IMAGE_SIZE)
dataset = StreetHazardsDataset(TRAIN_ROOT, 'training', img_t, mask_t)
```

---

## Key Constants

### From `dataloader.py`

```python
# Class Configuration
NUM_CLASSES = 13           # Classes 0-12 (known classes)
ANOMALY_CLASS_IDX = 13     # Class 13 (anomaly, test only)

# Class Names (14 total: 0-13)
CLASS_NAMES = [
    'unlabeled', 'building', 'fence', 'other', 'pedestrian',
    'pole', 'road line', 'road', 'sidewalk', 'vegetation',
    'car', 'wall', 'traffic sign', 'anomaly'
]

# ImageNet Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

---

## Multi-Scale Augmentation Strategy

The dataloader implements a sophisticated multi-scale augmentation strategy that achieved **50.26% mIoU**:

### Training Pipeline
1. **Random Horizontal Flip** (p=0.5)
   - Horizontal symmetry augmentation

2. **Random Scale** (0.5× to 2.0×)
   - Multi-scale training
   - Handles objects at different sizes
   - Critical for robust features

3. **Random Crop** (512×512)
   - Focuses on local features
   - Provides diversity in training samples

4. **Color Jitter**
   - Brightness: ±20%
   - Contrast: ±20%
   - Saturation: ±20%
   - Hue: ±10%
   - Improves robustness to lighting variations

5. **Normalization** (ImageNet stats)
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

### Why This Works
- **Scale diversity**: 0.5-2.0× range covers wide variety of object sizes
- **Spatial diversity**: Random crops provide many training samples per image
- **Color robustness**: Jitter handles different lighting/weather conditions
- **Consistency**: Same augmentations for image and mask (geometric transforms)

---

## Design Principles

1. **Modularity**: Each utility is self-contained and reusable
2. **Configuration-driven**: Uses `config.py` for all parameters
3. **Error handling**: Robust to corrupted images and edge cases
4. **Documentation**: Comprehensive docstrings for all functions
5. **Performance**: Efficient implementations with proper batching
6. **Flexibility**: Supports multiple architectures and configurations

---

## Testing

All utilities can be tested independently:

```bash
# Test dataloader
.venv/bin/python3 -c "from utils.dataloader import StreetHazardsDataset, get_transforms; print('✓ Dataloader OK')"

# Test model utils
.venv/bin/python3 -c "from utils.model_utils import load_model; print('✓ Model utils OK')"

# Test visualize
.venv/bin/python3 -c "from utils.visualize import color, COLORS; print('✓ Visualize OK')"

# Run class counter
.venv/bin/python3 utils/class_counter.py
```

---

## Dependencies

All utilities require:
- PyTorch (torch, torchvision)
- PIL (Pillow)
- NumPy
- tqdm (progress bars)
- matplotlib (class_counter visualizations)

Install via:
```bash
pip install -r requirements.txt
```

---

## Notes

- **Memory efficiency**: Dataloader uses lazy loading (loads images on-the-fly)
- **Thread safety**: All utilities are thread-safe for DataLoader workers
- **GPU acceleration**: Model loading automatically uses CUDA if available
- **Error handling**: Graceful degradation on corrupted images
- **Caching**: No caching - fresh data every epoch to maximize augmentation diversity
