"""
Model utilities for loading and managing segmentation models.
"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50
from transformers import SegformerForSemanticSegmentation


def load_model(model_path, device, num_classes=13, architecture='deeplabv3_resnet50'):
    """
    Load trained segmentation model.

    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on (cpu/cuda)
        num_classes: Number of output classes (default: 13 for StreetHazards)
        architecture: Model architecture to use. Options:
            - 'deeplabv3_resnet50' (default)
            - 'deeplabv3_resnet101'
            - 'fcn_resnet50'
            - 'segformer_b5'
            - 'hiera_large_224'

    Returns:
        model: Loaded segmentation model in eval mode
    """
    print(f"Loading model from {model_path}...")
    print(f"Architecture: {architecture}")

    # Create model architecture
    if architecture == 'deeplabv3_resnet50':
        model = deeplabv3_resnet50(weights=None)
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    elif architecture == 'deeplabv3_resnet101':
        model = deeplabv3_resnet101(weights=None)
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    elif architecture == 'fcn_resnet50':
        model = fcn_resnet50(weights=None)
        model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=1)
    elif architecture == 'segformer_b5':
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # Allow different number of classes
        )
    elif architecture == 'hiera_large_224':
        # Import HieraSegmentation from training script
        from models.training_scripts.hiera_large_224 import HieraSegmentation

        # Create the full segmentation model (backbone + decode head)
        model = HieraSegmentation(
            backbone_name='hiera_large_224',
            num_classes=num_classes,
            pretrained=False  # We'll load weights from checkpoint
        )
        print("✅ Hiera segmentation model initialized")
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. "
                        f"Choose from: deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, segformer_b5, hiera_large_224")

    # Load checkpoint (strict=False to ignore aux_classifier if present)
    # The aux_classifier is used during training but not during inference
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")

    return model
