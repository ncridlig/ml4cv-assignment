#!/usr/bin/env python3
"""
Simplified Augmentation Ablation Study
Reuses existing training infrastructure from models/training_scripts/deeplabv3plus_resnet50.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from pathlib import Path
import json
import time
import numpy as np
from tqdm import tqdm

# Project imports
from utils.dataloader import StreetHazardsDataset
from config import DEVICE, BATCH_SIZE, IMAGE_SIZE, TRAIN_ROOT, IGNORE_INDEX, NUM_WORKERS
from models.training_scripts.deeplabv3plus_resnet50 import train_one_epoch, compute_iou, validate, save_best_model


def train_config(aug_config, config_name, max_epochs=40, patience=3):
    """Train a single augmentation configuration"""
    print(f"\n{'='*80}")
    print(f"Training: {config_name}")
    print(f"{'='*80}\n")

    # Load datasets
    train_dataset = StreetHazardsDataset(TRAIN_ROOT, 'training', None, None, IMAGE_SIZE, aug_config)
    val_dataset = StreetHazardsDataset(TRAIN_ROOT, 'validation', None, None, IMAGE_SIZE, aug_config)

    # DataLoaders
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)
    model.aux_classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)
    model = model.to(DEVICE)

    # Optimizer & Loss 
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop with early stopping
    best_val_miou = 0
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")

        start = time.time()
        train_loss, main_loss, aux_loss = train_one_epoch(model, train_loader, optimizer, criterion, max_batches=None)
        val_miou = validate(model, val_loader, criterion, max_batches=None)
        epoch_time = time.time() - start

        print(f"Train Loss: {train_loss:.4f}, Aux Loss: {aux_loss:.4f}")
        print(f"Val mIou: {val_miou:.4f}")
        print(f"Time: {epoch_time:.1f}s")

        # Early stopping
        if val_miou > best_val_miou:
            best_val_miou = save_best_model(model, val_miou, best_val_miou, base_name=f"ablation_study/checkpoints/{config_name}_")
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f"✓ New best validation mIoU: {best_val_miou:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"\n⚠ Early stopping at epoch {epoch + 1}")
            break

    

    return {
        'best_val_miou': best_val_miou,
        'best_epoch': best_epoch,
        'final_epoch': epoch + 1,
        'stopped_early': epochs_no_improve >= patience
    }


def main():
    """Run ablation study"""
    # Augmentation configurations
    configs = {
        'No_Aug': {},
        '+Scale': {'scale': True},
        '+Scale+Rotate': {'scale': True, 'rotate': True},
        '+Scale+Rotate+Flip': {'scale': True, 'rotate': True, 'flip': True},
        '+Scale+Rotate+Flip+Color': {'scale': True, 'rotate': True, 'flip': True, 'color': True}
    }

    results = {}

    for name, config in configs.items():
        try:
            result = train_config(config, name)
            results[name] = result
            print(f"\n✅ Completed {name}: Val mIoU = {result['best_val_miou']:.4f}")
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            continue

    # Save results
    output_dir = Path('ablation_study/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'augmentation_ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
