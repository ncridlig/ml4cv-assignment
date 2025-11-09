"""
DeepLabV3+ ResNet50 Training with Multi-Scale Augmentation

Following DeepLabV3+ paper literally with variable crop sizes:
- Multi-scale random crop (0.5-2.0x scale range) - KEY IMPROVEMENT
  * Scale 0.5x: crop 256×256 → resize 512×512 (zoom in, see details)
  * Scale 1.0x: crop 512×512 → resize 512×512 (normal view)
  * Scale 2.0x: crop 1024×1024 → resize 512×512 (zoom out, see context)
  * No black padding - crop size adapts to scale!
- Random rotation (±10 degrees) REMOVED TO AVOID BLACK EDGES
- Random horizontal flip
- Color jitter with hue variation
- Gaussian blur (50% probability)

Expected improvement over baseline: +2-5% mIoU
Baseline: 37.57% mIoU (ResNet50 @ 512x512, weak augmentation)
Target: 40-42% mIoU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataloader import StreetHazardsDataset, get_transforms
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

print("="*80)
print("DEEPLABV3+ RESNET50 - MULTI-SCALE AUGMENTED TRAINING")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LR}")
print(f"Epochs: {EPOCHS}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Number of classes: 13 (ignoring anomaly class 13)")
print("\nAugmentations (following DeepLabV3+ paper):")
print("  ✅ Multi-scale random crop with VARIABLE crop sizes:")
print("      - Scale 0.5x: crop 256×256 → resize 512×512 (zoom in)")
print("      - Scale 1.0x: crop 512×512 → resize 512×512 (normal)")
print("      - Scale 2.0x: crop 1024×1024 → resize 512×512 (zoom out)")
print("      - No black padding!")
print("  ✅ Random rotation (±10°)")
print("  ✅ Random horizontal flip (50%)")
print("  ✅ Color jitter (brightness, contrast, saturation, hue)")
print("  ✅ Gaussian blur (50% probability)")
print("="*80)

# Note: NUM_CLASSES in training is 14 (includes anomaly class), but we ignore it
NUM_CLASSES = 14  # Override: 0-12 normal, 13 = anomaly (ignored in training)

# -----------------------------
# DATASETS
# -----------------------------
print("\nLoading datasets...")
train_transform, train_mask_transform = get_transforms(IMAGE_SIZE, is_training=True)
val_test_transform, val_test_mask_transform = get_transforms(IMAGE_SIZE, is_training=False)

train_dataset = StreetHazardsDataset(
    root_dir=TRAIN_ROOT,
    split='training',
    transform=train_transform,
    mask_transform=train_mask_transform
)
val_dataset = StreetHazardsDataset(
    root_dir=TRAIN_ROOT,
    split='validation',
    transform=val_test_transform,
    mask_transform=val_test_mask_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(val_dataset)}")
print(f"✅ Training batches: {len(train_loader)}")
print(f"✅ Validation batches: {len(val_loader)}")

# -----------------------------
# TENSORBOARD SETUP
# -----------------------------
writer = SummaryWriter(log_dir="runs/resnet50_augmented")

# -----------------------------
# MODEL
# -----------------------------
print("\nInitializing DeepLabV3+ ResNet50...")
model = deeplabv3_resnet50(weights='DEFAULT')
# Reconfigure to predict 13 classes
model.classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)
model.aux_classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)
model.to(DEVICE)

print(f"✅ Model loaded with pretrained ImageNet weights")
print(f"✅ Modified for 13-class segmentation")

# -----------------------------
# LOSS, OPTIMIZER, SCHEDULER
# -----------------------------
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

print(f"\n✅ Optimizer: Adam (lr={LR})")
print(f"✅ Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
print(f"✅ Loss: CrossEntropyLoss (ignore_index={IGNORE_INDEX})")

# -----------------------------
# METRICS
# -----------------------------
def compute_iou(preds, labels, num_classes=13, ignore_index=13):
    """Compute mean IoU (ignoring anomaly class)."""
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0

# -----------------------------
# TRAIN / VALIDATE
# -----------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, epoch):
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    total_iou = 0.0

    pbar = tqdm(loader, desc=f"Training Epoch {epoch}")
    for i, (images, masks, _) in enumerate(pbar):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()

        # Get both main and auxiliary outputs
        output_dict = model(images)
        main_output = output_dict['out']
        aux_output = output_dict['aux']

        # Compute losses
        main_loss = loss_fn(main_output, masks)
        aux_loss = loss_fn(aux_output, masks)
        loss = main_loss + 0.4 * aux_loss

        loss.backward()
        optimizer.step()

        # Compute IoU for monitoring
        with torch.no_grad():
            iou = compute_iou(main_output, masks)

        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item()
        total_iou += iou

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'main': f'{main_loss.item():.4f}',
            'aux': f'{aux_loss.item():.4f}',
            'iou': f'{iou:.4f}'
        })

        if (i + 1) % PRINT_FREQ == 0:
            avg_loss = total_loss / (i + 1)
            avg_iou = total_iou / (i + 1)
            print(f"  Batch [{i+1}/{len(loader)}] Loss: {loss.item():.4f}, IoU: {iou:.4f}, Avg Loss: {avg_loss:.4f}, Avg IoU: {avg_iou:.4f}")

    return (total_loss / len(loader),
            total_main_loss / len(loader),
            total_aux_loss / len(loader),
            total_iou / len(loader))


def validate(model, loader, loss_fn, epoch):
    model.eval()
    total_loss, total_iou = 0.0, 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validation Epoch {epoch}")
        for images, masks, _ in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)['out']
            loss = loss_fn(outputs, masks)
            iou = compute_iou(outputs, masks)
            total_loss += loss.item()
            total_iou += iou

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}'
            })

    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)

    print(f"\n{'='*80}")
    print(f"Validation Results - Epoch {epoch}")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  mIoU: {avg_iou:.4f} ({avg_iou*100:.2f}%)")
    print(f"{'='*80}\n")

    return avg_iou, avg_loss


def save_best_model(model, miou, best_miou, base_name="models/deeplabv3_resnet50_augmented"):
    """Save best model with timestamp and performance."""
    if miou > best_miou:
        now = datetime.now()
        timestamp = now.strftime("_%H_%M_%d-%m-%y")
        miou_str = f"_mIoU_{int(miou * 10000):04d}"
        path = f"{base_name}{timestamp}{miou_str}.pth"

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)

        print(f"\n🎉 {'='*80}")
        print(f"🎉 NEW BEST MODEL!")
        print(f"🎉 mIoU improved: {best_miou:.4f} → {miou:.4f} (+{(miou-best_miou):.4f})")
        print(f"🎉 Saved to: {path}")
        print(f"🎉 {'='*80}\n")

        return miou
    return best_miou

# -----------------------------
# MAIN TRAINING LOOP
# -----------------------------
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80 + "\n")

best_miou = 0.0
training_history = []

for epoch in range(1, EPOCHS + 1):
    print(f"\n{'#'*80}")
    print(f"# EPOCH {epoch}/{EPOCHS}")
    print(f"# Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"# Best mIoU so far: {best_miou:.4f}")
    print(f"{'#'*80}\n")

    # Training
    train_loss, main_loss, aux_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
    print(f"\nTraining Summary - Epoch {epoch}:")
    print(f"  Total Loss: {train_loss:.4f}")
    print(f"  Main Loss: {main_loss:.4f}")
    print(f"  Aux Loss: {aux_loss:.4f}")
    print(f"  Train IoU: {train_iou:.4f}")

    # Validation
    val_iou, val_loss = validate(model, val_loader, criterion, epoch)

    # Step scheduler
    scheduler.step(val_iou)

    # Save best model
    best_miou = save_best_model(model, val_iou, best_miou)

    # TensorBoard logging
    writer.add_scalar("Loss/train_total", train_loss, epoch)
    writer.add_scalar("Loss/train_main", main_loss, epoch)
    writer.add_scalar("Loss/train_aux", aux_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("IoU/train", train_iou, epoch)
    writer.add_scalar("mIoU/val", val_iou, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    # Track history
    training_history.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_iou': train_iou,
        'val_loss': val_loss,
        'val_iou': val_iou,
        'lr': optimizer.param_groups[0]["lr"]
    })

    # Log per-epoch comparison
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"  Train IoU:  {train_iou:.4f} | Val mIoU:  {val_iou:.4f}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

writer.close()

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Best validation mIoU: {best_miou:.4f} ({best_miou*100:.2f}%)")
print(f"Model saved in: models/")
print(f"TensorBoard logs: runs/resnet50_augmented/")
print("="*80 + "\n")

# -----------------------------
# SAVE TRAINING SUMMARY
# -----------------------------
summary_path = "assets/resnet50_augmented_training_summary.txt"
os.makedirs(os.path.dirname(summary_path), exist_ok=True)

with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("DEEPLABV3+ RESNET50 - MULTI-SCALE AUGMENTED TRAINING SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write("TRAINING CONFIGURATION\n")
    f.write("-"*80 + "\n")
    f.write(f"Model Architecture: DeepLabV3+ ResNet50\n")
    f.write(f"Pretrained Weights: ImageNet (torchvision DEFAULT)\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Image Size: {IMAGE_SIZE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Learning Rate: {LR}\n")
    f.write(f"Optimizer: Adam\n")
    f.write(f"Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)\n")
    f.write(f"Epochs Trained: {EPOCHS}\n")
    f.write(f"Number of Classes: 13 (0-12, ignoring anomaly class 13)\n")
    f.write(f"Loss Function: CrossEntropyLoss (ignore_index={IGNORE_INDEX})\n")
    f.write(f"Auxiliary Classifier: Enabled (weight=0.4)\n\n")

    f.write("DATA AUGMENTATION (ENHANCED - Following DeepLabV3+ Paper)\n")
    f.write("-"*80 + "\n")
    f.write("Training Augmentations:\n")
    f.write("  ✅ Multi-scale random crop with VARIABLE crop sizes:\n")
    f.write("      - Scale 0.5x: crop 256×256 → resize to 512×512 (zoom in, fine details)\n")
    f.write("      - Scale 1.0x: crop 512×512 → resize to 512×512 (normal view)\n")
    f.write("      - Scale 2.0x: crop 1024×1024 → resize to 512×512 (zoom out, context)\n")
    f.write("      - No black padding - crop size adapts to scale!\n")
    f.write("  ✅ Random rotation (±10 degrees)\n")
    f.write("  ✅ Random horizontal flip (p=0.5)\n")
    f.write("  ✅ Color jitter (brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)\n")
    f.write("  ✅ Gaussian blur (p=0.5, kernel=5, sigma=0.1-2.0)\n")
    f.write("  ✅ ImageNet normalization\n\n")

    f.write("DATASET STATISTICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Training Samples: {len(train_dataset)}\n")
    f.write(f"Validation Samples: {len(val_dataset)}\n")
    f.write(f"Training Batches: {len(train_loader)}\n")
    f.write(f"Validation Batches: {len(val_loader)}\n\n")

    f.write("FINAL RESULTS\n")
    f.write("-"*80 + "\n")
    f.write(f"Best Validation mIoU: {best_miou:.4f} ({best_miou*100:.2f}%)\n")
    f.write(f"Final Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n\n")

    f.write("COMPARISON WITH BASELINE\n")
    f.write("-"*80 + "\n")
    f.write(f"Baseline (weak augmentation):  37.57% mIoU\n")
    f.write(f"Augmented (this run):           {best_miou*100:.2f}% mIoU\n")
    f.write(f"Improvement:                    {(best_miou*100 - 37.57):.2f}% absolute\n\n")

    f.write("TRAINING HISTORY\n")
    f.write("-"*80 + "\n")
    f.write("Epoch | Train Loss | Train IoU | Val Loss | Val mIoU | LR\n")
    f.write("-"*80 + "\n")
    for h in training_history:
        f.write(f"{h['epoch']:5d} | {h['train_loss']:10.4f} | {h['train_iou']:9.4f} | "
                f"{h['val_loss']:8.4f} | {h['val_iou']:8.4f} | {h['lr']:.6f}\n")

    f.write("\n" + "="*80 + "\n")
    f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n")

print(f"📝 Training summary saved to: {summary_path}")
print("\n" + "="*80)
print("Ready for evaluation and anomaly detection!")
print("="*80 + "\n")
