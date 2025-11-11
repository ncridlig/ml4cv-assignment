"""
SegFormer-B5 Training Script for StreetHazards Semantic Segmentation

SegFormer advantages over DeepLabV3+:
- Better performance (84.0% mIoU on Cityscapes vs ~70% for DeepLabV3+)
- Zero-shot robustness (excellent domain shift handling)
- No positional encoding (resolution-agnostic)
- Efficient transformer-based architecture
- Expected mIoU improvement: 37.57% → 45-50%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import SegformerForSemanticSegmentation
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

print("="*60)
print("SEGFORMER-B5 TRAINING")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LR}")
print(f"Epochs: {EPOCHS}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Number of classes: 13 (ignoring anomaly class 13)")
print("="*60)

# Note: NUM_CLASSES in training is 14 (includes anomaly class), but we ignore it
NUM_CLASSES_TRAIN = 13  # Train on 0-12, ignore 13 (anomaly)

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
writer = SummaryWriter(log_dir="runs/segformer_b5_streethazards")

# -----------------------------
# MODEL
# -----------------------------
print("\nInitializing SegFormer-B5 model...")
print("Loading pretrained weights from nvidia/segformer-b5-finetuned-ade-640-640...")

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640",
    num_labels=NUM_CLASSES_TRAIN,
    ignore_mismatched_sizes=True,  # Allow different number of classes
)

model.to(DEVICE)
print(f"✅ SegFormer-B5 loaded successfully")
print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# -----------------------------
# LOSS, OPTIMIZER, SCHEDULER
# -----------------------------
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

print(f"\n✅ Optimizer: AdamW (lr={LR}, weight_decay=0.01)")
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
    total_iou = 0.0

    pbar = tqdm(loader, desc=f"Training Epoch {epoch}")
    for i, (images, masks, _) in enumerate(pbar):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()

        # SegFormer forward pass
        # Returns SegformerForSemanticSegmentationOutput with .logits attribute
        outputs = model(pixel_values=images)
        logits = outputs.logits  # Shape: (B, num_classes, H, W)

        # Upsample logits to match mask size if needed
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        # Compute IoU for monitoring
        with torch.no_grad():
            iou = compute_iou(logits, masks)

        total_loss += loss.item()
        total_iou += iou

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.4f}'
        })

        if (i + 1) % PRINT_FREQ == 0:
            avg_loss = total_loss / (i + 1)
            avg_iou = total_iou / (i + 1)
            print(f"  Batch [{i+1}/{len(loader)}] Loss: {loss.item():.4f}, IoU: {iou:.4f}, Avg Loss: {avg_loss:.4f}, Avg IoU: {avg_iou:.4f}")

    return total_loss / len(loader), total_iou / len(loader)


def validate(model, loader, loss_fn, epoch):
    model.eval()
    total_loss, total_iou = 0.0, 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validation Epoch {epoch}")
        for images, masks, _ in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            # SegFormer forward pass
            outputs = model(pixel_values=images)
            logits = outputs.logits

            # Upsample logits to match mask size if needed
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = nn.functional.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            loss = loss_fn(logits, masks)
            iou = compute_iou(logits, masks)

            total_loss += loss.item()
            total_iou += iou

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}'
            })

    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)

    print(f"\n{'='*60}")
    print(f"Validation Results - Epoch {epoch}")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  mIoU: {avg_iou:.4f} ({avg_iou*100:.2f}%)")
    print(f"{'='*60}\n")

    return avg_iou, avg_loss


def save_best_model(model, miou, best_miou, base_name="models/segformer_b5_streethazards"):
    """
    Save best model with timestamp in filename format: _HH_MM_DAY-MONTH-YY_mIoU_XXXX
    Example: segformer_b5_streethazards_14_30_07-11-25_mIoU_4523.pth
    """
    if miou > best_miou:
        # Generate timestamp: _HH_MM_DAY-MONTH-YY
        now = datetime.now()
        timestamp = now.strftime("_%H_%M_%d-%m-%y")

        # Format mIoU as integer (e.g., 0.4523 → 4523)
        miou_str = f"_mIoU_{int(miou * 10000):04d}"
        path = f"{base_name}{timestamp}{miou_str}.pth"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model state dict
        torch.save(model.state_dict(), path)

        print(f"\n🎉 {'='*60}")
        print(f"🎉 NEW BEST MODEL!")
        print(f"🎉 mIoU improved: {best_miou:.4f} → {miou:.4f} (+{(miou-best_miou):.4f})")
        print(f"🎉 Saved to: {path}")
        print(f"🎉 {'='*60}\n")

        return miou
    return best_miou

# -----------------------------
# MAIN TRAINING LOOP
# -----------------------------
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

best_miou = 0.0

for epoch in range(1, EPOCHS + 1):
    print(f"\n{'#'*60}")
    print(f"# EPOCH {epoch}/{EPOCHS}")
    print(f"# Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"# Best mIoU so far: {best_miou:.4f}")
    print(f"{'#'*60}\n")

    # Training
    train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
    print(f"\nTraining Summary - Epoch {epoch}:")
    print(f"  Average Loss: {train_loss:.4f}")
    print(f"  Average IoU: {train_iou:.4f}")

    # Validation
    val_iou, val_loss = validate(model, val_loader, criterion, epoch)

    # Step scheduler based on validation mIoU
    scheduler.step(val_iou)

    # Save best model
    best_miou = save_best_model(model, val_iou, best_miou)

    # TensorBoard logging
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("IoU/train", train_iou, epoch)
    writer.add_scalar("mIoU/val", val_iou, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    # Log per-epoch comparison
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"  Train IoU:  {train_iou:.4f} | Val mIoU:  {val_iou:.4f}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

writer.close()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Best validation mIoU: {best_miou:.4f} ({best_miou*100:.2f}%)")
print(f"Model saved in: models/")
print(f"TensorBoard logs: runs/segformer_b5_streethazards/")
print("="*60 + "\n")

# -----------------------------
# SAVE TRAINING SUMMARY LOG
# -----------------------------
summary_path = "assets/segformer_b5_training_summary.txt"
os.makedirs(os.path.dirname(summary_path), exist_ok=True)

with open(summary_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("SEGFORMER-B5 TRAINING SUMMARY\n")
    f.write("="*60 + "\n\n")

    f.write("TRAINING CONFIGURATION\n")
    f.write("-"*60 + "\n")
    f.write(f"Model Architecture: SegFormer-B5\n")
    f.write(f"Pretrained Weights: nvidia/segformer-b5-finetuned-ade-640-640\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Image Size: {IMAGE_SIZE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Learning Rate: {LR}\n")
    f.write(f"Weight Decay: 0.01\n")
    f.write(f"Optimizer: AdamW\n")
    f.write(f"Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)\n")
    f.write(f"Epochs Trained: {EPOCHS}\n")
    f.write(f"Number of Classes: 13 (0-12, ignoring anomaly class 13)\n")
    f.write(f"Loss Function: CrossEntropyLoss (ignore_index={IGNORE_INDEX})\n\n")

    f.write("DATA AUGMENTATION\n")
    f.write("-"*60 + "\n")
    f.write(f"Training Augmentations:\n")
    f.write(f"  - Resize to {IMAGE_SIZE}\n")
    f.write(f"  - RandomHorizontalFlip(p=0.5)\n")
    f.write(f"  - ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)\n")
    f.write(f"  - Normalize (ImageNet stats)\n\n")

    f.write("DATASET STATISTICS\n")
    f.write("-"*60 + "\n")
    f.write(f"Training Samples: {len(train_dataset)}\n")
    f.write(f"Validation Samples: {len(val_dataset)}\n")
    f.write(f"Training Batches: {len(train_loader)}\n")
    f.write(f"Validation Batches: {len(val_loader)}\n\n")

    f.write("FINAL RESULTS\n")
    f.write("-"*60 + "\n")
    f.write(f"Best Validation mIoU: {best_miou:.4f} ({best_miou*100:.2f}%)\n")
    f.write(f"Final Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n\n")

    f.write("OUTPUT FILES\n")
    f.write("-"*60 + "\n")
    f.write(f"Model Checkpoints: models/segformer_b5_streethazards_*.pth\n")
    f.write(f"TensorBoard Logs: runs/segformer_b5_streethazards/\n")
    f.write(f"Training Summary: {summary_path}\n\n")

    f.write("COMPARISON WITH BASELINE\n")
    f.write("-"*60 + "\n")
    f.write(f"DeepLabV3+ ResNet50 @ 512x512: 37.57% mIoU (baseline)\n")
    f.write(f"SegFormer-B5 @ 512x512:        {best_miou*100:.2f}% mIoU (this run)\n")
    f.write(f"Improvement: {(best_miou*100 - 37.57):.2f}% absolute\n\n")

    f.write("="*60 + "\n")
    f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n")

print(f"📝 Training summary saved to: {summary_path}")
