"""
Hiera Training Script for StreetHazards Semantic Segmentation

Hiera: Hierarchical Vision Transformer (Meta AI Research)
- Paper: "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles"
- Fast, powerful, and simple hierarchical vision transformer
- MAE-based spatial biases (no positional encoding needed)
- Excellent speed-accuracy tradeoff

Advantages over DeepLabV3+ and SegFormer:
- Faster inference than SegFormer (2211 im/s vs ~1000 im/s for SegFormer)
- Simple architecture without complex modules
- Strong hierarchical features for segmentation
- Optimized for 16GB VRAM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import StreetHazardsDataset, get_transforms
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
# OVERRIDE IMAGE SIZE FOR HIERA PRETRAINED MODEL
# -----------------------------
# Hiera pretrained models (hiera_large_224)
# are trained on 224×224 images. We must match this resolution
# to use pretrained weights properly.
IMAGE_SIZE = (224, 224)  # Override config.py IMAGE_SIZE - MUST BE TUPLE!
NUM_CLASSES_TRAIN = 13  # Train on 0-12, ignore 13 (anomaly)

if __name__ == "__main__":
    print("="*60)
    print("HIERA LARGE SEMANTIC SEGMENTATION TRAINING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} (Hiera Large pretrained resolution)")
    print(f"Number of classes: 13 (ignoring anomaly class 13)")
    print(f"GPU: Optimized for 16GB VRAM")
    print("="*60)

# -----------------------------
# HIERA SEGMENTATION MODEL
# -----------------------------
class HieraSegmentationHead(nn.Module):
    """
    Lightweight segmentation decoder for Hiera.

    Takes multi-scale features from Hiera encoder and progressively upsamples
    to produce full-resolution segmentation masks.

    Architecture inspired by SegFormer's MLP decoder (simple and efficient).
    """
    def __init__(self, in_channels_list, num_classes=13, embed_dim=256):
        """
        Args:
            in_channels_list: List of channel dimensions from Hiera stages
            num_classes: Number of segmentation classes
            embed_dim: Intermediate embedding dimension
        """
        super().__init__()

        # Project each stage to common embedding dimension
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, embed_dim, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Fusion module: combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        )

    def forward(self, features):
        """
        Args:
            features: List of [B, H, W, C] tensors from Hiera stages (NOTE: channels last!)

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Convert from Hiera format [B, H, W, C] to PyTorch format [B, C, H, W]
        features = [f.permute(0, 3, 1, 2) for f in features]  # [B, H, W, C] -> [B, C, H, W]

        # Target size (use largest feature map as reference)
        target_size = features[0].shape[-2:]

        # Project and upsample all features to target size
        upsampled_features = []
        for feat, proj in zip(features, self.projections):
            # Project to common dimension
            feat = proj(feat)
            # Upsample to target size
            if feat.shape[-2:] != target_size:
                feat = nn.functional.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            upsampled_features.append(feat)

        # Concatenate along channel dimension
        x = torch.cat(upsampled_features, dim=1)

        # Fuse features
        x = self.fusion(x)

        # Generate predictions
        x = self.head(x)

        return x


class HieraSegmentation(nn.Module):
    """Complete Hiera-based segmentation model."""

    def __init__(self, backbone_name='hiera_large_224', num_classes=13, pretrained=True):
        super().__init__()

        print(f"\nInitializing Hiera segmentation model...")
        print(f"  Backbone: {backbone_name}")
        print(f"  Pretrained: {pretrained}")
        print(f"  Num classes: {num_classes}")

        # Load Hiera backbone
        try:
            import hiera
        except ImportError:
            raise ImportError(
                "hiera-transformer not installed. Please run:\n"
                "  pip install hiera-transformer"
            )

        # Load pretrained Hiera
        if backbone_name == 'hiera_large_224':
            self.backbone = hiera.hiera_large_224(
                pretrained=pretrained,
                checkpoint="mae_in1k_ft_in1k" if pretrained else None
            )
            # Hiera Large: embed_dim=144, stages produce [144, 288, 576, 1152]
            stage_channels = [144, 288, 576, 1152]
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        print(f"✅ Backbone loaded: {backbone_name}")
        print(f"✅ Stage channels: {stage_channels}")

        # Create segmentation head
        self.decode_head = HieraSegmentationHead(
            in_channels_list=stage_channels,
            num_classes=num_classes,
            embed_dim=256
        )

        print(f"✅ Segmentation head created (embed_dim=256)")

    def forward(self, x):
        """
        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        input_size = x.shape[-2:]

        # Extract multi-scale features from Hiera
        _, intermediates = self.backbone(x, return_intermediates=True)

        # Decode to segmentation mask
        logits = self.decode_head(intermediates)

        # Upsample to input size
        if logits.shape[-2:] != input_size:
            logits = nn.functional.interpolate(
                logits, size=input_size, mode='bilinear', align_corners=False
            )

        return logits


# -----------------------------
# MAIN - Only run training when executed as script
# -----------------------------
if __name__ == "__main__":
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
        mask_transform=train_mask_transform,
        image_size=IMAGE_SIZE  # Pass 224x224 override
    )
    val_dataset = StreetHazardsDataset(
        root_dir=TRAIN_ROOT,
        split='validation',
        transform=val_test_transform,
        mask_transform=val_test_mask_transform,
        image_size=IMAGE_SIZE  # Pass 224x224 override
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
    writer = SummaryWriter(log_dir="models/runs/hiera_large_streethazards")
    
    # -----------------------------
    # MODEL
    # -----------------------------
    # Use Hiera-Large for more accuracy
    model = HieraSegmentation(
        backbone_name='hiera_large_224', 
        num_classes=NUM_CLASSES,
        pretrained=True
    )
    
    model.to(DEVICE)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\n✅ Model loaded to {DEVICE}")
    print(f"✅ Total parameters: {total_params:.1f}M")
    print(f"✅ Trainable parameters: {trainable_params:.1f}M")
    
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
    
            # Forward pass
            logits = model(images)
    
            # Compute loss
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
    
                # Forward pass
                logits = model(images)
    
                # Compute loss and IoU
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
    
    
    def save_best_model(model, miou, best_miou, base_name="models/checkpoints/hiera_large_cropaug_streethazards"):
        """
        Save best model with timestamp in filename format: _HH_MM_DAY-MONTH-YY_mIoU_XXXX
        Example: hiera_large_streethazards_14_30_07-11-25_mIoU_4523.pth
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
    print(f"TensorBoard logs: models/runs/hiera_large_streethazards/")
    print("="*60 + "\n")
