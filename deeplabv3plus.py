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

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 14           # 0–12 normal, 13 = anomaly (ignored in training)
IGNORE_INDEX = 13
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 20
PRINT_FREQ = 500
NUM_WORKERS = 4

# -----------------------------
# DATASETS
# -----------------------------
train_transform, train_mask_transform = get_transforms(512, is_training=True)
val_test_transform, val_test_mask_transform = get_transforms(512, is_training=False)

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True) # drop last avoids bath norm issue over singleton image
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# -----------------------------
# TENSORBOARD SETUP
# -----------------------------
writer = SummaryWriter(log_dir="runs/streethazards_experiment")

# -----------------------------
# MODEL
# -----------------------------
model = deeplabv3_resnet50(weights='DEFAULT')
# 13 normal classes + 1 anomaly class → ignore anomaly (index 13)
# Reconfigure the pretrained DeepLab model to predict 13 segmentation classes instead of the 21 COCO classes
model.classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)

# IMPORTANT: Also modify auxiliary classifier to predict 13 classes
# The aux_classifier helps with gradient flow during training
model.aux_classifier[-1] = nn.Conv2d(256, 13, kernel_size=1)

model.to(DEVICE)

# -----------------------------
# LOSS, OPTIMIZER, SCHEDULER
# -----------------------------
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

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
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    for i, (images, masks, _) in enumerate(tqdm(loader, desc="Training")):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()

        # Get both main and auxiliary outputs
        output_dict = model(images)
        main_output = output_dict['out']
        aux_output = output_dict['aux']

        # Compute losses
        main_loss = loss_fn(main_output, masks)
        aux_loss = loss_fn(aux_output, masks)

        # Combined loss (standard weight is 0.4 for auxiliary)
        loss = main_loss + 0.4 * aux_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item()
        if (i + 1) % PRINT_FREQ == 0:
            print(f"Iter [{i+1}/{len(loader)}] Loss: {loss.item():.4f} (main: {main_loss.item():.4f}, aux: {aux_loss.item():.4f})")
    return total_loss / len(loader), total_main_loss / len(loader), total_aux_loss / len(loader)


def validate(model, loader, loss_fn):
    model.eval()
    total_loss, total_iou = 0.0, 0.0
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Validating"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)['out']
            loss = loss_fn(outputs, masks)
            iou = compute_iou(outputs, masks)
            total_loss += loss.item()
            total_iou += iou
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    print(f"Validation Loss: {avg_loss:.4f}, mIoU: {avg_iou:.4f}")
    return avg_iou


def save_best_model(model, miou, best_miou, base_name="models/deeplabv3_"):
    """
    Save best model with timestamp in filename format: _HH_MM_DAY-MONTH-25
    Example: best_deeplabv3_streethazards_14_30_04-11-25.pth for Nov 4, 2025 at 14:30
    """
    if miou > best_miou:
        # Generate timestamp: _HH_MM_DAY-MONTH-25
        now = datetime.now()
        timestamp = now.strftime("_%H_%M_%d-%m-%y")  # Format: _14_30_04-11-25
        performance = f"_mIoU_{miou:.4f}"
        path = f"{base_name}{timestamp}{performance}.pth"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(model.state_dict(), path)
        print(f"✅ Saved new best model (mIoU={miou:.4f}) to: {path}")
        return miou
    return best_miou

# -----------------------------
# MAIN TRAINING LOOP
# -----------------------------
best_miou = 0.0

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_loss, main_loss, aux_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    miou = validate(model, val_loader, criterion)
    scheduler.step(miou)
    best_miou = save_best_model(model, miou, best_miou)

    # ---- TensorBoard logging ----
    writer.add_scalar("Loss/train_total", train_loss, epoch)
    writer.add_scalar("Loss/train_main", main_loss, epoch)
    writer.add_scalar("Loss/train_aux", aux_loss, epoch)
    writer.add_scalar("mIoU/val", miou, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

writer.close()
print("Training complete.")

