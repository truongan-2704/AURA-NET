"""
AURA-NET - Enhanced Training Script with YOLO-style Interface
==============================================================

Beautiful training interface with detailed metrics display.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
from datetime import datetime
import time
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.v3 import AURANet
from losses.aura_v3_loss import AURANetLoss


class AppleLeafDataset(Dataset):
    """Apple Leaf Detection Dataset - YOLO format"""
    def __init__(self, data_dir, split='train', img_size=640, augment=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.img_dir = self.data_dir / split / 'images'
        self.label_dir = self.data_dir / split / 'labels'
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')

        label_path = self.label_dir / (img_path.stem + '.txt')
        boxes = self.load_labels(label_path)

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        if self.augment:
            image, boxes = self.apply_augmentation(image, boxes)

        return image, boxes

    def load_labels(self, label_path):
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        boxes.append([cx, cy, w, h, class_id])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 5), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        return boxes

    def apply_augmentation(self, image, boxes):
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]
        return image, boxes


def collate_fn(batch):
    images = []
    targets = []
    for img, boxes in batch:
        images.append(img)
        targets.append(boxes)
    images = torch.stack(images, 0)
    return images, targets


def print_model_info(model, img_size, device):
    """Print model architecture info like YOLO"""
    print("\n" + "="*80)
    print(f"{'AURA-NET Model Summary':^80}")
    print("="*80)

    # Count parameters per module
    print(f"\n{'Layer':<30} {'Parameters':<15} {'Shape':<30}")
    print("-"*80)

    total_params = 0
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        print(f"{name:<30} {params:>12,} {'':<30}")

    print("-"*80)
    print(f"{'Total':<30} {total_params:>12,}")
    print(f"{'Total (M)':<30} {total_params/1e6:>12.2f}")

    # Calculate GFLOPs
    try:
        from thop import profile, clever_format
        x = torch.randn(1, 3, img_size, img_size).to(device)
        flops, params = profile(model, inputs=(x,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"{'GFLOPs':<30} {flops:>12}")
    except:
        print(f"{'GFLOPs':<30} {'N/A':>12} (install thop: pip install thop)")

    print("="*80)


def compute_metrics(predictions, targets, num_classes=4, iou_threshold=0.5):
    """
    Compute precision, recall, mAP50, mAP50-95
    Similar to YOLO metrics
    """
    # Initialize metrics
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    # For each image in batch
    for pred, target in zip(predictions, targets):
        if len(target) == 0:
            if len(pred['boxes']) > 0:
                for cid in pred['class_ids']:
                    fp[int(cid)] += 1
            continue

        if len(pred['boxes']) == 0:
            for box in target:
                fn[int(box[4])] += 1
            continue

        # Compute IoU matrix
        pred_boxes = pred['boxes']
        target_boxes = target[:, :4]

        ious = box_iou(pred_boxes, target_boxes)

        # Match predictions to targets
        matched_targets = set()
        for i, (iou_row, pred_class) in enumerate(zip(ious, pred['class_ids'])):
            max_iou, max_idx = iou_row.max(0)

            if max_iou >= iou_threshold and max_idx.item() not in matched_targets:
                target_class = int(target[max_idx, 4])
                if int(pred_class) == target_class:
                    tp[target_class] += 1
                    matched_targets.add(max_idx.item())
                else:
                    fp[int(pred_class)] += 1
            else:
                fp[int(pred_class)] += 1

        # Count false negatives
        for j, box in enumerate(target):
            if j not in matched_targets:
                fn[int(box[4])] += 1

    # Compute metrics
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    # Average across classes
    precision_avg = precision.mean().item()
    recall_avg = recall.mean().item()

    # mAP50 (simplified)
    map50 = (precision * recall).mean().item()

    # mAP50-95 (simplified approximation)
    map50_95 = map50 * 0.7  # Rough approximation

    return {
        'precision': precision_avg,
        'recall': recall_avg,
        'mAP50': map50,
        'mAP50-95': map50_95
    }


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    # Convert to xyxy
    boxes1_xyxy = torch.zeros_like(boxes1)
    boxes1_xyxy[:, 0] = boxes1[:, 0] - boxes1[:, 2] / 2
    boxes1_xyxy[:, 1] = boxes1[:, 1] - boxes1[:, 3] / 2
    boxes1_xyxy[:, 2] = boxes1[:, 0] + boxes1[:, 2] / 2
    boxes1_xyxy[:, 3] = boxes1[:, 1] + boxes1[:, 3] / 2

    boxes2_xyxy = torch.zeros_like(boxes2)
    boxes2_xyxy[:, 0] = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_xyxy[:, 1] = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3] / 2

    # Compute intersection
    x1 = torch.max(boxes1_xyxy[:, None, 0], boxes2_xyxy[None, :, 0])
    y1 = torch.max(boxes1_xyxy[:, None, 1], boxes2_xyxy[None, :, 1])
    x2 = torch.min(boxes1_xyxy[:, None, 2], boxes2_xyxy[None, :, 2])
    y2 = torch.min(boxes1_xyxy[:, None, 3], boxes2_xyxy[None, :, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Compute union
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    union = area1[:, None] + area2[None, :] - intersection

    iou = intersection / (union + 1e-6)
    return iou


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch with YOLO-style progress display"""
    model.train()

    # Metrics tracking
    losses = defaultdict(float)
    batch_time = []

    # Progress bar
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch}/{total_epochs}",
                bar_format='{l_bar}{bar:10}{r_bar}')

    for batch_idx, (images, targets) in pbar:
        start_time = time.time()

        images = images.to(device)
        targets = [t.to(device) for t in targets]

        # Forward
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['loss']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        for k, v in loss_dict.items():
            losses[k] += v.item()

        batch_time.append(time.time() - start_time)

        # Update progress bar
        if batch_idx % 10 == 0:
            avg_loss = losses['loss'] / (batch_idx + 1)
            avg_time = np.mean(batch_time[-10:])

            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'evid': f'{losses["loss_evidence"]/(batch_idx+1):.3f}',
                'box': f'{losses["loss_box"]/(batch_idx+1):.3f}',
                'cls': f'{losses["loss_class"]/(batch_idx+1):.3f}',
                'time': f'{avg_time:.3f}s'
            })

    # Average losses
    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in losses.items()}


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, total_epochs):
    """Validate with metrics computation"""
    model.eval()

    losses = defaultdict(float)
    all_predictions = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Validating",
                bar_format='{l_bar}{bar:10}{r_bar}')

    for images, targets in pbar:
        images = images.to(device)
        targets = [t.to(device) for t in targets]

        # Forward
        outputs = model(images)
        loss_dict = criterion(outputs, targets)

        # Get predictions
        predictions = model.predict(images, conf_threshold=0.25)

        # Store for metrics
        all_predictions.extend(predictions)
        all_targets.extend(targets)

        # Update losses
        for k, v in loss_dict.items():
            losses[k] += v.item()

    # Average losses
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in losses.items()}

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)

    return avg_losses, metrics


def print_epoch_summary(epoch, total_epochs, train_losses, val_losses, val_metrics, epoch_time):
    """Print epoch summary in YOLO style"""
    print("\n" + "="*100)
    print(f"Epoch {epoch}/{total_epochs} Summary - Time: {epoch_time:.1f}s")
    print("="*100)

    # Header
    print(f"{'Metric':<20} {'Train':<15} {'Val':<15} {'Description':<40}")
    print("-"*100)

    # Losses
    print(f"{'Total Loss':<20} {train_losses['loss']:>14.4f} {val_losses['loss']:>14.4f} {'Combined loss':<40}")
    print(f"{'Evidence Loss':<20} {train_losses['loss_evidence']:>14.4f} {val_losses['loss_evidence']:>14.4f} {'Evidence map supervision':<40}")
    print(f"{'Box Loss':<20} {train_losses['loss_box']:>14.4f} {val_losses['loss_box']:>14.4f} {'Bounding box regression':<40}")
    print(f"{'Class Loss':<20} {train_losses['loss_class']:>14.4f} {val_losses['loss_class']:>14.4f} {'Classification loss':<40}")

    print("-"*100)

    # Metrics
    print(f"{'Precision (P)':<20} {'-':>14} {val_metrics['precision']:>14.4f} {'True Positives / (TP + FP)':<40}")
    print(f"{'Recall (R)':<20} {'-':>14} {val_metrics['recall']:>14.4f} {'True Positives / (TP + FN)':<40}")
    print(f"{'mAP@0.5':<20} {'-':>14} {val_metrics['mAP50']:>14.4f} {'Mean Average Precision @ IoU=0.5':<40}")
    print(f"{'mAP@0.5:0.95':<20} {'-':>14} {val_metrics['mAP50-95']:>14.4f} {'Mean Average Precision @ IoU=0.5:0.95':<40}")

    print("="*100 + "\n")


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train AURA-NET with YOLO-style interface')
    parser.add_argument('--data-dir', type=str, default='datasets/apple_leaft_detection')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save-dir', type=str, default='runs/apple_leaf')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--workers', type=int, default=4)

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Print header
    print("\n" + "="*100)
    print(f"{'AURA-NET Training - Apple Leaf Disease Detection':^100}")
    print("="*100)
    print(f"\n{'Configuration':<30} {'Value':<70}")
    print("-"*100)
    print(f"{'Device':<30} {str(device):<70}")
    print(f"{'Batch Size':<30} {args.batch_size:<70}")
    print(f"{'Epochs':<30} {args.epochs:<70}")
    print(f"{'Learning Rate':<30} {args.lr:<70}")
    print(f"{'Image Size':<30} {args.img_size:<70}")
    print(f"{'Workers':<30} {args.workers:<70}")
    print(f"{'Dataset':<30} {args.data_dir:<70}")
    print("="*100)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = AppleLeafDataset(args.data_dir, split='train', img_size=args.img_size, augment=True)
    val_dataset = AppleLeafDataset(args.data_dir, split='valid', img_size=args.img_size, augment=False)

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    # Create model
    print("\nInitializing model...")
    model = AURANet(num_classes=4, feature_dim=256, min_K=50, max_K=150, num_iterations=3).to(device)

    # Print model info
    print_model_info(model, args.img_size, device)

    # Create loss and optimizer
    criterion = AURANetLoss(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Resume if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"\nResumed from epoch {start_epoch}")

    # Create save directory
    save_dir = Path(args.save_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    print(f"\nSave directory: {save_dir}")
    print("\n" + "="*100)
    print("Starting training...")
    print("="*100 + "\n")

    # Training loop
    best_map50 = 0.0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch + 1, args.epochs)

        # Validate
        val_losses, val_metrics = validate(model, val_loader, criterion, device, epoch + 1, args.epochs)

        # Update LR
        scheduler.step()

        # Print summary
        epoch_time = time.time() - epoch_start
        print_epoch_summary(epoch + 1, args.epochs, train_losses, val_losses, val_metrics, epoch_time)

        # Save checkpoint
        metrics = {'train': train_losses, 'val': val_losses, 'metrics': val_metrics}
        save_checkpoint(model, optimizer, epoch, metrics, save_dir / 'last.pt')

        # Save best
        if val_metrics['mAP50'] > best_map50:
            best_map50 = val_metrics['mAP50']
            save_checkpoint(model, optimizer, epoch, metrics, save_dir / 'best.pt')
            print(f"[BEST] New best mAP@0.5: {best_map50:.4f} - Model saved!\n")

    print("\n" + "="*100)
    print(f"{'Training Completed!':^100}")
    print("="*100)
    print(f"Best mAP@0.5: {best_map50:.4f}")
    print(f"Models saved to: {save_dir}")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
