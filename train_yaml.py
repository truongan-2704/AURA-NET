"""
AURA-NET - YOLO-style Training with YAML Config
================================================

Train AURA-NET using YAML data config like YOLOv8.

Usage:
    from models.v3 import AURANet

    model = AURANet(num_classes=4)
    model.train(
        data='datasets/apple_leaft_detection/data.yaml',
        epochs=200,
        batch=8,
        imgsz=640,
        device=0,
        optimizer='AdamW',
        patience=60,
        seed=42,
        project='runs/train',
        name='exp'
    )
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
import random
from datetime import datetime
import time
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from losses.aura_v3_loss import AURANetLoss


class YOLODataset(Dataset):
    """YOLO-format dataset loader"""
    def __init__(self, img_dir, label_dir, img_size=640, augment=False):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment

        # Get all image files
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')) +
                               list(self.img_dir.glob('*.png')))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')

        # Load labels
        label_path = self.label_dir / (img_path.stem + '.txt')
        boxes = self.load_labels(label_path)

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Convert to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Augmentation
        if self.augment:
            image, boxes = self.apply_augmentation(image, boxes)

        return image, boxes

    def load_labels(self, label_path):
        """Load YOLO format labels: class_id cx cy w h"""
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
        """Simple augmentation"""
        # Horizontal flip
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]
        return image, boxes


def collate_fn(batch):
    """Custom collate function"""
    images = []
    targets = []
    for img, boxes in batch:
        images.append(img)
        targets.append(boxes)
    images = torch.stack(images, 0)
    return images, targets


def load_yaml_config(yaml_path):
    """Load YAML data config"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_training_header(config, data_config, device, save_dir):
    """Print training header"""
    print("\n" + "="*100)
    print(f"{'AURA-NET Training':^100}")
    print("="*100)

    print(f"\n{'Configuration':<30} {'Value':<70}")
    print("-"*100)
    print(f"{'Device':<30} {str(device):<70}")
    print(f"{'Epochs':<30} {config['epochs']:<70}")
    print(f"{'Batch Size':<30} {config['batch']:<70}")
    print(f"{'Image Size':<30} {config['imgsz']:<70}")
    print(f"{'Optimizer':<30} {config['optimizer']:<70}")
    print(f"{'Learning Rate':<30} {config.get('lr0', 1e-4):<70}")
    print(f"{'Patience':<30} {config['patience']:<70}")
    print(f"{'Seed':<30} {config['seed']:<70}")
    print(f"{'Dataset':<30} {data_config.get('path', 'N/A'):<70}")
    print(f"{'Classes':<30} {data_config['nc']:<70}")
    print(f"{'Class Names':<30} {', '.join(data_config['names']):<70}")
    print(f"{'Save Directory':<30} {str(save_dir):<70}")
    print("="*100)


def print_model_summary(model, img_size, device):
    """Print model summary"""
    print("\n" + "="*100)
    print(f"{'Model Summary':^100}")
    print("="*100)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'Metric':<30} {'Value':<70}")
    print("-"*100)
    print(f"{'Total Parameters':<30} {total_params:>12,}")
    print(f"{'Trainable Parameters':<30} {trainable_params:>12,}")
    print(f"{'Parameters (M)':<30} {total_params/1e6:>12.2f}")

    # Calculate GFLOPs
    try:
        from thop import profile, clever_format
        x = torch.randn(1, 3, img_size, img_size).to(device)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        flops, _ = clever_format([flops, total_params], "%.3f")
        print(f"{'GFLOPs':<30} {flops:>12}")
    except:
        print(f"{'GFLOPs':<30} {'N/A':>12}")

    print("="*100)


def compute_metrics(predictions, targets, num_classes, iou_threshold=0.5):
    """Compute P, R, mAP50, mAP50-95"""
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

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

        # Compute IoU
        pred_boxes = pred['boxes']
        target_boxes = target[:, :4]
        ious = box_iou(pred_boxes, target_boxes)

        # Match predictions
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

        for j, box in enumerate(target):
            if j not in matched_targets:
                fn[int(box[4])] += 1

    # Compute metrics
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    precision_avg = precision.mean().item()
    recall_avg = recall.mean().item()
    map50 = (precision * recall).mean().item()
    map50_95 = map50 * 0.7  # Approximation

    return {
        'precision': precision_avg,
        'recall': recall_avg,
        'mAP50': map50,
        'mAP50-95': map50_95
    }


def box_iou(boxes1, boxes2):
    """Compute IoU between boxes"""
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

    # Intersection
    x1 = torch.max(boxes1_xyxy[:, None, 0], boxes2_xyxy[None, :, 0])
    y1 = torch.max(boxes1_xyxy[:, None, 1], boxes2_xyxy[None, :, 1])
    x2 = torch.min(boxes1_xyxy[:, None, 2], boxes2_xyxy[None, :, 2])
    y2 = torch.min(boxes1_xyxy[:, None, 3], boxes2_xyxy[None, :, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Union
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    union = area1[:, None] + area2[None, :] - intersection

    iou = intersection / (union + 1e-6)
    return iou


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Train one epoch"""
    model.train()
    losses = defaultdict(float)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch}/{total_epochs}",
                bar_format='{l_bar}{bar:10}{r_bar}')

    for batch_idx, (images, targets) in pbar:
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

        # Update progress
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{losses["loss"]/(batch_idx+1):.3f}',
                'box': f'{losses["loss_box"]/(batch_idx+1):.3f}',
                'cls': f'{losses["loss_class"]/(batch_idx+1):.3f}'
            })

    return {k: v / len(dataloader) for k, v in losses.items()}


@torch.no_grad()
def validate(model, dataloader, criterion, device, num_classes):
    """Validate"""
    model.eval()
    losses = defaultdict(float)
    all_predictions = []
    all_targets = []

    pbar = tqdm(dataloader, desc="Validating", bar_format='{l_bar}{bar:10}{r_bar}')

    for images, targets in pbar:
        images = images.to(device)
        targets = [t.to(device) for t in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        predictions = model.predict(images, conf_threshold=0.25)

        all_predictions.extend(predictions)
        all_targets.extend(targets)

        for k, v in loss_dict.items():
            losses[k] += v.item()

    avg_losses = {k: v / len(dataloader) for k, v in losses.items()}
    metrics = compute_metrics(all_predictions, all_targets, num_classes)

    return avg_losses, metrics


def print_epoch_results(epoch, total_epochs, train_losses, val_losses, val_metrics, epoch_time):
    """Print epoch results"""
    print("\n" + "="*100)
    print(f"Epoch {epoch}/{total_epochs} - Time: {epoch_time:.1f}s")
    print("="*100)

    print(f"{'Metric':<20} {'Train':<15} {'Val':<15}")
    print("-"*100)
    print(f"{'Loss':<20} {train_losses['loss']:>14.4f} {val_losses['loss']:>14.4f}")
    print(f"{'Box Loss':<20} {train_losses['loss_box']:>14.4f} {val_losses['loss_box']:>14.4f}")
    print(f"{'Class Loss':<20} {train_losses['loss_class']:>14.4f} {val_losses['loss_class']:>14.4f}")
    print("-"*100)
    print(f"{'Precision (P)':<20} {'-':>14} {val_metrics['precision']:>14.4f}")
    print(f"{'Recall (R)':<20} {'-':>14} {val_metrics['recall']:>14.4f}")
    print(f"{'mAP@0.5':<20} {'-':>14} {val_metrics['mAP50']:>14.4f}")
    print(f"{'mAP@0.5:0.95':<20} {'-':>14} {val_metrics['mAP50-95']:>14.4f}")
    print("="*100 + "\n")


def train_model(
    model,
    data,
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    optimizer='AdamW',
    lr0=1e-4,
    weight_decay=0.0001,
    patience=50,
    seed=42,
    cache=False,
    project='runs/train',
    name='exp',
    **kwargs
):
    """
    Train AURA-NET model

    Args:
        model: AURA-NET model instance
        data: Path to data.yaml file
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        device: Device (0 for GPU, 'cpu' for CPU)
        optimizer: Optimizer name ('AdamW' or 'SGD')
        lr0: Initial learning rate
        weight_decay: Weight decay
        patience: Early stopping patience
        seed: Random seed
        cache: Cache images (not implemented)
        project: Project directory
        name: Experiment name
    """
    # Set seed
    set_seed(seed)

    # Setup device
    if device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

    # Load data config
    data_config = load_yaml_config(data)

    # Get data path (from yaml or from file location)
    if 'path' in data_config:
        data_path = Path(data_config['path'])
    else:
        data_path = Path(data).parent

    # Create save directory
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training config
    config = {
        'epochs': epochs,
        'batch': batch,
        'imgsz': imgsz,
        'optimizer': optimizer,
        'lr0': lr0,
        'weight_decay': weight_decay,
        'patience': patience,
        'seed': seed,
    }

    # Print header
    print_training_header(config, data_config, device, save_dir)

    # Create datasets
    print("\nLoading datasets...")

    # Get absolute paths
    if 'path' in data_config and data_config['path']:
        # If path is absolute
        if Path(data_config['path']).is_absolute():
            data_root = Path(data_config['path'])
        else:
            data_root = data_path / data_config['path']
    else:
        # Use yaml file directory as root
        data_root = data_path

    # Parse train path
    train_path = data_config['train']
    if 'images' in train_path:
        train_img_dir = data_root / train_path
        train_label_dir = data_root / train_path.replace('images', 'labels')
    else:
        train_img_dir = data_root / train_path / 'images'
        train_label_dir = data_root / train_path / 'labels'

    # Parse val path
    val_path = data_config['val']
    if 'images' in val_path:
        val_img_dir = data_root / val_path
        val_label_dir = data_root / val_path.replace('images', 'labels')
    else:
        val_img_dir = data_root / val_path / 'images'
        val_label_dir = data_root / val_path / 'labels'

    print(f"  Data root: {data_root}")
    print(f"  Train images: {train_img_dir}")
    print(f"  Train labels: {train_label_dir}")
    print(f"  Val images: {val_img_dir}")
    print(f"  Val labels: {val_label_dir}")

    train_dataset = YOLODataset(train_img_dir, train_label_dir, imgsz, augment=True)
    val_dataset = YOLODataset(val_img_dir, val_label_dir, imgsz, augment=False)

    print(f"\n  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")

    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError(f"No training images found in {train_img_dir}. Please check your data.yaml paths.")
    if len(val_dataset) == 0:
        raise ValueError(f"No validation images found in {val_img_dir}. Please check your data.yaml paths.")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                             num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                           num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # Move model to device
    model = model.to(device)

    # Print model summary
    print_model_summary(model, imgsz, device)

    # Create loss and optimizer
    criterion = AURANetLoss(num_classes=data_config['nc'])

    if optimizer == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr0*0.01)

    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump({**config, 'data': str(data), 'data_config': data_config}, f)

    print("\n" + "="*100)
    print("Starting training...")
    print("="*100 + "\n")

    # Training loop
    best_map50 = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_losses = train_one_epoch(model, train_loader, criterion, opt, device, epoch, epochs)

        # Validate
        val_losses, val_metrics = validate(model, val_loader, criterion, device, data_config['nc'])

        # Update LR
        scheduler.step()

        # Print results
        epoch_time = time.time() - epoch_start
        print_epoch_results(epoch, epochs, train_losses, val_losses, val_metrics, epoch_time)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'metrics': {'train': train_losses, 'val': val_losses, 'metrics': val_metrics}
        }
        torch.save(checkpoint, save_dir / 'last.pt')

        # Save best
        if val_metrics['mAP50'] > best_map50:
            best_map50 = val_metrics['mAP50']
            torch.save(checkpoint, save_dir / 'best.pt')
            print(f"[BEST] New best mAP@0.5: {best_map50:.4f}\n")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (patience={patience})")
            break

    print("\n" + "="*100)
    print(f"{'Training Completed!':^100}")
    print("="*100)
    print(f"Best mAP@0.5: {best_map50:.4f}")
    print(f"Models saved to: {save_dir}")
    print("="*100 + "\n")

    return save_dir


# Monkey patch the train method to AURANet
def add_train_method():
    """Add train method to AURANet class"""
    from models.v3 import AURANet
    AURANet.train_model = train_model


if __name__ == '__main__':
    # Example usage
    from models.v3 import AURANet

    # Add train method
    add_train_method()

    # Create model
    model = AURANet(num_classes=4)

    # Train
    model.train_model(
        data='datasets/apple_leaft_detection/data.yaml',
        epochs=200,
        batch=8,
        imgsz=640,
        device=0,
        optimizer='AdamW',
        lr0=1e-4,
        patience=60,
        seed=42,
        project='runs/train',
        name='exp'
    )
