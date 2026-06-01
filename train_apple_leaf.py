"""
AURA-NET - Training Script for Apple Leaf Detection
====================================================

Train AURA-NET on Apple Leaf Disease Detection dataset.

Dataset: 4 classes
- Black Rot
- Powdery Mildew
- Rust
- Scab

Total: 4556 train, 570 val, 569 test images
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

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.v3 import AURANet
from losses.aura_v3_loss import AURANetLoss


class AppleLeafDataset(Dataset):
    """
    Apple Leaf Detection Dataset

    Format: YOLO (class_id cx cy w h) normalized
    """
    def __init__(self, data_dir, split='train', img_size=640, augment=False):
        """
        Args:
            data_dir: Path to dataset root (e.g., 'datasets/apple_leaft_detection')
            split: 'train', 'valid', or 'test'
            img_size: Target image size
            augment: Whether to apply augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment

        # Paths
        self.img_dir = self.data_dir / split / 'images'
        self.label_dir = self.data_dir / split / 'labels'

        # Get all image files
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')))

        print(f"[{split}] Found {len(self.img_files)} images")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')

        # Load labels
        label_path = self.label_dir / (img_path.stem + '.txt')
        boxes = self.load_labels(label_path)

        # Resize image
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Convert to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Apply augmentation if needed
        if self.augment:
            image, boxes = self.apply_augmentation(image, boxes)

        return image, boxes

    def load_labels(self, label_path):
        """Load YOLO format labels."""
        boxes = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        boxes.append([cx, cy, w, h, class_id])

        if len(boxes) == 0:
            # Empty annotation
            boxes = torch.zeros((0, 5), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)

        return boxes

    def apply_augmentation(self, image, boxes):
        """Simple augmentation: horizontal flip."""
        if torch.rand(1) > 0.5:
            # Horizontal flip
            image = torch.flip(image, dims=[2])
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip cx

        return image, boxes


def collate_fn(batch):
    """Custom collate function for variable number of boxes."""
    images = []
    targets = []

    for img, boxes in batch:
        images.append(img)
        targets.append(boxes)

    images = torch.stack(images, 0)

    return images, targets


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    loss_evidence = 0.0
    loss_box = 0.0
    loss_class = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = [t.to(device) for t in targets]

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        loss_evidence += loss_dict['loss_evidence'].item()
        loss_box += loss_dict['loss_box'].item()
        loss_class += loss_dict['loss_class'].item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'evid': f"{loss_dict['loss_evidence'].item():.4f}",
            'box': f"{loss_dict['loss_box'].item():.4f}",
            'cls': f"{loss_dict['loss_class'].item():.4f}"
        })

    # Average losses
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'loss_evidence': loss_evidence / num_batches,
        'loss_box': loss_box / num_batches,
        'loss_class': loss_class / num_batches,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    loss_evidence = 0.0
    loss_box = 0.0
    loss_class = 0.0

    pbar = tqdm(dataloader, desc="Validation")

    for images, targets in pbar:
        images = images.to(device)
        targets = [t.to(device) for t in targets]

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss_dict = criterion(outputs, targets)

        # Update metrics
        total_loss += loss_dict['loss'].item()
        loss_evidence += loss_dict['loss_evidence'].item()
        loss_box += loss_dict['loss_box'].item()
        loss_class += loss_dict['loss_class'].item()

    # Average losses
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'loss_evidence': loss_evidence / num_batches,
        'loss_box': loss_box / num_batches,
        'loss_class': loss_class / num_batches,
    }


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train AURA-NET on Apple Leaf Detection')
    parser.add_argument('--data-dir', type=str, default='datasets/apple_leaft_detection',
                       help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default='runs/apple_leaf',
                       help='Save directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("AURA-NET - Apple Leaf Detection Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.img_size}")
    print("=" * 60)

    # Create datasets
    train_dataset = AppleLeafDataset(
        args.data_dir,
        split='train',
        img_size=args.img_size,
        augment=True
    )

    val_dataset = AppleLeafDataset(
        args.data_dir,
        split='valid',
        img_size=args.img_size,
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create model
    model = AURANet(
        num_classes=4,  # Apple leaf has 4 classes
        feature_dim=256,
        min_K=50,  # Smaller K for smaller objects
        max_K=150,
        num_iterations=3
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params / 1e6:.2f}M")

    # Create loss function
    criterion = AURANetLoss(num_classes=4)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0001
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Resume from checkpoint if specified
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
    print("\nStarting training...\n")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Print metrics
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Evidence: {train_metrics['loss_evidence']:.4f} | "
              f"{val_metrics['loss_evidence']:.4f}")
        print(f"  Box: {train_metrics['loss_box']:.4f} | "
              f"{val_metrics['loss_box']:.4f}")
        print(f"  Class: {train_metrics['loss_class']:.4f} | "
              f"{val_metrics['loss_class']:.4f}")

        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            {'train': train_metrics, 'val': val_metrics},
            save_dir / 'last.pt'
        )

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics},
                save_dir / 'best.pt'
            )
            print(f"[BEST] Saved best model with val_loss={best_val_loss:.4f}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
