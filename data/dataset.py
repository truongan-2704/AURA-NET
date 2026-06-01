"""
Dataset loader for YOLO format
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from .transforms import get_transforms
from .collate import collate_fn


class AURADataset(Dataset):
    """
    Dataset for AURA-Net training
    Supports YOLO format: class x_center y_center width height (normalized)
    """
    
    def __init__(self, img_dir, label_dir, img_size=640, augment=False, cache_images=False):
        """
        Args:
            img_dir: Directory containing images
            label_dir: Directory containing label files
            img_size: Target image size
            augment: Whether to apply data augmentation
            cache_images: Whether to cache images in memory
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Get image files
        self.img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.img_files.extend(list(self.img_dir.glob(ext)))
            self.img_files.extend(list(self.img_dir.glob(ext.upper())))
        
        self.img_files = sorted(self.img_files)
        
        # Get corresponding label files
        self.label_files = []
        for img_file in self.img_files:
            label_file = self.label_dir / (img_file.stem + '.txt')
            self.label_files.append(label_file)
        
        # Cache
        self.imgs = [None] * len(self.img_files)
        if cache_images:
            self._cache_images()
        
        # Transforms
        self.transforms = get_transforms(img_size, augment)
        
        print(f'Dataset: {len(self.img_files)} images found in {img_dir}')
    
    def _cache_images(self):
        """Cache images in memory"""
        print('Caching images...')
        for i, img_file in enumerate(self.img_files):
            self.imgs[i] = cv2.imread(str(img_file))
            if i % 100 == 0:
                print(f'Cached {i}/{len(self.img_files)} images')
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            img: Tensor (3, H, W)
            target: Tensor (N, 5) containing [class, x, y, w, h] normalized
            img_path: Image file path
        """
        # Load image
        if self.imgs[idx] is not None:
            img = self.imgs[idx].copy()
        else:
            img = cv2.imread(str(self.img_files[idx]))
        
        if img is None:
            raise ValueError(f'Failed to load image: {self.img_files[idx]}')
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        
        # Load labels
        label_file = self.label_files[idx]
        labels = []
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        labels.append([cls, x, y, w, h])
        
        labels = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
        
        # Apply transforms
        if self.transforms:
            img, labels = self.transforms(img, labels)
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        labels = torch.from_numpy(labels).float()
        
        return img, labels, str(self.img_files[idx])


def create_dataloader(dataset_config, img_size=640, batch_size=8,
                     split='train', shuffle=True, num_workers=4):
    """
    Create dataloader

    Args:
        dataset_config: Dataset configuration dict
        img_size: Image size
        batch_size: Batch size
        split: 'train', 'val', or 'test'
        shuffle: Whether to shuffle
        num_workers: Number of worker processes

    Returns:
        dataloader: DataLoader instance
    """
    # Get paths
    root = Path(dataset_config['path'])

    # Handle different dataset structures
    # Structure 1: train/images, train/labels (YOLO format)
    # Structure 2: images/train, labels/train (alternative format)
    img_path_str = dataset_config[split]

    if 'images' in img_path_str:
        # Already contains 'images' folder
        img_dir = root / img_path_str
        # Replace 'images' with 'labels' to get label dir
        label_path_str = img_path_str.replace('images', 'labels')
        label_dir = root / label_path_str
    else:
        # Old format: assume split points to images directly
        img_dir = root / img_path_str
        label_dir = root / 'labels' / split
    
    # Create dataset
    augment = (split == 'train')
    dataset = AURADataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        augment=augment,
        cache_images=False
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == 'train')
    )
    
    return dataloader
