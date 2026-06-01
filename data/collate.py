"""
Collate function for batching
"""

import torch


def collate_fn(batch):
    """
    Custom collate function for variable number of objects per image
    
    Args:
        batch: List of (img, labels, path) tuples
    
    Returns:
        imgs: Tensor (B, 3, H, W)
        labels: List of tensors, each (N, 5)
        paths: List of image paths
    """
    imgs, labels, paths = zip(*batch)
    
    # Stack images
    imgs = torch.stack(imgs, 0)
    
    # Labels remain as list since each image may have different number of objects
    labels = list(labels)
    
    return imgs, labels, paths
