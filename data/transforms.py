"""
Data augmentation and transforms
"""

import cv2
import numpy as np
import random


class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, labels):
        for t in self.transforms:
            img, labels = t(img, labels)
        return img, labels


class Resize:
    """Resize image and adjust labels"""
    def __init__(self, img_size=640):
        self.img_size = img_size
    
    def __call__(self, img, labels):
        h0, w0 = img.shape[:2]
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Labels are already normalized, no need to adjust
        return img, labels


class RandomHorizontalFlip:
    """Random horizontal flip"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, labels):
        if random.random() < self.p:
            img = np.fliplr(img).copy()
            if len(labels) > 0:
                labels[:, 1] = 1.0 - labels[:, 1]  # Flip x coordinate
        return img, labels


class RandomBrightness:
    """Random brightness adjustment"""
    def __init__(self, limit=0.2):
        self.limit = limit
    
    def __call__(self, img, labels):
        alpha = 1.0 + random.uniform(-self.limit, self.limit)
        img = np.clip(img * alpha, 0, 255).astype(np.uint8)
        return img, labels


class RandomContrast:
    """Random contrast adjustment"""
    def __init__(self, limit=0.2):
        self.limit = limit
    
    def __call__(self, img, labels):
        alpha = 1.0 + random.uniform(-self.limit, self.limit)
        img = np.clip(128 + alpha * (img - 128), 0, 255).astype(np.uint8)
        return img, labels


class RandomHueSaturation:
    """Random hue and saturation adjustment"""
    def __init__(self, hue_limit=20, sat_limit=30):
        self.hue_limit = hue_limit
        self.sat_limit = sat_limit
    
    def __call__(self, img, labels):
        if random.random() < 0.5:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Hue
            img_hsv[:, :, 0] += random.uniform(-self.hue_limit, self.hue_limit)
            img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 179)
            
            # Saturation
            img_hsv[:, :, 1] += random.uniform(-self.sat_limit, self.sat_limit)
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
            
            img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return img, labels


class Normalize:
    """Normalize image (optional, can be done in dataset)"""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, img, labels):
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img, labels


def get_transforms(img_size=640, augment=False):
    """
    Get transform pipeline
    
    Args:
        img_size: Target image size
        augment: Whether to apply augmentation
    
    Returns:
        transforms: Compose object
    """
    transforms_list = [Resize(img_size)]
    
    if augment:
        transforms_list.extend([
            RandomHorizontalFlip(p=0.5),
            RandomBrightness(limit=0.2),
            RandomContrast(limit=0.2),
            RandomHueSaturation(hue_limit=10, sat_limit=20)
        ])
    
    return Compose(transforms_list)
