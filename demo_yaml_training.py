"""
AURA-NET - YAML Training Demo
==============================

Train AURA-NET using YAML config like YOLOv8.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.v3 import AURANet
from train_yaml import train_model

# Monkey patch train method to model
AURANet.train = train_model


def main():
    """Main training function"""

    # Create model
    print("Creating AURA-NET model...")
    model = AURANet(num_classes=4)  # 4 classes for Apple Leaf

    # Train with YAML config (YOLO-style)
    print("\nStarting training with YAML config...")

    model.train(
        data='datasets/apple_leaft_detection/data.yaml',  # Path to data.yaml
        epochs=200,                 # Number of epochs
        batch=8,                    # Batch size
        imgsz=640,                  # Image size
        device=0,                   # GPU id (0 for GPU, 'cpu' for CPU)
        optimizer='AdamW',          # Optimizer: 'AdamW' or 'SGD'
        lr0=1e-4,                   # Initial learning rate
        weight_decay=0.0001,        # Weight decay
        patience=60,                # Early stopping patience
        seed=42,                    # Random seed
        cache=False,                # Cache images (not implemented yet)
        project='runs/train',       # Project directory
        name='apple_leaf_exp'       # Experiment name
    )

    print("\nTraining completed!")
    print("Results saved to: runs/train/apple_leaf_exp/")


if __name__ == '__main__':
    main()
