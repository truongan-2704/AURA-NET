"""
AURA-NET Training on Kaggle - Complete Working Example
======================================================

This example works on Kaggle notebooks
"""

import sys
from pathlib import Path

# Add AURA-NET to path (adjust path based on your Kaggle setup)
# If AURA-NET is in /kaggle/working/AURA-NET
sys.path.insert(0, '/kaggle/working/AURA-NET')

# Or if it's in /kaggle/input
# sys.path.insert(0, '/kaggle/input/aura-net-code/AURA-NET')

# Now import
from models import AURANetTiny
from train_yaml import add_train_method

# Add train method
add_train_method()

# Create model
model = AURANetTiny(num_classes=9)

# Train
model.train_model(
    data="/kaggle/input/datasets/kcnttngotruongan/bccd-new/data.yaml",
    epochs=250,
    batch=24,
    imgsz=640,
    device=0,
    optimizer="AdamW",
    lr0=2e-4,
    patience=70,
    seed=42,
    project="runs/train",
    name="auranet_tiny_bccd",
    workers=4
)

print("\n" + "="*80)
print("Training completed!")
print("="*80)
