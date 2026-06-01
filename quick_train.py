"""
AURA-NET - Quick YAML Training
===============================

Simplified training script using YAML config.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.v3 import AURANet
from train_yaml import train_model

# Add train method to model
AURANet.train = train_model

# Create and train model
model = AURANet(num_classes=4)

# Train (YOLO-style API)
model.train(
    data='datasets/apple_leaft_detection/data.yaml',
    epochs=50,          # Giảm xuống 50 để test nhanh
    batch=8,
    imgsz=640,
    device=0,           # 0 = GPU, 'cpu' = CPU
    optimizer='AdamW',
    patience=30,
    seed=42,
    project='runs/train',
    name='exp'
)
