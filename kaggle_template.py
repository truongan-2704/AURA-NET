"""
AURA-NET on Kaggle - Copy & Paste Template
===========================================

Just copy this code to your Kaggle notebook!
"""

import sys

# ⚠️ IMPORTANT: Add AURA-NET to path first!
sys.path.insert(0, '/kaggle/working/AURA-NET')

# Import
from models import AURANetTiny  # or AURANetNano, AURANetS, AURANetM, AURANetL
from train_yaml import add_train_method

# Setup
add_train_method()  # No arguments!

# Create model
model = AURANetTiny(num_classes=9)

# Train
model.train_model(
    data="/kaggle/input/datasets/kcnttngotruongan/bccd-new/data.yaml",
    epochs=250,
    batch=24,
    device=0,
    lr0=2e-4,
    patience=70,
    project="runs/train",
    name="auranet_tiny_bccd"
)

"""
VARIANTS:
- AURANetNano: Ultra fast (2-3h), ~0.8M params
- AURANetTiny: Best for Kaggle (3-4h), ~1.5M params ⭐
- AURANetS: Good balance (4-5h), ~2.5M params
- AURANetM: High accuracy (6-8h), ~9.6M params
- AURANetL: Maximum accuracy (10-12h), ~18M params

COMMON ISSUES:
1. ModuleNotFoundError → Add sys.path.insert(0, '/kaggle/working/AURA-NET')
2. TypeError add_train_method → Don't pass arguments: add_train_method()
3. Out of Memory → Reduce batch size or use smaller variant
"""
