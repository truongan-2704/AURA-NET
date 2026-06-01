"""
AURA-NET Training Example for Kaggle
=====================================

Complete example showing how to use AURA-NET variants on Kaggle
"""

# Import variant (choose one)
from models import AURANetTiny  # Recommended for Kaggle
# from models import AURANetNano  # Ultra fast
# from models import AURANetS     # Good balance
# from models import AURANetM     # High accuracy

from train_yaml import add_train_method

# Add train method to all variants
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
