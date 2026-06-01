# AURA-NET - Kaggle Training Guide
**Hướng dẫn train AURA-NET trên Kaggle**

---

## 📋 Chuẩn bị

### 1. Upload Dataset lên Kaggle

**Option 1: Upload từ máy local**
1. Nén dataset thành file zip:
   ```bash
   cd datasets
   zip -r apple_leaft_detection.zip apple_leaft_detection/
   ```

2. Tạo dataset mới trên Kaggle:
   - Vào https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload file `apple_leaft_detection.zip`
   - Đặt tên: `apple-leaf-detection`

**Option 2: Sử dụng dataset có sẵn**
- Dataset đã có trên Roboflow: https://universe.roboflow.com/clljy/apple_leaf_detection/dataset/1
- Hoặc tìm trên Kaggle: https://www.kaggle.com/datasets

### 2. Upload Code lên Kaggle

**Tạo Kaggle Notebook:**
1. Vào https://www.kaggle.com/code
2. Click "New Notebook"
3. Chọn "GPU P100" hoặc "GPU T4 x2" (free)
4. Copy code từ các file sau vào notebook

---

## 🚀 Kaggle Notebook Setup

### Cell 1: Install Dependencies

```python
# Install required packages
!pip install -q thop pyyaml pillow tqdm

# Check GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 2: Clone/Upload Code

**Option A: Upload files trực tiếp**
```python
# Upload các file .py vào Kaggle notebook
# File -> Upload -> Chọn các file:
# - models/v3/*.py
# - losses/aura_v3_loss.py
# - train_apple_leaf_enhanced.py
```

**Option B: Clone từ GitHub (nếu có)**
```python
!git clone https://github.com/your-username/AURA-NET.git
%cd AURA-NET
```

**Option C: Tạo structure thủ công**
```python
import os

# Tạo cấu trúc thư mục
os.makedirs('models/v3', exist_ok=True)
os.makedirs('losses', exist_ok=True)
os.makedirs('runs', exist_ok=True)

# Copy code vào các file (xem phần dưới)
```

### Cell 3: Setup Dataset Path

```python
# Kaggle dataset path
DATASET_PATH = '/kaggle/input/apple-leaf-detection/apple_leaft_detection'

# Hoặc nếu upload trực tiếp
# DATASET_PATH = '/kaggle/working/apple_leaft_detection'

# Verify dataset
import os
print("Dataset structure:")
for root, dirs, files in os.walk(DATASET_PATH):
    level = root.replace(DATASET_PATH, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 2:  # Only show 2 levels
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files
            print(f'{subindent}{file}')
```

### Cell 4: Copy Model Code

```python
%%writefile models/v3/aura_net_v3.py
# Copy toàn bộ nội dung từ file models/v3/aura_net_v3.py
# (Xem file gốc trong project)
```

```python
%%writefile models/v3/__init__.py
"""
AURA-NET Package
"""
from .aura_net_v3 import AURANet
from .dual_stream_extractor import DualStreamFeatureExtractor
from .evidence_proposal_network import EvidenceProposalNetwork
from .adaptive_evidence_router import AdaptiveEvidenceRouter
from .evidence_to_object_decoder import EvidenceToObjectDecoder

__all__ = [
    'AURANet',
    'DualStreamFeatureExtractor',
    'EvidenceProposalNetwork',
    'AdaptiveEvidenceRouter',
    'EvidenceToObjectDecoder',
]
```

### Cell 5: Copy Loss Code

```python
%%writefile losses/aura_v3_loss.py
# Copy toàn bộ nội dung từ file losses/aura_v3_loss.py
```

```python
%%writefile losses/__init__.py
from .aura_v3_loss import AURANetLoss, EvidenceLoss, CIoULoss, FocalLoss, UncertaintyLoss

__all__ = ['AURANetLoss', 'EvidenceLoss', 'CIoULoss', 'FocalLoss', 'UncertaintyLoss']
```

### Cell 6: Copy Training Script

```python
%%writefile train_kaggle.py
# Copy nội dung từ train_apple_leaf_enhanced.py
# Hoặc sử dụng script đơn giản hơn (xem dưới)
```

### Cell 7: Start Training

```python
# Training với GPU
!python train_kaggle.py \
    --data-dir /kaggle/input/apple-leaf-detection/apple_leaft_detection \
    --batch-size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --img-size 640 \
    --device cuda \
    --workers 2 \
    --save-dir /kaggle/working/runs/apple_leaf
```

### Cell 8: Monitor Training

```python
# Xem logs
!tail -f /kaggle/working/runs/apple_leaf/*/training.log

# Hoặc load checkpoint để xem metrics
import torch
checkpoint = torch.load('/kaggle/working/runs/apple_leaf/*/best.pt')
print(f"Best epoch: {checkpoint['epoch']}")
print(f"Best mAP@0.5: {checkpoint['metrics']['metrics']['mAP50']:.4f}")
```

### Cell 9: Save Results

```python
# Download trained model
from IPython.display import FileLink

# Tạo link download
FileLink('/kaggle/working/runs/apple_leaf/*/best.pt')

# Hoặc copy về output để Kaggle tự động save
!cp -r /kaggle/working/runs /kaggle/working/output/
```

---

## 📝 Script Training Đơn Giản cho Kaggle

Nếu muốn script ngắn gọn hơn, tạo file `train_kaggle_simple.py`:

```python
"""
Simple Kaggle Training Script for AURA-NET
"""
import sys
import torch
from pathlib import Path

# Add to path
sys.path.append('/kaggle/working')

from models.v3 import AURANet
from losses import AURANetLoss
from train_apple_leaf_enhanced import (
    AppleLeafDataset, collate_fn, 
    train_one_epoch, validate, 
    save_checkpoint
)
from torch.utils.data import DataLoader

# Config
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = '/kaggle/input/apple-leaf-detection/apple_leaft_detection'
SAVE_DIR = '/kaggle/working/runs'

print(f"Device: {DEVICE}")
print(f"Training for {EPOCHS} epochs")

# Create datasets
train_dataset = AppleLeafDataset(DATA_DIR, 'train', 640, augment=True)
val_dataset = AppleLeafDataset(DATA_DIR, 'valid', 640, augment=False)

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, 
                         num_workers=2, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False,
                       num_workers=2, collate_fn=collate_fn)

# Create model
model = AURANet(num_classes=4).to(DEVICE)
criterion = AURANetLoss(num_classes=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Training loop
best_map = 0.0
for epoch in range(EPOCHS):
    train_losses = train_one_epoch(model, train_loader, criterion, 
                                   optimizer, DEVICE, epoch+1, EPOCHS)
    val_losses, val_metrics = validate(model, val_loader, criterion, 
                                      DEVICE, epoch+1, EPOCHS)
    
    # Save best
    if val_metrics['mAP50'] > best_map:
        best_map = val_metrics['mAP50']
        save_checkpoint(model, optimizer, epoch, 
                       {'train': train_losses, 'val': val_losses, 
                        'metrics': val_metrics},
                       Path(SAVE_DIR) / 'best.pt')
        print(f"[BEST] mAP@0.5: {best_map:.4f}")

print(f"\nTraining completed! Best mAP@0.5: {best_map:.4f}")
```

---

## ⚙️ Kaggle Settings

### GPU Settings
- **Free tier**: GPU P100 (16GB) - 30 hours/week
- **Batch size recommendations**:
  - P100: 16-24
  - T4: 8-16

### Time Limits
- **Free**: 9 hours per session
- **Verified**: 12 hours per session
- Tip: Save checkpoint mỗi epoch để có thể resume

### Storage
- **Working directory**: `/kaggle/working` (20GB)
- **Output**: Tự động save khi notebook kết thúc
- **Temp**: `/kaggle/temp` (5GB)

---

## 🔧 Tips & Tricks

### 1. Resume Training
```python
# Trong training script
if Path('/kaggle/working/runs/last.pt').exists():
    checkpoint = torch.load('/kaggle/working/runs/last.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

### 2. Reduce Memory Usage
```python
# Giảm batch size
BATCH_SIZE = 8

# Giảm K (evidence regions)
model = AURANet(num_classes=4, min_K=30, max_K=100)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Trong training loop
with autocast():
    outputs = model(images)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Faster Training
```python
# Tăng workers
num_workers = 4

# Pin memory
pin_memory = True

# Persistent workers
persistent_workers = True
```

### 4. Save Checkpoints Regularly
```python
# Save mỗi 10 epochs
if (epoch + 1) % 10 == 0:
    save_checkpoint(model, optimizer, epoch, metrics,
                   f'/kaggle/working/runs/epoch_{epoch+1}.pt')
```

---

## 📊 Expected Training Time

**On Kaggle GPU P100:**
- **1 epoch**: ~15-20 minutes (4556 images, batch_size=16)
- **50 epochs**: ~12-16 hours
- **100 epochs**: ~24-32 hours (cần chạy 2-3 sessions)

**Recommendations:**
- Train 50 epochs trong 1 session
- Nếu cần 100 epochs, chia làm 2 sessions với resume

---

## 📥 Download Results

```python
# Tạo zip file
!zip -r results.zip /kaggle/working/runs

# Download
from IPython.display import FileLink
FileLink('/kaggle/working/results.zip')
```

---

## 🎯 Complete Kaggle Notebook Template

Tôi đã tạo template hoàn chỉnh. Bạn có thể:

1. **Copy toàn bộ code** từ các file trong project
2. **Paste vào Kaggle notebook** theo thứ tự các cell ở trên
3. **Chạy từng cell** theo thứ tự
4. **Monitor training** và download model khi xong

**Hoặc đơn giản hơn:**
- Upload toàn bộ project lên GitHub
- Clone trong Kaggle notebook
- Chạy training script

---

**Good luck with training on Kaggle! 🚀**
