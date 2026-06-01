# AURA-NET - Quick Command Reference
**Tổng hợp các câu lệnh quan trọng**

---

## 🖥️ Training trên máy Local

### Training cơ bản
```bash
# Training đơn giản (giao diện cơ bản)
python train_apple_leaf.py

# Training với giao diện đẹp (giống YOLO)
python train_apple_leaf_enhanced.py

# Training với GPU
python train_apple_leaf_enhanced.py --device cuda --batch-size 16

# Training nhanh để test
python train_apple_leaf_enhanced.py --epochs 10 --batch-size 4
```

### Training với tùy chỉnh
```bash
python train_apple_leaf_enhanced.py \
    --data-dir datasets/apple_leaft_detection \
    --batch-size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --img-size 640 \
    --device cuda \
    --workers 4 \
    --save-dir runs/apple_leaf_v1
```

### Resume training
```bash
python train_apple_leaf_enhanced.py \
    --resume runs/apple_leaf/20260601_120000/last.pt \
    --epochs 100
```

---

## ☁️ Training trên Kaggle

### Setup trong Kaggle Notebook

**Cell 1: Install & Check GPU**
```python
!pip install -q thop pyyaml pillow tqdm

import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Cell 2: Setup Dataset**
```python
DATASET_PATH = '/kaggle/input/apple-leaf-detection/apple_leaft_detection'

# Verify
import os
print(os.listdir(DATASET_PATH))
```

**Cell 3: Upload Code**
```python
# Option 1: Upload files qua Kaggle UI
# Option 2: Clone từ GitHub
!git clone https://github.com/your-username/AURA-NET.git
%cd AURA-NET
```

**Cell 4: Start Training**
```python
!python train_apple_leaf_enhanced.py \
    --data-dir /kaggle/input/apple-leaf-detection/apple_leaft_detection \
    --batch-size 16 \
    --epochs 50 \
    --device cuda \
    --workers 2 \
    --save-dir /kaggle/working/runs
```

**Cell 5: Download Results**
```python
!zip -r results.zip /kaggle/working/runs
from IPython.display import FileLink
FileLink('/kaggle/working/results.zip')
```

---

## 🧪 Testing & Inference

### Test trên toàn bộ test set
```bash
python test_apple_leaf.py \
    --checkpoint runs/apple_leaf/20260601_120000/best.pt \
    --split test \
    --save-dir results/test_results
```

### Test trên validation set
```bash
python test_apple_leaf.py \
    --checkpoint runs/apple_leaf/20260601_120000/best.pt \
    --split valid \
    --conf-threshold 0.3
```

### Test trên 1 ảnh
```bash
python test_apple_leaf.py \
    --checkpoint runs/apple_leaf/20260601_120000/best.pt \
    --image datasets/apple_leaft_detection/test/images/sample.jpg \
    --save-dir results/
```

---

## 🔍 Kiểm tra Setup

### Test dataset loading
```bash
python test_apple_setup.py
```

### Test model forward pass
```python
python -c "
from models.v3 import AURANet
import torch

model = AURANet(num_classes=4)
x = torch.randn(1, 3, 640, 640)
output = model(x)
print('Model OK!')
print(f'Output boxes: {output[\"boxes\"].shape}')
"
```

### Kiểm tra GPU
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📊 Xem kết quả Training

### Load checkpoint
```python
import torch

checkpoint = torch.load('runs/apple_leaf/20260601_120000/best.pt')
print(f"Epoch: {checkpoint['epoch']}")
print(f"Train loss: {checkpoint['metrics']['train']['loss']:.4f}")
print(f"Val loss: {checkpoint['metrics']['val']['loss']:.4f}")
print(f"mAP@0.5: {checkpoint['metrics']['metrics']['mAP50']:.4f}")
```

### Xem config
```bash
cat runs/apple_leaf/20260601_120000/config.yaml
```

### List checkpoints
```bash
ls -lh runs/apple_leaf/20260601_120000/
```

---

## 🎯 Batch Size Recommendations

### Theo GPU Memory
```bash
# GPU 4GB (hoặc CPU)
--batch-size 2

# GPU 8GB
--batch-size 4-8

# GPU 16GB (RTX 3080, V100)
--batch-size 16-24

# GPU 24GB+ (RTX 3090, A100)
--batch-size 32-48
```

### Theo Dataset Size
```bash
# Small dataset (<1000 images)
--batch-size 8-16

# Medium dataset (1000-10000 images) - Apple Leaf
--batch-size 16-32

# Large dataset (>10000 images)
--batch-size 32-64
```

---

## ⚡ Tối ưu Training Speed

### Tăng workers
```bash
--workers 4  # hoặc 8 nếu CPU mạnh
```

### Mixed precision (GPU)
```python
# Thêm vào training script
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, targets)
```

### Giảm image size
```bash
--img-size 512  # thay vì 640
```

---

## 📁 Cấu trúc Files

```
AURA-NET/
├── datasets/
│   └── apple_leaft_detection/     # Dataset
│       ├── train/
│       ├── valid/
│       └── test/
│
├── models/v3/                      # Model code
│   ├── aura_net_v3.py
│   ├── dual_stream_extractor.py
│   └── ...
│
├── losses/                         # Loss functions
│   └── aura_v3_loss.py
│
├── train_apple_leaf.py             # Training (basic)
├── train_apple_leaf_enhanced.py    # Training (YOLO-style)
├── test_apple_leaf.py              # Testing
├── test_apple_setup.py             # Setup verification
│
├── runs/                           # Training outputs
│   └── apple_leaf/
│       └── YYYYMMDD_HHMMSS/
│           ├── best.pt
│           ├── last.pt
│           └── config.yaml
│
└── results/                        # Test results
    └── test_results/
        ├── *.jpg
        └── results.json
```

---

## 🐛 Troubleshooting

### Out of Memory
```bash
# Giảm batch size
--batch-size 4

# Giảm image size
--img-size 512

# Giảm K (evidence regions)
# Edit trong code: min_K=30, max_K=100
```

### Training quá chậm
```bash
# Tăng workers
--workers 8

# Giảm epochs để test
--epochs 10

# Sử dụng GPU
--device cuda
```

### Loss không giảm
```bash
# Giảm learning rate
--lr 5e-5

# Tăng epochs
--epochs 100

# Check data loading
python test_apple_setup.py
```

---

## 📚 Tài liệu tham khảo

- **Training Guide**: `APPLE_LEAF_TRAINING_GUIDE.md`
- **Kaggle Guide**: `KAGGLE_TRAINING_GUIDE.md`
- **Model Design**: `AURA_NET_V3_DESIGN.md`
- **Update Summary**: `UPDATE_SUMMARY.md`

---

## 🚀 Quick Start (TL;DR)

```bash
# 1. Kiểm tra setup
python test_apple_setup.py

# 2. Training
python train_apple_leaf_enhanced.py --epochs 50 --batch-size 16

# 3. Testing
python test_apple_leaf.py \
    --checkpoint runs/apple_leaf/*/best.pt \
    --split test \
    --save-dir results/
```

---

**Happy Training! 🎉**
