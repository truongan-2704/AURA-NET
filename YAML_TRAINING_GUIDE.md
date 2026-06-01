# AURA-NET - YAML Training Guide
**Hướng dẫn train với file YAML giống YOLOv8**

---

## 🎯 Cách sử dụng

### **Method 1: Script đơn giản nhất**

```python
from models.v3 import AURANet
from train_yaml import train_model

# Add train method
AURANet.train = train_model

# Create model
model = AURANet(num_classes=4)

# Train (giống YOLOv8)
model.train(
    data='datasets/apple_leaft_detection/data.yaml',
    epochs=200,
    batch=8,
    imgsz=640,
    device=0,
    optimizer='AdamW',
    patience=60,
    seed=42,
    project='runs/train',
    name='exp'
)
```

### **Method 2: Chạy script có sẵn**

```bash
# Quick training (50 epochs)
python quick_train.py

# Full training (200 epochs)
python demo_yaml_training.py
```

---

## 📋 Tham số Training

### **Bắt buộc:**
- `data`: Path đến file `data.yaml`

### **Tùy chọn:**
```python
model.train(
    data='path/to/data.yaml',    # REQUIRED
    
    # Training settings
    epochs=200,                   # Số epochs
    batch=8,                      # Batch size
    imgsz=640,                    # Image size
    
    # Device & Optimizer
    device=0,                     # 0=GPU, 'cpu'=CPU
    optimizer='AdamW',            # 'AdamW' hoặc 'SGD'
    lr0=1e-4,                     # Learning rate
    weight_decay=0.0001,          # Weight decay
    
    # Early stopping
    patience=60,                  # Patience epochs
    
    # Reproducibility
    seed=42,                      # Random seed
    
    # Save settings
    project='runs/train',         # Project folder
    name='exp',                   # Experiment name
    
    # Other (not implemented yet)
    cache=False,                  # Cache images
    amp=False,                    # Mixed precision
)
```

---

## 📁 File data.yaml

File `data.yaml` phải có format:

```yaml
path: datasets/apple_leaft_detection  # Dataset root
train: train/images                    # Train images path
val: test/images                       # Val images path
test: test/images                      # Test images path (optional)

nc: 4                                  # Number of classes
names: ['Black Rot', 'Powdery_mildew', 'Rust', 'scab']  # Class names
```

**Ví dụ thực tế** (Apple Leaf Detection):
```yaml
path: datasets/apple_leaft_detection
train: train/images
val: test/images
test: test/images

nc: 4
names: ['Black Rot', 'Powdery_mildew', 'Rust', 'scab']

roboflow:
  workspace: clljy
  project: apple_leaf_detection
  version: 1
  license: CC BY 4.0
  url: https://universe.roboflow.com/clljy/apple_leaf_detection/dataset/1
```

---

## 🚀 Ví dụ sử dụng

### **1. Training cơ bản**

```python
from models.v3 import AURANet
from train_yaml import train_model

AURANet.train = train_model

model = AURANet(num_classes=4)
model.train(
    data='datasets/apple_leaft_detection/data.yaml',
    epochs=50,
    batch=8,
    device=0
)
```

### **2. Training với tùy chỉnh**

```python
model.train(
    data='datasets/apple_leaft_detection/data.yaml',
    epochs=200,
    batch=16,
    imgsz=640,
    device=0,
    optimizer='AdamW',
    lr0=1e-4,
    weight_decay=0.0001,
    patience=60,
    seed=42,
    project='runs/apple_leaf',
    name='experiment_1'
)
```

### **3. Training trên CPU**

```python
model.train(
    data='datasets/apple_leaft_detection/data.yaml',
    epochs=10,
    batch=2,
    device='cpu'
)
```

### **4. Training trên Kaggle**

```python
# Trong Kaggle Notebook
from models.v3 import AURANet
from train_yaml import train_model

AURANet.train = train_model

model = AURANet(num_classes=4)
model.train(
    data='/kaggle/input/apple-leaf-detection/data.yaml',
    epochs=200,
    batch=16,
    device=0,
    project='/kaggle/working/runs',
    name='kaggle_exp'
)
```

---

## 📊 Output

### **Giao diện training:**

```
====================================================================================================
                                    AURA-NET Training                                    
====================================================================================================

Configuration                  Value                                                                 
----------------------------------------------------------------------------------------------------
Device                         cuda:0                                                                
Epochs                         200                                                                   
Batch Size                     8                                                                     
Image Size                     640                                                                   
Optimizer                      AdamW                                                                 
Learning Rate                  0.0001                                                                
Patience                       60                                                                    
Seed                           42                                                                    
Dataset                        datasets/apple_leaft_detection                                        
Classes                        4                                                                     
Class Names                    Black Rot, Powdery_mildew, Rust, scab                                
Save Directory                 runs/train/exp                                                        
====================================================================================================

Loading datasets...
  Train: 4556 images
  Val: 570 images

====================================================================================================
                                      Model Summary                                      
====================================================================================================

Metric                         Value                                                                 
----------------------------------------------------------------------------------------------------
Total Parameters                     9,631,039
Trainable Parameters                 9,631,039
Parameters (M)                            9.63
GFLOPs                                  87.242G
====================================================================================================

====================================================================================================
Starting training...
====================================================================================================

Epoch 1/200: 100%|██████████| 570/570 [02:15<00:00,  4.21it/s, loss=2.543, box=1.234, cls=0.485]
Validating: 100%|██████████| 72/72 [00:18<00:00,  3.89it/s]

====================================================================================================
Epoch 1/200 - Time: 153.2s
====================================================================================================
Metric               Train           Val            
----------------------------------------------------------------------------------------------------
Loss                      2.5432          2.3456
Box Loss                  1.2345          1.1234
Class Loss                0.4853          0.4568
----------------------------------------------------------------------------------------------------
Precision (P)             -               0.7234
Recall (R)                -               0.6543
mAP@0.5                   -               0.5678
mAP@0.5:0.95              -               0.3975
====================================================================================================

[BEST] New best mAP@0.5: 0.5678

...

====================================================================================================
                                  Training Completed!                                  
====================================================================================================
Best mAP@0.5: 0.8234
Models saved to: runs/train/exp
====================================================================================================
```

### **Files được tạo:**

```
runs/train/exp/
├── best.pt          # Best model checkpoint
├── last.pt          # Last epoch checkpoint
└── config.yaml      # Training configuration
```

---

## 🔧 So sánh với YOLOv8

### **YOLOv8:**
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='data.yaml',
    epochs=200,
    batch=8,
    imgsz=640,
    device=0
)
```

### **AURA-NET:**
```python
from models.v3 import AURANet
from train_yaml import train_model

AURANet.train = train_model

model = AURANet(num_classes=4)
model.train(
    data='data.yaml',
    epochs=200,
    batch=8,
    imgsz=640,
    device=0
)
```

**→ API hoàn toàn giống nhau!**

---

## 💡 Tips

### **1. Tối ưu batch size theo GPU:**
```python
# GPU 8GB
batch=4-8

# GPU 16GB
batch=16-24

# GPU 24GB+
batch=32-48
```

### **2. Early stopping:**
```python
# Tự động dừng nếu không cải thiện sau 60 epochs
patience=60
```

### **3. Reproducibility:**
```python
# Kết quả giống nhau mỗi lần chạy
seed=42
```

### **4. Resume training:**
```python
# Load checkpoint và tiếp tục train
checkpoint = torch.load('runs/train/exp/last.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.train(data='data.yaml', epochs=300)  # Tiếp tục đến epoch 300
```

---

## 📚 Tài liệu liên quan

- **Training Guide**: `APPLE_LEAF_TRAINING_GUIDE.md`
- **Kaggle Guide**: `KAGGLE_TRAINING_GUIDE.md`
- **Commands Reference**: `COMMANDS_REFERENCE.md`

---

**Happy Training! 🚀**
