# AURA-NET Usage on Kaggle
**Hướng dẫn sử dụng AURA-NET trên Kaggle**

---

## 🚀 Quick Start

### **Import và khởi tạo model:**

```python
# Cách 1: Import với underscore
from models import AURANet_Nano, AURANet_Tiny, AURANet_S, AURANet_M, AURANet_L, AURANet_X

# Cách 2: Import không underscore (dễ gõ hơn)
from models import AURANetNano, AURANetTiny, AURANetS, AURANetM, AURANetL, AURANetX

# Thêm train method
from train_yaml import add_train_method

# Chọn variant phù hợp với GPU của bạn
add_train_method(AURANetTiny)
model = AURANetTiny(num_classes=9)
```

---

## 📋 Chọn Variant phù hợp với Kaggle GPU

### **Kaggle Free GPU (T4 - 16GB):**

| Variant | Batch Size | Training Time | Recommended |
|---------|-----------|---------------|-------------|
| **Nano** | 32-48 | ~2-3 hours | ✅ Ultra fast |
| **Tiny** | 24-32 | ~3-4 hours | ✅ Very fast |
| **S** | 16-24 | ~4-5 hours | ✅ Fast |
| **M** | 8-16 | ~6-8 hours | ✅ Balanced |
| **L** | 6-8 | ~10-12 hours | ⚠️ Slow |
| **X** | 2-4 | ~20+ hours | ❌ Too slow |

**Khuyến nghị:** Sử dụng **Tiny** hoặc **S** cho Kaggle Free GPU

---

## 💻 Code Examples

### **1. AURA-NET-Nano (Ultra Fast)**

```python
from models import AURANetNano
from train_yaml import add_train_method

add_train_method(AURANetNano)
model = AURANetNano(num_classes=9)

model.train_model(
    data="/kaggle/input/datasets/kcnttngotruongan/bccd-new/data.yaml",
    epochs=300,        # Train longer for small model
    batch=32,          # Large batch
    imgsz=640,
    device=0,
    optimizer="AdamW",
    lr0=2e-4,          # Higher learning rate
    patience=80,
    seed=42,
    project="runs/train",
    name="auranet_nano_bccd",
    workers=4
)
```

**Ưu điểm:**
- ⚡ Cực nhanh (~2-3 giờ)
- 💾 Tiết kiệm bộ nhớ
- 🔋 Phù hợp cho mobile deployment

**Nhược điểm:**
- 📉 Độ chính xác thấp hơn

---

### **2. AURA-NET-Tiny (Recommended for Kaggle)** ⭐

```python
from models import AURANetTiny
from train_yaml import add_train_method

add_train_method(AURANetTiny)
model = AURANetTiny(num_classes=9)

model.train_model(
    data="/kaggle/input/datasets/kcnttngotruongan/bccd-new/data.yaml",
    epochs=250,        # Train longer
    batch=24,          # Large batch
    imgsz=640,
    device=0,
    optimizer="AdamW",
    lr0=2e-4,          # Higher learning rate
    patience=70,
    seed=42,
    project="runs/train",
    name="auranet_tiny_bccd",
    workers=4
)
```

**Ưu điểm:**
- ⚡ Rất nhanh (~3-4 giờ)
- 📊 Độ chính xác tốt
- 🎯 **Best choice cho Kaggle**

---

### **3. AURA-NET-S (Small)**

```python
from models import AURANetS
from train_yaml import add_train_method

add_train_method(AURANetS)
model = AURANetS(num_classes=9)

model.train_model(
    data="/kaggle/input/datasets/kcnttngotruongan/bccd-new/data.yaml",
    epochs=200,
    batch=16,
    imgsz=640,
    device=0,
    optimizer="AdamW",
    lr0=1.5e-4,
    patience=60,
    seed=42,
    project="runs/train",
    name="auranet_s_bccd",
    workers=4
)
```

**Ưu điểm:**
- ⚡ Nhanh (~4-5 giờ)
- 📊 Độ chính xác tốt hơn Tiny
- 🎯 Cân bằng tốt

---

### **4. AURA-NET-M (Medium - Default)**

```python
from models import AURANetM
from train_yaml import add_train_method

add_train_method(AURANetM)
model = AURANetM(num_classes=9)

model.train_model(
    data="/kaggle/input/datasets/kcnttngotruongan/bccd-new/data.yaml",
    epochs=200,
    batch=8,
    imgsz=640,
    device=0,
    optimizer="AdamW",
    lr0=1e-4,
    patience=60,
    seed=42,
    project="runs/train",
    name="auranet_m_bccd",
    workers=4
)
```

**Ưu điểm:**
- 📊 Độ chính xác cao
- ⚖️ Cân bằng tốt
- 🎯 Default choice

**Nhược điểm:**
- ⏱️ Chậm hơn (~6-8 giờ)

---

### **5. AURA-NET-L (Large - Optimized)**

```python
from models import AURANetL
from train_yaml import add_train_method

add_train_method(AURANetL)
model = AURANetL(num_classes=9)

model.train_model(
    data="/kaggle/input/datasets/kcnttngotruongan/bccd-new/data.yaml",
    epochs=150,        # Train less
    batch=6,           # Smaller batch
    imgsz=640,
    device=0,
    optimizer="AdamW",
    lr0=8e-5,          # Lower learning rate
    patience=50,
    seed=42,
    project="runs/train",
    name="auranet_l_bccd",
    workers=4
)
```

**Ưu điểm:**
- 📊 Độ chính xác rất cao
- 🎯 Tối ưu hơn so với version cũ

**Nhược điểm:**
- ⏱️ Chậm (~10-12 giờ)
- ⚠️ Có thể timeout trên Kaggle

---

## 🔧 Tips for Kaggle

### **1. Tối ưu thời gian training:**

```python
# Sử dụng mixed precision
model.train_model(
    data="data.yaml",
    epochs=200,
    batch=16,
    amp=True,          # Enable mixed precision
    device=0
)
```

### **2. Checkpoint để tránh mất dữ liệu:**

```python
model.train_model(
    data="data.yaml",
    epochs=200,
    batch=16,
    save_period=10,    # Save every 10 epochs
    device=0
)
```

### **3. Early stopping:**

```python
model.train_model(
    data="data.yaml",
    epochs=200,
    batch=16,
    patience=50,       # Stop if no improvement after 50 epochs
    device=0
)
```

### **4. Resume training:**

```python
model.train_model(
    data="data.yaml",
    epochs=200,
    batch=16,
    resume=True,       # Resume from last checkpoint
    device=0
)
```

---

## 📊 Expected Performance on BCCD Dataset

| Variant | mAP@0.5 | Training Time | Memory | Recommended |
|---------|---------|---------------|--------|-------------|
| Nano | ~75% | 2-3h | ~2GB | Mobile deployment |
| Tiny | ~80% | 3-4h | ~3GB | ⭐ **Best for Kaggle** |
| S | ~85% | 4-5h | ~4GB | Good balance |
| M | ~88% | 6-8h | ~6GB | High accuracy |
| L | ~90% | 10-12h | ~8GB | Maximum accuracy |
| X | ~92% | 20+h | ~12GB | ❌ Too slow for Kaggle |

---

## 🎯 Recommended Workflow

### **Step 1: Quick experiment với Nano/Tiny**

```python
from models import AURANetTiny
from train_yaml import add_train_method

add_train_method(AURANetTiny)
model = AURANetTiny(num_classes=9)

# Quick training để test pipeline
model.train_model(
    data="data.yaml",
    epochs=50,         # Quick test
    batch=24,
    device=0
)
```

### **Step 2: Full training với S hoặc M**

```python
from models import AURANetS
from train_yaml import add_train_method

add_train_method(AURANetS)
model = AURANetS(num_classes=9)

# Full training
model.train_model(
    data="data.yaml",
    epochs=200,
    batch=16,
    device=0,
    patience=60
)
```

### **Step 3: Fine-tune với L (nếu còn thời gian)**

```python
from models import AURANetL
from train_yaml import add_train_method

add_train_method(AURANetL)
model = AURANetL(num_classes=9)

# Fine-tune for best accuracy
model.train_model(
    data="data.yaml",
    epochs=150,
    batch=6,
    device=0,
    patience=50
)
```

---

## 🐛 Common Issues

### **1. Out of Memory (OOM)**

**Solution:** Giảm batch size hoặc chuyển sang variant nhỏ hơn

```python
# Thay vì
model.train_model(batch=16, ...)

# Thử
model.train_model(batch=8, ...)  # Hoặc batch=4
```

### **2. Training quá chậm**

**Solution:** Chuyển sang variant nhỏ hơn

```python
# Thay vì AURANetM
from models import AURANetS  # Hoặc AURANetTiny

add_train_method(AURANetS)
model = AURANetS(num_classes=9)
```

### **3. Kaggle timeout (9 hours)**

**Solution:** 
- Sử dụng Tiny hoặc S
- Giảm số epochs
- Enable early stopping

```python
model.train_model(
    epochs=150,        # Giảm từ 200
    patience=40,       # Early stopping
    device=0
)
```

---

## 📚 Complete Example

```python
# Import
from models import AURANetTiny
from train_yaml import add_train_method

# Setup
add_train_method(AURANetTiny)
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
    workers=4,
    amp=True,          # Mixed precision
    save_period=10     # Save checkpoints
)

# Inference
import torch
from PIL import Image
import torchvision.transforms as T

# Load image
image = Image.open("test.jpg")
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor()
])
x = transform(image).unsqueeze(0)

# Predict
model.eval()
with torch.no_grad():
    predictions = model.predict(x, conf_threshold=0.25)

print(f"Detected {len(predictions[0]['boxes'])} objects")
```

---

## 🎓 Summary

**Cho Kaggle Free GPU (T4 - 16GB):**

1. **Quick test**: Nano (2-3h)
2. **Best choice**: Tiny (3-4h) ⭐
3. **Good balance**: S (4-5h)
4. **High accuracy**: M (6-8h)
5. **Maximum accuracy**: L (10-12h) ⚠️

**Khuyến nghị:** Bắt đầu với **AURANetTiny** cho Kaggle!

---

**Happy Training on Kaggle! 🚀**
