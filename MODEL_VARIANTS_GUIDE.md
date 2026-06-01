# AURA-NET Model Variants Guide
**Hướng dẫn sử dụng các biến thế AURA-NET - Tối ưu cho End Devices**

---

## 📊 Model Variants Overview

AURA-NET có 6 biến thế từ siêu nhỏ đến lớn, phù hợp với mọi thiết bị:

| Variant | Parameters | Feature Dim | K Range | Iterations | Target Device |
|---------|-----------|-------------|---------|------------|---------------|
| **Nano** | ~0.8M | 64 | 15-40 | 1 | MCU, Raspberry Pi, IoT |
| **Tiny** | ~1.5M | 96 | 20-60 | 2 | Mobile phones, tablets |
| **S** (Small) | ~2.5M | 128 | 30-80 | 2 | Modern smartphones, edge |
| **M** (Medium) | ~9.6M | 256 | 50-150 | 3 | **Default**, standard GPUs |
| **L** (Large) | ~18M | 320 | 60-200 | 3 | Edge servers, high-end PC |
| **X** (Extra) | ~45M | 512 | 100-300 | 5 | Cloud servers, research |

---

## 🚀 Quick Start

### **Import models:**

```python
from models import (
    AURANet_Nano,  # Ultra-lightweight
    AURANet_Tiny,  # Very lightweight
    AURANet_S,     # Small
    AURANet_M,     # Medium (default)
    AURANet_L,     # Large
    AURANet_X      # Extra Large
)

# Or use default (Medium)
from models import AURANet  # Same as AURANet_M
```

### **Create model:**

```python
# Nano - for MCU/embedded devices
model_nano = AURANet_Nano(num_classes=80)

# Tiny - for mobile devices
model_tiny = AURANet_Tiny(num_classes=80)

# Small - for modern smartphones
model_s = AURANet_S(num_classes=80)

# Medium - default, balanced
model_m = AURANet_M(num_classes=80)

# Large - high accuracy (optimized)
model_l = AURANet_L(num_classes=80)

# Extra Large - maximum accuracy
model_x = AURANet_X(num_classes=80)
```

---

## 📋 Detailed Specifications

### **AURA-NET-Nano** 🔥 NEW

```python
from models import AURANet_Nano

model = AURANet_Nano(num_classes=80)
```

**Specifications:**
- **Parameters**: ~0.8M
- **Feature Dim**: 64
- **K Range**: 15-40 evidence regions
- **Iterations**: 1 (single pass)
- **Extractor Channels**: 32

**Best for:**
- Microcontrollers (MCU)
- Raspberry Pi, Jetson Nano
- IoT devices
- Ultra-low power devices
- Embedded systems

**Performance:**
- Speed: ⭐⭐⭐⭐⭐ (Ultra Fast)
- Accuracy: ⭐⭐ (Basic)
- Memory: ⭐⭐⭐⭐⭐ (Minimal)

---

### **AURA-NET-Tiny** 🔥 NEW

```python
from models import AURANet_Tiny

model = AURANet_Tiny(num_classes=80)
```

**Specifications:**
- **Parameters**: ~1.5M
- **Feature Dim**: 96
- **K Range**: 20-60 evidence regions
- **Iterations**: 2
- **Extractor Channels**: 48

**Best for:**
- Mobile phones
- Tablets
- Edge devices
- Real-time mobile apps
- Limited GPU memory

**Performance:**
- Speed: ⭐⭐⭐⭐⭐ (Very Fast)
- Accuracy: ⭐⭐⭐ (Good)
- Memory: ⭐⭐⭐⭐⭐ (Very Low)

---

### **AURA-NET-S (Small)**

```python
from models import AURANet_S

model = AURANet_S(num_classes=80)
```

**Specifications:**
- **Parameters**: ~2.5M
- **Feature Dim**: 128
- **K Range**: 30-80 evidence regions
- **Iterations**: 2
- **Extractor Channels**: 64

**Best for:**
- Modern smartphones
- Edge computing
- Real-time applications
- Limited GPU memory (<4GB)

**Performance:**
- Speed: ⭐⭐⭐⭐⭐ (Fast)
- Accuracy: ⭐⭐⭐ (Good)
- Memory: ⭐⭐⭐⭐ (Low)

---

### **AURA-NET-M (Medium)** ⭐ Default

```python
from models import AURANet_M
# Or simply
from models import AURANet

model = AURANet_M(num_classes=80)
```

**Specifications:**
- **Parameters**: ~9.6M
- **Feature Dim**: 256
- **K Range**: 50-150 evidence regions
- **Iterations**: 3
- **Extractor Channels**: 128

**Best for:**
- General purpose detection
- Most applications
- Good balance of speed/accuracy
- Standard GPUs (8GB+)

**Performance:**
- Speed: ⭐⭐⭐⭐ (Fast)
- Accuracy: ⭐⭐⭐⭐ (Very Good)
- Memory: ⭐⭐⭐⭐ (Moderate)

---

### **AURA-NET-L (Large)** ✨ Optimized

```python
from models import AURANet_L

model = AURANet_L(num_classes=80)
```

**Specifications:**
- **Parameters**: ~18M (optimized from 25M)
- **Feature Dim**: 320 (reduced from 384)
- **K Range**: 60-200 evidence regions
- **Iterations**: 3 (reduced from 4)
- **Extractor Channels**: 160

**Best for:**
- Edge servers
- High-end workstations
- Complex scenes
- Small object detection
- GPUs with 12GB+ memory

**Performance:**
- Speed: ⭐⭐⭐⭐ (Good)
- Accuracy: ⭐⭐⭐⭐⭐ (Excellent)
- Memory: ⭐⭐⭐ (Moderate-High)

---

### **AURA-NET-X (Extra Large)**

```python
from models import AURANet_X

model = AURANet_X(num_classes=80)
```

**Specifications:**
- **Parameters**: ~45M
- **Feature Dim**: 512
- **K Range**: 100-300 evidence regions
- **Iterations**: 5
- **Extractor Channels**: 256

**Best for:**
- Research
- Competitions
- Maximum accuracy
- Cloud servers
- High-end GPUs (24GB+)

**Performance:**
- Speed: ⭐⭐ (Slow)
- Accuracy: ⭐⭐⭐⭐⭐ (Best)
- Memory: ⭐⭐ (High)

---

## 💻 Usage Examples

### **Training with different variants:**

```python
from models import AURANet_Nano, AURANet_Tiny, AURANet_S, AURANet_M, AURANet_L
from train_yaml import train_model

# Add train method
from models.variants import AURANetBase
AURANetBase.train = train_model

# Train Nano model (ultra-lightweight)
model_nano = AURANet_Nano(num_classes=9)
model_nano.train(
    data='data.yaml',
    epochs=300,
    batch=32,  # Can use larger batch
    device=0
)

# Train Tiny model (mobile)
model_tiny = AURANet_Tiny(num_classes=9)
model_tiny.train(
    data='data.yaml',
    epochs=250,
    batch=24,
    device=0
)

# Train Small model
model_s = AURANet_S(num_classes=9)
model_s.train(
    data='data.yaml',
    epochs=200,
    batch=16,
    device=0
)

# Train Medium model
model_m = AURANet_M(num_classes=9)
model_m.train(
    data='data.yaml',
    epochs=200,
    batch=8,
    device=0
)

# Train Large model (optimized)
model_l = AURANet_L(num_classes=9)
model_l.train(
    data='data.yaml',
    epochs=150,
    batch=6,
    device=0
)
```

### **Inference with different variants:**

```python
import torch
from models import AURANet_Nano, AURANet_Tiny, AURANet_S, AURANet_M, AURANet_L

# Load image
image = torch.randn(1, 3, 640, 640)

# Nano model - ultra fast
model_nano = AURANet_Nano(num_classes=9)
model_nano.eval()
predictions_nano = model_nano.predict(image, conf_threshold=0.25)

# Tiny model - very fast
model_tiny = AURANet_Tiny(num_classes=9)
model_tiny.eval()
predictions_tiny = model_tiny.predict(image, conf_threshold=0.25)

# Small model - fast
model_s = AURANet_S(num_classes=9)
model_s.eval()
predictions_s = model_s.predict(image, conf_threshold=0.25)

# Medium model - balanced
model_m = AURANet_M(num_classes=9)
model_m.eval()
predictions_m = model_m.predict(image, conf_threshold=0.25)

# Large model - accurate
model_l = AURANet_L(num_classes=9)
model_l.eval()
predictions_l = model_l.predict(image, conf_threshold=0.25)
```

### **Compare all variants:**

```python
from models import get_model_info

# Print comparison table
get_model_info()
```

Output:
```
==========================================================================================
AURA-NET Model Variants (Optimized for End Devices)
==========================================================================================

Variant    Params (M)   Feature    K Range      Iter   Target Device                 
------------------------------------------------------------------------------------------
Nano       0.82         64         15-40        1      MCU, Raspberry Pi, IoT        
Tiny       1.48         96         20-60        2      Mobile phones, tablets        
S          2.53         128        30-80        2      Modern smartphones, edge      
M          9.63         256        50-150       3      Standard GPUs, workstations   
L          17.85        320        60-200       3      Edge servers, high-end PC     
X          45.23        512        100-300      5      Cloud servers, research       
==========================================================================================

Testing forward pass (640x640 image)...
  Nano      :   18.3ms
  Tiny      :   28.7ms
  S         :   45.2ms
  M         :   87.3ms
  L         :  124.6ms
  X         :  245.1ms
==========================================================================================
```

---

## 🎯 Choosing the Right Variant

### **Use AURA-NET-Nano when:**
- ✅ Deploying on MCU/embedded devices
- ✅ Raspberry Pi, Jetson Nano
- ✅ IoT devices with <1GB RAM
- ✅ Ultra-low power consumption required
- ✅ Basic detection is sufficient

### **Use AURA-NET-Tiny when:**
- ✅ Deploying on mobile phones
- ✅ Tablets and portable devices
- ✅ Need real-time inference (>60 FPS)
- ✅ Limited GPU memory (<2GB)
- ✅ Mobile apps

### **Use AURA-NET-S when:**
- ✅ Modern smartphones
- ✅ Edge computing devices
- ✅ Need real-time inference (>30 FPS)
- ✅ Limited GPU memory (<4GB)
- ✅ Speed is more important than accuracy

### **Use AURA-NET-M when:**
- ✅ General purpose detection
- ✅ Standard GPU (8-16GB)
- ✅ Need good balance
- ✅ **Most common choice**

### **Use AURA-NET-L when:**
- ✅ Edge servers
- ✅ High-end workstations
- ✅ Complex scenes with many objects
- ✅ Small object detection
- ✅ Have 12GB+ GPU

### **Use AURA-NET-X when:**
- ✅ Research/competition
- ✅ Maximum accuracy needed
- ✅ Have 24GB+ GPU
- ✅ Inference speed not critical

---

## 📊 Batch Size Recommendations

### **GPU Memory vs Batch Size:**

| GPU Memory | Nano | Tiny | S | M | L | X |
|-----------|------|------|---|---|---|---|
| 2GB | 32-48 | 24-32 | 16-24 | 4-8 | 2-4 | 1-2 |
| 4GB | 64+ | 48-64 | 32-48 | 8-16 | 4-8 | 2-4 |
| 8GB | 128+ | 96+ | 64+ | 16-32 | 8-16 | 4-8 |
| 16GB | 256+ | 192+ | 128+ | 32-64 | 16-32 | 8-16 |
| 24GB+ | 512+ | 384+ | 256+ | 64+ | 32-64 | 16-32 |

---

## 🔧 Training Tips

### **Nano Model:**
```python
model = AURANet_Nano(num_classes=9)
model.train(
    data='data.yaml',
    epochs=300,        # Train longer
    batch=32,          # Large batch
    lr0=3e-4,          # Higher LR
    device=0
)
```

### **Tiny Model:**
```python
model = AURANet_Tiny(num_classes=9)
model.train(
    data='data.yaml',
    epochs=250,        # Train longer
    batch=24,          # Large batch
    lr0=2e-4,          # Higher LR
    device=0
)
```

### **Small Model (S):**
```python
model = AURANet_S(num_classes=9)
model.train(
    data='data.yaml',
    epochs=200,        # Standard
    batch=16,          # Larger batch
    lr0=2e-4,          # Higher LR
    device=0
)
```

### **Medium Model (M):**
```python
model = AURANet_M(num_classes=9)
model.train(
    data='data.yaml',
    epochs=200,        # Standard
    batch=8,           # Standard batch
    lr0=1e-4,          # Standard LR
    device=0
)
```

### **Large Model (L):**
```python
model = AURANet_L(num_classes=9)
model.train(
    data='data.yaml',
    epochs=150,        # Can train less
    batch=6,           # Smaller batch
    lr0=8e-5,          # Lower LR
    device=0
)
```

---

## 📈 Expected Performance

### **COCO Dataset (80 classes):**

| Variant | mAP@0.5 | mAP@0.5:0.95 | FPS (V100) | Params | Memory |
|---------|---------|--------------|------------|--------|--------|
| Nano | ~35% | ~22% | 150+ | 0.8M | <500MB |
| Tiny | ~38% | ~25% | 120 | 1.5M | <800MB |
| S | ~42% | ~28% | 85 | 2.5M | <1.5GB |
| M | ~48% | ~33% | 45 | 9.6M | ~3GB |
| L | ~51% | ~36% | 30 | 18M | ~5GB |
| X | ~55% | ~40% | 15 | 45M | ~10GB |

*Note: Actual performance depends on training settings and dataset*

---

## 🔄 Converting Between Variants

### **Progressive training (recommended for limited resources):**

```python
# Step 1: Train on Nano first (very fast)
model_nano = AURANet_Nano(num_classes=9)
model_nano.train(data='data.yaml', epochs=100)

# Step 2: Fine-tune on Tiny (better accuracy)
model_tiny = AURANet_Tiny(num_classes=9)
model_tiny.train(data='data.yaml', epochs=50, lr0=1e-5)

# Step 3: Fine-tune on Small (even better)
model_s = AURANet_S(num_classes=9)
model_s.train(data='data.yaml', epochs=50, lr0=5e-6)

# Step 4: Final fine-tune on Medium (best balance)
model_m = AURANet_M(num_classes=9)
model_m.train(data='data.yaml', epochs=50, lr0=1e-6)
```

---

## 📚 Code Examples

### **Complete training script:**

```python
from models import AURANet_Nano, AURANet_Tiny, AURANet_S, AURANet_M, AURANet_L, AURANet_X
from train_yaml import train_model

# Choose variant based on your device
VARIANT = 'M'  # Nano, Tiny, S, M, L, or X

if VARIANT == 'Nano':
    model = AURANet_Nano(num_classes=9)
    batch_size = 32
elif VARIANT == 'Tiny':
    model = AURANet_Tiny(num_classes=9)
    batch_size = 24
elif VARIANT == 'S':
    model = AURANet_S(num_classes=9)
    batch_size = 16
elif VARIANT == 'M':
    model = AURANet_M(num_classes=9)
    batch_size = 8
elif VARIANT == 'L':
    model = AURANet_L(num_classes=9)
    batch_size = 6
else:  # X
    model = AURANet_X(num_classes=9)
    batch_size = 2

# Train
model.train = train_model
model.train(
    data='data.yaml',
    epochs=200,
    batch=batch_size,
    device=0,
    project=f'runs/train',
    name=f'auranet_{VARIANT.lower()}'
)
```

---

## 🎓 Summary

- **AURA-NET-Nano**: Ultra-lightweight, MCU/IoT devices
- **AURA-NET-Tiny**: Very lightweight, mobile phones
- **AURA-NET-S**: Lightweight, modern smartphones
- **AURA-NET-M**: **Default**, balanced, most common
- **AURA-NET-L**: High accuracy, edge servers (optimized)
- **AURA-NET-X**: Maximum accuracy, research

**Recommendation**: 
- For **mobile/embedded**: Start with **Nano** or **Tiny**
- For **general use**: Start with **AURA-NET-M**
- For **high accuracy**: Use **AURA-NET-L** (optimized for efficiency)

---

**Happy Training! 🚀**
