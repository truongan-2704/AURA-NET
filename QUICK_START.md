# AURA-NET Variants - Quick Reference

## 📦 Available Variants

| Variant | Parameters | Import Name | Use Case |
|---------|-----------|-------------|----------|
| **Nano** | ~0.8M | `AURANetNano` or `AURANet_Nano` | MCU, IoT devices |
| **Tiny** | ~1.5M | `AURANetTiny` or `AURANet_Tiny` | Mobile phones ⭐ |
| **S** | ~2.5M | `AURANetS` or `AURANet_S` | Smartphones |
| **M** | ~9.6M | `AURANetM` or `AURANet_M` | Standard GPUs |
| **L** | ~18M | `AURANetL` or `AURANet_L` | Edge servers |
| **X** | ~45M | `AURANetX` or `AURANet_X` | Cloud servers |

## 🚀 Usage on Kaggle

### Step 1: Import

```python
# Choose one variant (with or without underscore)
from models import AURANetTiny  # Recommended for Kaggle
# from models import AURANet_Tiny  # Same as above
```

### Step 2: Add train method

```python
from train_yaml import add_train_method

add_train_method()  # No arguments needed!
```

### Step 3: Create and train

```python
model = AURANetTiny(num_classes=9)

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
```

## 📝 Complete Example

```python
from models import AURANetTiny
from train_yaml import add_train_method

add_train_method()

model = AURANetTiny(num_classes=9)

model.train_model(
    data="/kaggle/input/datasets/kcnttngotruongan/bccd-new/data.yaml",
    epochs=250,
    batch=24,
    device=0
)
```

## 🎯 Recommended Settings for Kaggle

### Nano (Ultra Fast - 2-3 hours)
```python
model = AURANetNano(num_classes=9)
model.train_model(data="data.yaml", epochs=300, batch=32, lr0=3e-4)
```

### Tiny (Best for Kaggle - 3-4 hours) ⭐
```python
model = AURANetTiny(num_classes=9)
model.train_model(data="data.yaml", epochs=250, batch=24, lr0=2e-4)
```

### S (Good Balance - 4-5 hours)
```python
model = AURANetS(num_classes=9)
model.train_model(data="data.yaml", epochs=200, batch=16, lr0=1.5e-4)
```

### M (High Accuracy - 6-8 hours)
```python
model = AURANetM(num_classes=9)
model.train_model(data="data.yaml", epochs=200, batch=8, lr0=1e-4)
```

### L (Maximum Accuracy - 10-12 hours)
```python
model = AURANetL(num_classes=9)
model.train_model(data="data.yaml", epochs=150, batch=6, lr0=8e-5)
```

## 📚 Documentation

- **Full Guide**: See [MODEL_VARIANTS_GUIDE.md](MODEL_VARIANTS_GUIDE.md)
- **Kaggle Usage**: See [KAGGLE_USAGE.md](KAGGLE_USAGE.md)
- **Example Code**: See [train_kaggle_example.py](train_kaggle_example.py)

## ❓ Common Issues

### Issue: `TypeError: add_train_method() takes 0 positional arguments but 1 was given`

**Solution**: Don't pass any arguments to `add_train_method()`

```python
# ❌ Wrong
add_train_method(AURANetTiny)

# ✅ Correct
add_train_method()
```

### Issue: `ImportError: cannot import name 'AURANetTiny'`

**Solution**: Import from `models`, not `models.v3`

```python
# ❌ Wrong
from models.v3 import AURANetTiny

# ✅ Correct
from models import AURANetTiny
```

### Issue: Out of Memory

**Solution**: Use smaller variant or reduce batch size

```python
# Try smaller variant
from models import AURANetNano  # Instead of Tiny

# Or reduce batch size
model.train_model(batch=8)  # Instead of batch=24
```

## 🎓 Summary

**For Kaggle:**
1. Import: `from models import AURANetTiny`
2. Setup: `add_train_method()`
3. Create: `model = AURANetTiny(num_classes=9)`
4. Train: `model.train_model(data="data.yaml", epochs=250, batch=24)`

**Recommended**: Start with **AURANetTiny** for Kaggle!
