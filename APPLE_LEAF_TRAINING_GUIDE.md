# AURA-NET - Apple Leaf Detection Training Guide

## 📊 Dataset Information

**Apple Leaf Disease Detection Dataset**
- **Classes**: 4
  - Black Rot
  - Powdery Mildew
  - Rust
  - Scab
- **Training images**: 4,556
- **Validation images**: 570
- **Test images**: 569
- **Format**: YOLO (class_id cx cy w h)
- **Source**: Roboflow

## 🚀 Quick Start

### 1. Training

```bash
# Basic training
python train_apple_leaf.py

# Custom settings
python train_apple_leaf.py \
    --batch-size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --img-size 640 \
    --device cuda

# Resume from checkpoint
python train_apple_leaf.py --resume runs/apple_leaf/20260601_120000/best.pt
```

### 2. Testing

```bash
# Test on entire test set
python test_apple_leaf.py \
    --checkpoint runs/apple_leaf/20260601_120000/best.pt \
    --split test \
    --save-dir results/apple_leaf

# Test on single image
python test_apple_leaf.py \
    --checkpoint runs/apple_leaf/20260601_120000/best.pt \
    --image datasets/apple_leaft_detection/test/images/sample.jpg \
    --save-dir results/
```

## 📁 Project Structure

```
AURA-NET/
├── datasets/
│   └── apple_leaft_detection/
│       ├── train/
│       │   ├── images/     # 4,556 images
│       │   └── labels/     # YOLO format
│       ├── valid/
│       │   ├── images/     # 570 images
│       │   └── labels/
│       ├── test/
│       │   ├── images/     # 569 images
│       │   └── labels/
│       └── data.yaml       # Dataset config
│
├── train_apple_leaf.py     # Training script
├── test_apple_leaf.py      # Testing/inference script
│
├── runs/                   # Training outputs
│   └── apple_leaf/
│       └── YYYYMMDD_HHMMSS/
│           ├── best.pt     # Best model
│           ├── last.pt     # Last checkpoint
│           └── config.yaml # Training config
│
└── results/                # Test results
    └── apple_leaf/
        ├── *.jpg           # Visualizations
        └── results.json    # Detection results
```

## ⚙️ Training Parameters

### Default Settings
```yaml
batch_size: 8
epochs: 50
learning_rate: 1e-4
img_size: 640
optimizer: AdamW
scheduler: CosineAnnealingLR
weight_decay: 0.0001
```

### Model Configuration
```python
num_classes: 4              # Apple leaf has 4 disease classes
feature_dim: 256
min_K: 50                   # Smaller K for leaf disease detection
max_K: 150
num_iterations: 3
```

## 📈 Training Tips

### 1. Batch Size
- **GPU 8GB**: batch_size=4-8
- **GPU 16GB**: batch_size=16-24
- **GPU 24GB+**: batch_size=32+

### 2. Learning Rate
- Start with `1e-4`
- If loss plateaus early, try `5e-5` or `1e-5`
- Use learning rate warmup for first few epochs

### 3. Data Augmentation
Currently implemented:
- Horizontal flip (50% probability)

Can add more:
- Random brightness/contrast
- Random rotation
- Mosaic augmentation
- MixUp

### 4. Training Duration
- **Quick test**: 10-20 epochs
- **Good results**: 50-100 epochs
- **Best results**: 100-200 epochs

## 📊 Expected Results

### Training Progress
```
Epoch 1/50
Train Loss: 2.5432 | Val Loss: 2.3456
  Evidence: 0.8234 | 0.7654
  Box: 1.2345 | 1.1234
  Class: 0.4853 | 0.4568

Epoch 10/50
Train Loss: 1.2345 | Val Loss: 1.1234
  Evidence: 0.3456 | 0.3234
  Box: 0.6789 | 0.6123
  Class: 0.2100 | 0.1877

...

[BEST] Saved best model with val_loss=0.8765
```

### Inference Output
```
Detections: 3
  1. Rust: 0.892
  2. Powdery Mildew: 0.765
  3. Black Rot: 0.543
```

## 🔧 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python train_apple_leaf.py --batch-size 4

# Reduce image size
python train_apple_leaf.py --img-size 512

# Reduce K (number of evidence regions)
# Edit train_apple_leaf.py: min_K=30, max_K=100
```

### Loss Not Decreasing
- Check learning rate (try lower: 5e-5 or 1e-5)
- Check data loading (visualize some batches)
- Increase training epochs
- Try different optimizer settings

### Low Detection Accuracy
- Train longer (more epochs)
- Increase batch size if possible
- Add more data augmentation
- Tune confidence threshold (default: 0.25)

## 📝 Example Commands

### Training Examples

```bash
# Fast training (for testing)
python train_apple_leaf.py --epochs 10 --batch-size 4

# Production training
python train_apple_leaf.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-4 \
    --save-dir runs/apple_leaf_v1

# Resume training
python train_apple_leaf.py \
    --resume runs/apple_leaf_v1/20260601_120000/last.pt \
    --epochs 150
```

### Testing Examples

```bash
# Test on validation set
python test_apple_leaf.py \
    --checkpoint runs/apple_leaf/20260601_120000/best.pt \
    --split valid \
    --conf-threshold 0.3

# Test on test set with visualization
python test_apple_leaf.py \
    --checkpoint runs/apple_leaf/20260601_120000/best.pt \
    --split test \
    --save-dir results/test_results

# Test single image
python test_apple_leaf.py \
    --checkpoint runs/apple_leaf/20260601_120000/best.pt \
    --image path/to/image.jpg \
    --conf-threshold 0.25 \
    --save-dir results/
```

## 📊 Monitoring Training

### Check Training Progress
```bash
# View saved checkpoints
ls -lh runs/apple_leaf/20260601_120000/

# Check config
cat runs/apple_leaf/20260601_120000/config.yaml
```

### Load and Inspect Checkpoint
```python
import torch

checkpoint = torch.load('runs/apple_leaf/20260601_120000/best.pt')
print(f"Epoch: {checkpoint['epoch']}")
print(f"Train loss: {checkpoint['metrics']['train']['loss']:.4f}")
print(f"Val loss: {checkpoint['metrics']['val']['loss']:.4f}")
```

## 🎯 Next Steps

1. **Train baseline model**
   ```bash
   python train_apple_leaf.py --epochs 50
   ```

2. **Evaluate on test set**
   ```bash
   python test_apple_leaf.py \
       --checkpoint runs/apple_leaf/*/best.pt \
       --split test \
       --save-dir results/baseline
   ```

3. **Tune hyperparameters**
   - Try different learning rates
   - Adjust batch size
   - Add more augmentation

4. **Compare with YOLO/other models**
   - Train YOLOv8 on same dataset
   - Compare mAP, speed, model size

## 📚 References

- Dataset: [Apple Leaf Detection on Roboflow](https://universe.roboflow.com/clljy/apple_leaf_detection/dataset/1)
- AURA-NET Paper: (Coming soon)
- AURA-NET Design: [AURA_NET_V3_DESIGN.md](AURA_NET_V3_DESIGN.md)

---

**Good luck with training! 🚀**
