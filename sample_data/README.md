# Sample Training Data for AURA-NET

## 📁 Cấu trúc thư mục

```
sample_data/
├── images/              # Ảnh training (format: .npy)
│   ├── sample_0000.npy
│   ├── sample_0001.npy
│   └── ...
├── labels/              # Annotations (YOLO format)
│   ├── sample_0000.txt
│   ├── sample_0001.txt
│   └── ...
├── dataset_config.json  # Cấu hình dataset
└── README.md           # File này
```

## 📋 Format dữ liệu

### Images
- Format: NumPy array (.npy)
- Shape: (640, 640, 3)
- Type: uint8
- Range: [0, 255]

### Labels (YOLO format)
Mỗi dòng trong file .txt tương ứng với một object:
```
class_id cx cy w h
```

Trong đó:
- `class_id`: ID của class (0-79 cho COCO)
- `cx, cy`: Tọa độ center của box (normalized [0, 1])
- `w, h`: Width và height của box (normalized [0, 1])

Ví dụ:
```
70 0.588 0.742 0.119 0.260
53 0.437 0.595 0.147 0.100
```

## 🔧 Sử dụng

### Load một sample

```python
import numpy as np
import torch

# Load image
image = np.load('sample_data/images/sample_0000.npy')
image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

# Load labels
boxes = []
with open('sample_data/labels/sample_0000.txt', 'r') as f:
    for line in f:
        class_id, cx, cy, w, h = map(float, line.strip().split())
        boxes.append([cx, cy, w, h, class_id])

boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
```

### Tạo thêm data

```bash
# Tạo 100 samples
python sample_train_data.py --mode generate --num-samples 100 --output-dir sample_data

# Test một batch
python sample_train_data.py --mode demo
```

## 📊 Thống kê

- **Số lượng samples**: 20
- **Image size**: 640x640
- **Số classes**: 80 (COCO)
- **Số objects/image**: 1-5 (random)

## ⚠️ Lưu ý

Đây là **dữ liệu mẫu tổng hợp** (synthetic data) để test training pipeline, không phải real data.

Để train mô hình thực tế, sử dụng dataset thật như:
- COCO: https://cocodataset.org/
- Pascal VOC: http://host.robots.ox.ac.uk/pascal/VOC/
- Custom dataset của bạn

## 🚀 Training với sample data

```python
from models.v3 import AURANet
from losses.aura_v3_loss import AURANetLoss

# Create model
model = AURANet(num_classes=80)
loss_fn = AURANetLoss(num_classes=80)

# Load sample batch
images, targets = load_sample_batch('sample_data')

# Forward pass
output = model(images)

# Compute loss
loss_dict = loss_fn(output, targets)
print(f"Total loss: {loss_dict['loss'].item():.4f}")
```
