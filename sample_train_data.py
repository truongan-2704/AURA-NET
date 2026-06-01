"""
AURA-NET - Sample Training Data Generator
==========================================

Tạo dữ liệu mẫu để test training pipeline.
"""

import torch
import numpy as np
from pathlib import Path
import json


def generate_sample_image(height=640, width=640):
    """
    Tạo ảnh mẫu với random noise và một số hình dạng đơn giản.

    Returns:
        image: (3, H, W) tensor
    """
    # Background noise
    image = torch.rand(3, height, width) * 0.3

    # Thêm một số hình chữ nhật ngẫu nhiên (giả lập objects)
    num_objects = np.random.randint(1, 6)

    for _ in range(num_objects):
        # Random position and size
        cx = np.random.uniform(0.2, 0.8)
        cy = np.random.uniform(0.2, 0.8)
        w = np.random.uniform(0.1, 0.3)
        h = np.random.uniform(0.1, 0.3)

        # Convert to pixel coordinates
        x1 = int((cx - w/2) * width)
        y1 = int((cy - h/2) * height)
        x2 = int((cx + w/2) * width)
        y2 = int((cy + h/2) * height)

        # Random color
        color = torch.rand(3, 1, 1)

        # Draw rectangle
        image[:, y1:y2, x1:x2] = color

    return image


def generate_sample_annotations(num_objects=3, num_classes=80):
    """
    Tạo annotations mẫu cho một ảnh.

    Args:
        num_objects: Số lượng objects trong ảnh
        num_classes: Số lượng classes

    Returns:
        boxes: (N, 5) tensor - [cx, cy, w, h, class_id] normalized [0, 1]
    """
    boxes = []

    for _ in range(num_objects):
        # Random box (normalized coordinates)
        cx = np.random.uniform(0.2, 0.8)
        cy = np.random.uniform(0.2, 0.8)
        w = np.random.uniform(0.05, 0.3)
        h = np.random.uniform(0.05, 0.3)

        # Random class
        class_id = np.random.randint(0, num_classes)

        boxes.append([cx, cy, w, h, class_id])

    return torch.tensor(boxes, dtype=torch.float32)


def create_sample_batch(batch_size=4, num_classes=80):
    """
    Tạo một batch dữ liệu mẫu.

    Args:
        batch_size: Kích thước batch
        num_classes: Số lượng classes

    Returns:
        images: (B, 3, 640, 640) tensor
        targets: List of (N, 5) tensors - annotations cho mỗi ảnh
    """
    images = []
    targets = []

    for _ in range(batch_size):
        # Generate image
        image = generate_sample_image()
        images.append(image)

        # Generate annotations
        num_objects = np.random.randint(1, 6)
        boxes = generate_sample_annotations(num_objects, num_classes)
        targets.append(boxes)

    images = torch.stack(images)

    return images, targets


def save_sample_dataset(output_dir='sample_data', num_samples=100):
    """
    Tạo và lưu một dataset mẫu.

    Args:
        output_dir: Thư mục output
        num_samples: Số lượng samples
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create directories
    (output_path / 'images').mkdir(exist_ok=True)
    (output_path / 'labels').mkdir(exist_ok=True)

    print(f"Generating {num_samples} sample images...")

    for i in range(num_samples):
        # Generate image
        image = generate_sample_image()

        # Save image (as numpy array)
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        np.save(output_path / 'images' / f'sample_{i:04d}.npy', image_np)

        # Generate and save annotations
        num_objects = np.random.randint(1, 6)
        boxes = generate_sample_annotations(num_objects, num_classes=80)

        # Save in YOLO format: class_id cx cy w h (normalized)
        with open(output_path / 'labels' / f'sample_{i:04d}.txt', 'w') as f:
            for box in boxes:
                cx, cy, w, h, class_id = box.tolist()
                f.write(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

    # Create dataset config
    config = {
        'num_samples': num_samples,
        'num_classes': 80,
        'image_size': [640, 640],
        'format': 'YOLO',
        'description': 'Sample dataset for AURA-NET testing'
    }

    with open(output_path / 'dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n[OK] Sample dataset saved to: {output_path}")
    print(f"   - Images: {output_path / 'images'}")
    print(f"   - Labels: {output_path / 'labels'}")
    print(f"   - Config: {output_path / 'dataset_config.json'}")


def demo_training_batch():
    """
    Demo: Tạo một batch để test training loop.
    """
    print("=" * 60)
    print("AURA-NET - Sample Training Batch Demo")
    print("=" * 60)

    # Create sample batch
    batch_size = 4
    images, targets = create_sample_batch(batch_size=batch_size)

    print(f"\n[Batch Information]")
    print(f"   Images shape: {images.shape}")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of targets: {len(targets)}")

    print(f"\n[Sample Annotations]")
    for i, target in enumerate(targets):
        print(f"   Image {i}: {len(target)} objects")
        print(f"      Boxes shape: {target.shape}")
        if len(target) > 0:
            print(f"      First box: cx={target[0, 0]:.3f}, cy={target[0, 1]:.3f}, "
                  f"w={target[0, 2]:.3f}, h={target[0, 3]:.3f}, class={int(target[0, 4])}")

    # Test with model (if available)
    try:
        from models.v3 import AURANet
        from losses.aura_v3_loss import AURANetLoss

        print(f"\n[Testing with AURA-NET model...]")

        model = AURANet(num_classes=80)
        loss_fn = AURANetLoss(num_classes=80)

        # Forward pass
        with torch.no_grad():
            output = model(images)

        print(f"\n[Model Output]")
        print(f"   Boxes: {output['boxes'].shape}")
        print(f"   Classes: {output['classes'].shape}")
        print(f"   Scores: {output['scores'].shape}")
        print(f"   Evidence map: {output['evidence_map'].shape}")

        # Compute loss
        loss_dict = loss_fn(output, targets)

        print(f"\n[Loss Values]")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.item():.4f}")

        print(f"\n[OK] Training batch test successful!")

    except ImportError as e:
        print(f"\n[WARNING] Model not available: {e}")
        print(f"   Skipping model test.")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate sample training data')
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'generate'],
                       help='Mode: demo (test batch) or generate (create dataset)')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default='sample_data',
                       help='Output directory for generated dataset')

    args = parser.parse_args()

    if args.mode == 'demo':
        demo_training_batch()
    else:
        save_sample_dataset(args.output_dir, args.num_samples)
