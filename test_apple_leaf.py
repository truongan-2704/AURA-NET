"""
AURA-NET - Inference Script for Apple Leaf Detection
=====================================================

Test trained model on Apple Leaf Detection dataset.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import argparse
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.v3 import AURANet


# Class names
CLASS_NAMES = ['Black Rot', 'Powdery Mildew', 'Rust', 'Scab']

# Colors for visualization (RGB)
COLORS = [
    (255, 0, 0),      # Red - Black Rot
    (0, 255, 0),      # Green - Powdery Mildew
    (0, 0, 255),      # Blue - Rust
    (255, 255, 0),    # Yellow - Scab
]


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    model = AURANet(
        num_classes=4,
        feature_dim=256,
        min_K=50,
        max_K=150,
        num_iterations=3
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")

    return model


def preprocess_image(image_path, img_size=640):
    """Load and preprocess image."""
    image = Image.open(image_path).convert('RGB')
    orig_size = image.size

    # Resize
    image_resized = image.resize((img_size, img_size), Image.BILINEAR)

    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    return image, image_tensor, orig_size


def draw_predictions(image, predictions, conf_threshold=0.25):
    """Draw bounding boxes on image."""
    draw = ImageDraw.Draw(image)
    width, height = image.size

    boxes = predictions['boxes']
    scores = predictions['scores']
    class_ids = predictions['class_ids']

    for box, score, class_id in zip(boxes, scores, class_ids):
        if score < conf_threshold:
            continue

        # Convert normalized coordinates to pixel coordinates
        cx, cy, w, h = box
        x1 = int((cx - w/2) * width)
        y1 = int((cy - h/2) * height)
        x2 = int((cx + w/2) * width)
        y2 = int((cy + h/2) * height)

        # Get color and class name
        color = COLORS[int(class_id)]
        class_name = CLASS_NAMES[int(class_id)]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label = f"{class_name}: {score:.2f}"

        # Draw label background
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)

    return image


@torch.no_grad()
def predict_image(model, image_path, device, img_size=640, conf_threshold=0.25, save_path=None):
    """Predict on a single image."""
    # Load and preprocess
    image, image_tensor, orig_size = preprocess_image(image_path, img_size)
    image_tensor = image_tensor.to(device)

    # Predict
    predictions = model.predict(image_tensor, conf_threshold=conf_threshold)[0]

    # Move to CPU
    predictions = {k: v.cpu() for k, v in predictions.items()}

    # Draw predictions
    result_image = draw_predictions(image.copy(), predictions, conf_threshold)

    # Save if specified
    if save_path:
        result_image.save(save_path)
        print(f"Saved to {save_path}")

    # Print detections
    num_detections = len(predictions['boxes'])
    print(f"\nDetections: {num_detections}")
    for i, (box, score, class_id) in enumerate(zip(
        predictions['boxes'], predictions['scores'], predictions['class_ids']
    )):
        if score >= conf_threshold:
            print(f"  {i+1}. {CLASS_NAMES[int(class_id)]}: {score:.3f}")

    return result_image, predictions


def test_dataset(model, data_dir, split='test', device='cuda', img_size=640,
                conf_threshold=0.25, save_dir=None):
    """Test on entire dataset split."""
    data_path = Path(data_dir) / split / 'images'
    image_files = sorted(list(data_path.glob('*.jpg')))

    print(f"\nTesting on {len(image_files)} images from {split} set...")

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    results = []

    for img_path in tqdm(image_files, desc=f"Testing {split}"):
        # Predict
        image, image_tensor, orig_size = preprocess_image(img_path, img_size)
        image_tensor = image_tensor.to(device)

        predictions = model.predict(image_tensor, conf_threshold=conf_threshold)[0]
        predictions = {k: v.cpu() for k, v in predictions.items()}

        # Save visualization if specified
        if save_dir:
            result_image = draw_predictions(image.copy(), predictions, conf_threshold)
            result_image.save(save_path / img_path.name)

        # Store results
        results.append({
            'image': img_path.name,
            'num_detections': len(predictions['boxes']),
            'detections': [
                {
                    'class': CLASS_NAMES[int(cid)],
                    'confidence': float(score),
                    'box': box.tolist()
                }
                for box, score, cid in zip(
                    predictions['boxes'],
                    predictions['scores'],
                    predictions['class_ids']
                )
                if score >= conf_threshold
            ]
        })

    # Save results
    if save_dir:
        with open(save_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path / 'results.json'}")

    # Print statistics
    total_detections = sum(r['num_detections'] for r in results)
    avg_detections = total_detections / len(results)

    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  Total images: {len(results)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections/image: {avg_detections:.2f}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test AURA-NET on Apple Leaf Detection')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Single image to test')
    parser.add_argument('--data-dir', type=str, default='datasets/apple_leaft_detection',
                       help='Path to dataset')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split to test')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save results')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("AURA-NET - Apple Leaf Detection Inference")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print("=" * 60)

    # Load model
    model = load_model(args.checkpoint, device)

    # Test single image or dataset
    if args.image:
        print(f"\nTesting single image: {args.image}")
        save_path = args.save_dir / Path(args.image).name if args.save_dir else None
        result_image, predictions = predict_image(
            model, args.image, device, args.img_size, args.conf_threshold, save_path
        )

        # Show image if possible
        try:
            result_image.show()
        except:
            print("Cannot display image")
    else:
        # Test on dataset
        results = test_dataset(
            model, args.data_dir, args.split, device,
            args.img_size, args.conf_threshold, args.save_dir
        )


if __name__ == '__main__':
    main()
