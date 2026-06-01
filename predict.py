"""
Prediction script for AURA-NET v3.0 with YOLO-style interface
Evidence-Centric Object Detection
"""

import os
import sys
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.v3 import AURANetV3
from utils.checkpoint import load_model_weights
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Predict with AURA-Net')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image or directory')
    parser.add_argument('--data', type=str, default='configs/dataset.yaml',
                       help='Path to dataset config (for class names)')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640,
                       help='Inference image size')
    parser.add_argument('--conf', '--conf-thres', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', '--iou-thres', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use (0, 1, cpu)')
    parser.add_argument('--project', type=str, default='runs/predict',
                       help='Save results to project/name')
    parser.add_argument('--name', type=str, default='exp',
                       help='Save results to project/name')
    parser.add_argument('--save', action='store_true',
                       help='Save results')
    parser.add_argument('--nosave', action='store_true',
                       help='Do not save results')
    parser.add_argument('--visualize-evidence', action='store_true',
                       help='Visualize evidence map')
    parser.add_argument('--line-thickness', type=int, default=2,
                       help='Bounding box thickness')
    parser.add_argument('--hide-labels', action='store_true',
                       help='Hide labels')
    parser.add_argument('--hide-conf', action='store_true',
                       help='Hide confidence scores')
    return parser.parse_args()


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image while maintaining aspect ratio

    Args:
        img: Input image
        new_shape: Target shape (height, width)
        color: Padding color

    Returns:
        img: Resized and padded image
        ratio: Resize ratio
        (dw, dh): Padding
    """
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def preprocess_image(img_path, img_size=640):
    """
    Preprocess image for inference

    Args:
        img_path: Path to image
        img_size: Target image size

    Returns:
        img_tensor: Preprocessed image tensor (1, 3, H, W)
        img_orig: Original image (H, W, 3) in BGR
        ratio: Resize ratio
        pad: Padding (dw, dh)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f'Failed to load image: {img_path}')

    img_orig = img.copy()

    # Letterbox
    img, ratio, pad = letterbox(img, new_shape=(img_size, img_size))

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # To tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, img_orig, ratio, pad


def scale_boxes(boxes, img_shape, img_orig_shape, ratio, pad):
    """
    Scale boxes from model output to original image coordinates

    Args:
        boxes: Boxes in normalized coordinates (N, 4) [x1, y1, x2, y2]
        img_shape: Model input shape (H, W)
        img_orig_shape: Original image shape (H, W)
        ratio: Resize ratio
        pad: Padding (dw, dh)

    Returns:
        boxes: Scaled boxes in original image coordinates
    """
    # Scale from normalized to model input size
    boxes[:, [0, 2]] *= img_shape[1]
    boxes[:, [1, 3]] *= img_shape[0]

    # Remove padding
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]

    # Scale to original image size
    boxes[:, :4] /= ratio

    # Clip to image bounds
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img_orig_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img_orig_shape[0])

    return boxes


def draw_boxes(img, detections, class_names, hide_labels=False, hide_conf=False, line_thickness=2):
    """
    Draw bounding boxes on image

    Args:
        img: Image array (H, W, 3) in BGR
        detections: Tensor (N, 9) [x1, y1, x2, y2, obj_conf, cls_conf, cls_id, evidence, uncertainty]
        class_names: List of class names
        hide_labels: Hide labels
        hide_conf: Hide confidence scores
        line_thickness: Box line thickness

    Returns:
        img: Image with boxes drawn
    """
    import matplotlib.pyplot as plt

    for det in detections:
        x1, y1, x2, y2 = det[:4].cpu().numpy().astype(int)
        obj_conf = det[4].item()
        cls_conf = det[5].item()
        cls_id = int(det[6].item())
        conf = obj_conf * cls_conf

        # Color based on class
        color = tuple(int(c * 255) for c in plt.cm.tab10(cls_id % 10)[:3])
        color = (color[2], color[1], color[0])  # RGB to BGR

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        # Draw label
        if not hide_labels:
            class_name = class_names[cls_id] if cls_id < len(class_names) else f'Class {cls_id}'

            if not hide_conf:
                label = f'{class_name} {conf:.2f}'
            else:
                label = class_name

            # Get label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw label background
            cv2.rectangle(img, (x1, y1 - label_h - baseline - 5),
                         (x1 + label_w, y1), color, -1)

            # Draw label text
            cv2.putText(img, label, (x1, y1 - baseline - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def visualize_evidence_map(img, evidence_map, alpha=0.5):
    """
    Overlay evidence map on image

    Args:
        img: Image array (H, W, 3) in BGR
        evidence_map: Evidence map tensor (1, H, W)
        alpha: Overlay transparency

    Returns:
        img_vis: Image with evidence overlay
    """
    # Convert to numpy
    evidence_map = evidence_map.squeeze().cpu().numpy()

    # Resize to match image
    h, w = img.shape[:2]
    evidence_map = cv2.resize(evidence_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to 0-255
    evidence_map = (evidence_map * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(evidence_map, cv2.COLORMAP_JET)

    # Overlay
    img_vis = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return img_vis


def main():
    args = parse_args()

    # Setup device
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')

    # Setup save directory
    save_dir = Path(args.project) / args.name
    if args.save and not args.nosave:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Load class names
    class_names = None
    if Path(args.data).exists():
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
            class_names = data_config.get('names', None)

    # Load model config from checkpoint
    checkpoint = torch.load(args.weights, map_location='cpu')
    if 'config' in checkpoint:
        config = checkpoint['config']
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
    else:
        # Fallback: use default config
        config_path = Path('configs/aura_net_s.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError('Could not find model config')

    # Build model
    print('Loading model...')
    num_classes = len(class_names) if class_names else 80
    model = AURANetV3(
        num_classes=num_classes,
        channels=[64, 128, 256, 512],
        num_evidence_regions=config.get('num_evidence_regions', 100),
        num_decoder_layers=config.get('num_decoder_layers', 3)
    )
    model = load_model_weights(model, args.weights, device)
    model = model.to(device)
    model.eval()

    # Get image files
    source_path = Path(args.source)
    if source_path.is_file():
        img_files = [source_path]
    elif source_path.is_dir():
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            img_files.extend(list(source_path.glob(ext)))
        img_files = sorted(img_files)
    else:
        raise ValueError(f'Invalid source: {args.source}')

    print(f'Found {len(img_files)} images')

    # Run inference
    print('Running inference...')

    with torch.no_grad():
        for img_file in tqdm(img_files):
            # Preprocess
            img_tensor, img_orig, ratio, pad = preprocess_image(img_file, args.imgsz)
            img_tensor = img_tensor.to(device)

            # Forward pass
            outputs = model(img_tensor)

            # Extract outputs
            boxes = outputs['boxes'][0]  # [N, 4]
            classes = outputs['classes'][0]  # [N, num_classes]
            scores = outputs['scores'][0]  # [N]
            evidence_map = outputs['evidence_map']  # [1, 1, H, W]

            # Filter by confidence
            mask = scores > args.conf
            boxes = boxes[mask]
            scores = scores[mask]
            pred_classes = classes[mask].argmax(dim=-1)

            # Convert to detection format [x1, y1, x2, y2, score, class]
            if len(boxes) > 0:
                det = torch.cat([
                    boxes,
                    scores.unsqueeze(1),
                    pred_classes.unsqueeze(1).float()
                ], dim=1)
            else:
                det = torch.zeros((0, 6), device=device)

            # Scale boxes to original image
            if len(det) > 0:
                det[:, :4] = scale_boxes(
                    det[:, :4],
                    (args.imgsz, args.imgsz),
                    img_orig.shape[:2],
                    ratio, pad
                )

            # Draw boxes
            img_vis = img_orig.copy()
            if len(det) > 0:
                img_vis = draw_boxes(
                    img_vis, det, class_names or [],
                    args.hide_labels, args.hide_conf, args.line_thickness
                )

            # Print detections
            print(f'\n{img_file.name}: {len(det)} detections')
            for d in det:
                cls_id = int(d[5].item())
                conf = d[4].item()
                class_name = class_names[cls_id] if class_names and cls_id < len(class_names) else f'Class {cls_id}'
                print(f'  {class_name}: {conf:.3f}')

            # Save results
            if args.save and not args.nosave:
                save_path = save_dir / img_file.name
                cv2.imwrite(str(save_path), img_vis)

                # Save evidence map
                if args.visualize_evidence:
                    img_evidence = visualize_evidence_map(img_orig, evidence_map)
                    evidence_path = save_dir / f'{img_file.stem}_evidence{img_file.suffix}'
                    cv2.imwrite(str(evidence_path), img_evidence)

    if args.save and not args.nosave:
        print(f'\nResults saved to: {save_dir}')


if __name__ == '__main__':
    main()
