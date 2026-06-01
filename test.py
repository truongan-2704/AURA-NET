"""
Test script for AURA-Net
Evaluates model on test set with full metrics and visualization
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.aura_net import build_aura_net
from data.dataset import create_dataloader
from utils.checkpoint import load_model_weights
from utils.metrics import MetricsCalculator
from utils.boxes import xywh2xyxy
from utils.logger import setup_logger
from utils.plot import (
    plot_confusion_matrix, plot_pr_curve, plot_f1_curve,
    plot_p_curve, plot_r_curve, plot_batch_images
)


def parse_args():
    parser = argparse.ArgumentParser(description='Test AURA-Net')
    parser.add_argument('--config', type=str, default='configs/aura_net_s.yaml',
                       help='Path to config file')
    parser.add_argument('--data', type=str, default='configs/dataset.yaml',
                       help='Path to dataset config')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--save-dir', type=str, default='runs/test',
                       help='Directory to save results')
    return parser.parse_args()


def test(model, dataloader, device, conf_thres, iou_thres, num_classes, class_names, save_dir):
    """
    Run testing with full metrics

    Returns:
        metrics: Dictionary of metrics
    """
    model.eval()

    metrics_calc = MetricsCalculator(
        num_classes=num_classes,
        conf_thres=conf_thres,
        iou_thres=iou_thres
    )

    # Store first batch for visualization
    first_batch_imgs = None
    first_batch_labels = None
    first_batch_preds = None

    with torch.no_grad():
        for batch_idx, (imgs, targets, paths) in enumerate(tqdm(dataloader, desc='Testing')):
            imgs = imgs.to(device)

            # Forward pass
            predictions = model(imgs)

            # Decode predictions
            detections = model.decoder.decode_predictions(predictions, conf_thres)

            # Apply NMS
            from utils.boxes import non_max_suppression
            detections = non_max_suppression(detections, iou_thres)

            # Store first batch for visualization
            if batch_idx == 0:
                first_batch_imgs = imgs.cpu()
                first_batch_labels = targets
                first_batch_preds = detections

            # Process each image in batch
            for i, (det, target) in enumerate(zip(detections, targets)):
                if len(det) == 0:
                    pred_boxes = torch.zeros((0, 4), device=device)
                    pred_classes = torch.zeros((0,), device=device, dtype=torch.long)
                    pred_scores = torch.zeros((0,), device=device)
                else:
                    # Extract predictions
                    pred_boxes = det[:, :4]  # x1, y1, x2, y2 (normalized)
                    pred_scores = det[:, 4] * det[:, 5]  # obj_conf * class_conf
                    pred_classes = det[:, 6].long()

                # Extract ground truth
                if target is None or len(target) == 0:
                    gt_boxes = torch.zeros((0, 4), device=device)
                    gt_classes = torch.zeros((0,), device=device, dtype=torch.long)
                else:
                    target = target.to(device)
                    gt_classes = target[:, 0].long()
                    gt_boxes_xywh = target[:, 1:5]  # x, y, w, h (normalized)
                    gt_boxes = xywh2xyxy(gt_boxes_xywh)

                # Update metrics
                metrics_calc.update(
                    [pred_boxes], [pred_classes], [pred_scores],
                    [gt_boxes], [gt_classes]
                )

    # Compute metrics
    metrics = metrics_calc.compute()

    # Plot visualizations
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    if 'confusion_matrix' in metrics:
        plot_confusion_matrix(
            metrics['confusion_matrix'], class_names,
            save_dir / 'confusion_matrix.png', normalize=False
        )
        plot_confusion_matrix(
            metrics['confusion_matrix'], class_names,
            save_dir / 'confusion_matrix_normalized.png', normalize=True
        )

    # PR curve
    if 'p_curve' in metrics and 'r_curve' in metrics:
        plot_pr_curve(
            metrics['r_curve'].mean(0), metrics['p_curve'],
            metrics['per_class_ap'], save_dir / 'PR_curve.png', class_names
        )

    # F1 curve
    if 'f1_curve' in metrics:
        plot_f1_curve(metrics['px'], metrics['f1_curve'], save_dir / 'F1_curve.png')

    # P curve
    if 'p_curve' in metrics:
        plot_p_curve(metrics['px'], metrics['p_curve'], save_dir / 'P_curve.png')

    # R curve
    if 'r_curve' in metrics:
        plot_r_curve(metrics['px'], metrics['r_curve'], save_dir / 'R_curve.png')

    # Plot test batch with ground truth
    if first_batch_imgs is not None:
        plot_batch_images(
            first_batch_imgs, first_batch_labels, class_names,
            save_dir / 'test_batch_labels.jpg'
        )

    # Plot test batch with predictions
    if first_batch_imgs is not None and first_batch_preds is not None:
        # Convert predictions to label format for visualization
        pred_labels = []
        for det in first_batch_preds:
            if len(det) > 0:
                # Convert from [x1, y1, x2, y2, obj_conf, cls_conf, cls_id, ...] to [cls, x, y, w, h]
                boxes_xyxy = det[:, :4]
                cls_ids = det[:, 6]

                # Convert to xywh
                x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                pred_label = torch.stack([cls_ids, x_center, y_center, w, h], dim=1)
                pred_labels.append(pred_label.cpu().numpy())
            else:
                pred_labels.append([])

        plot_batch_images(
            first_batch_imgs, pred_labels, class_names,
            save_dir / 'test_batch_pred.jpg'
        )

    return metrics


def main():
    args = parse_args()

    # Load configs
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'Using device: {device}')

    # Setup save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger('AURA-Net-Test', log_file=save_dir / 'test.log')
    logger.info(f'Testing AURA-Net with weights: {args.weights}')

    # Update config
    config['model']['num_classes'] = data_config['nc']
    class_names = data_config['names']

    # Build model
    logger.info('Building model...')
    model = build_aura_net(config)

    # Load weights
    logger.info(f'Loading weights from {args.weights}')
    model = load_model_weights(model, args.weights, device)
    model = model.to(device)

    # Create dataloader
    logger.info('Creating test dataloader...')
    test_loader = create_dataloader(
        data_config,
        img_size=config['model']['img_size'],
        batch_size=args.batch_size,
        split='test',
        shuffle=False,
        num_workers=4
    )

    logger.info(f'Test images: {len(test_loader.dataset)}')
    logger.info(f'Test batches: {len(test_loader)}')

    # Run testing
    logger.info('Running test...')
    metrics = test(
        model, test_loader, device,
        args.conf_thres, args.iou_thres,
        config['model']['num_classes'], class_names, save_dir
    )

    # Print results
    print('\n' + '='*70)
    print('TEST RESULTS'.center(70))
    print('='*70)
    print(f'Precision:        {metrics["precision"]:.4f}')
    print(f'Recall:           {metrics["recall"]:.4f}')
    print(f'F1 Score:         {metrics["f1"]:.4f}')
    print(f'mAP@0.5:          {metrics["mAP@0.5"]:.4f}')
    print(f'mAP@0.5:0.95:     {metrics["mAP@0.5:0.95"]:.4f}')
    print('='*70)

    # Per-class AP
    if 'per_class_ap' in metrics and len(metrics['per_class_ap']) > 0:
        print('\nPer-Class AP@0.5:')
        for i, ap in enumerate(metrics['per_class_ap']):
            class_name = class_names[i] if i < len(class_names) else f'Class {i}'
            print(f'  {class_name:<20} {ap:.4f}')
        print('='*70)

    # Save results to CSV
    results_dict = {
        'metric': ['Precision', 'Recall', 'F1', 'mAP@0.5', 'mAP@0.5:0.95'],
        'value': [
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['mAP@0.5'],
            metrics['mAP@0.5:0.95']
        ]
    }
    df = pd.DataFrame(results_dict)
    df.to_csv(save_dir / 'test_results.csv', index=False)

    logger.info(f'Results saved to: {save_dir}')
    logger.info(f'  - Test results CSV: {save_dir / "test_results.csv"}')
    logger.info(f'  - Confusion matrix: {save_dir / "confusion_matrix.png"}')
    logger.info(f'  - PR curve: {save_dir / "PR_curve.png"}')
    logger.info(f'  - F1 curve: {save_dir / "F1_curve.png"}')

    print(f'\nResults saved to: {save_dir}')


if __name__ == '__main__':
    main()
