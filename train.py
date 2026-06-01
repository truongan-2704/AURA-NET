"""
Training script for AURA-NET v3.0 with full metrics and visualization
Evidence-Centric Object Detection
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.v3 import AURANetV3
from losses import AURANetLoss
from data.dataset import create_dataloader
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.model_info import print_model_info
from utils.logger import setup_logger
from utils.metrics import MetricsCalculator
from utils.boxes import xywh2xyxy
from utils.plot import (
    plot_confusion_matrix, plot_pr_curve, plot_f1_curve,
    plot_p_curve, plot_r_curve, plot_labels_distribution,
    plot_training_results, plot_batch_images
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train AURA-Net')
    parser.add_argument('--config', type=str, default='configs/aura_net_s.yaml',
                       help='Path to config file')
    parser.add_argument('--data', type=str, default='configs/dataset.yaml',
                       help='Path to dataset config')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    return parser.parse_args()


def print_banner():
    """Print YOLO-style banner"""
    banner = """
    ======================================================================

         ###    ##  ##  #####    ###           ##  ##  #####  #####
        ## ##   ##  ##  ##  ##  ## ##          ### ##  ##       ##
        #####   ##  ##  #####   #####  #####   ######  #####    ##
        ## ##   ##  ##  ## ##   ## ##          ## ###  ##       ##
        ## ##    ####   ##  ##  ## ##          ##  ##  #####    ##

              AURA-NET v3.0 - Evidence-Centric Detection
                    Adaptive Evidence-Based Architecture

    ======================================================================
    """
    print(banner)


def print_hyperparameters(config, data_config, device, train_loader, val_loader):
    """Print hyperparameters in YOLO style"""
    print("\n" + "="*70)
    print("HYPERPARAMETERS".center(70))
    print("="*70)

    # Model info
    print(f"\n{'Model Configuration:':<30}")
    print(f"  {'Architecture:':<28} AURA-NET v3.0 (Evidence-Centric)")
    print(f"  {'Input size:':<28} {config['model']['img_size']}x{config['model']['img_size']}")
    print(f"  {'Classes:':<28} {config['model']['num_classes']}")
    print(f"  {'Evidence regions:':<28} {config['model'].get('num_evidence_regions', 100)}")
    print(f"  {'Decoder layers:':<28} {config['model'].get('num_decoder_layers', 3)}")

    # Training config
    print(f"\n{'Training Configuration:':<30}")
    print(f"  {'Epochs:':<28} {config['train']['epochs']}")
    print(f"  {'Batch size:':<28} {config['train']['batch_size']}")
    print(f"  {'Optimizer:':<28} {config['train']['optimizer']}")
    print(f"  {'Learning rate:':<28} {config['train']['lr']}")
    print(f"  {'Weight decay:':<28} {config['train']['weight_decay']}")
    print(f"  {'Scheduler:':<28} {config['train']['scheduler']}")
    print(f"  {'AMP:':<28} {config['train']['amp']}")
    print(f"  {'Device:':<28} {device}")

    # Dataset info
    print(f"\n{'Dataset Configuration:':<30}")
    print(f"  {'Train images:':<28} {len(train_loader.dataset)}")
    print(f"  {'Val images:':<28} {len(val_loader.dataset)}")
    print(f"  {'Train batches:':<28} {len(train_loader)}")
    print(f"  {'Val batches:':<28} {len(val_loader)}")

    # Loss weights
    print(f"\n{'Loss Weights:':<30}")
    for key, value in config['loss'].items():
        print(f"  {key + ':':<28} {value}")

    print("="*70 + "\n")


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, total_epochs, logger, img_size):
    """Train for one epoch with YOLO-style display"""
    model.train()

    total_loss = 0
    loss_components = {
        'evidence': 0, 'box': 0, 'class': 0, 'uncertainty': 0
    }

    num_batches = len(dataloader)
    num_instances = 0

    # YOLO-style progress bar with tqdm
    pbar = tqdm(dataloader, total=num_batches,
                bar_format='{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                ncols=120)

    for imgs, targets, paths in pbar:
        imgs = imgs.to(device)

        # Count instances in this batch
        batch_instances = sum(len(t) for t in targets) if isinstance(targets, list) else len(targets)
        num_instances += batch_instances

        # Forward pass with AMP
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type=device_type, enabled=(device.type == 'cuda')):
            outputs = model(imgs)

            # Extract outputs
            boxes = outputs['boxes']  # [B, N, 4]
            classes = outputs['classes']  # [B, N, num_classes]
            scores = outputs['scores']  # [B, N]
            evidence_map = outputs['evidence_map']  # [B, 1, H, W]
            uncertainty = outputs['uncertainty']  # [B, N, 2]

            # Compute loss
            loss, loss_dict = criterion(
                boxes, classes, scores, evidence_map, uncertainty, targets
            )

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]

        # YOLO-style description
        mem = f'{torch.cuda.memory_reserved() / 1E9:.2f}G' if torch.cuda.is_available() else '0.00G'

        desc = (f"{epoch + 1:>6}/{total_epochs:<3} "
                f"{mem:>10} "
                f"{loss_dict['box']:>10.3f} "
                f"{loss_dict['class']:>10.3f} "
                f"{loss_dict['evidence']:>10.3f} "
                f"{batch_instances:>10} "
                f"{img_size:>10}")

        pbar.set_description(desc)

    # Average losses
    avg_loss = total_loss / len(dataloader)
    for key in loss_components:
        loss_components[key] /= len(dataloader)

    avg_instances = num_instances / num_batches

    return avg_loss, loss_components, avg_instances


def validate(model, dataloader, criterion, device, config, class_names):
    """
    Run validation with full metrics

    Returns:
        val_loss: Average validation loss
        metrics: Dictionary of metrics
    """
    model.eval()

    conf_thres = config['val']['conf_thres']
    iou_thres = config['val']['iou_thres']
    num_classes = config['model']['num_classes']

    metrics_calc = MetricsCalculator(num_classes=num_classes, conf_thres=conf_thres, iou_thres=iou_thres)

    total_loss = 0
    loss_components = {'evidence': 0, 'box': 0, 'class': 0}

    with torch.no_grad():
        for imgs, targets, paths in tqdm(dataloader, desc='Validation', ncols=120):
            imgs = imgs.to(device)

            # Forward pass
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with autocast(device_type=device_type, enabled=(device.type == 'cuda')):
                outputs = model(imgs)

                # Extract outputs
                boxes = outputs['boxes']
                classes = outputs['classes']
                scores = outputs['scores']
                evidence_map = outputs['evidence_map']
                uncertainty = outputs['uncertainty']

                # Compute loss
                loss, loss_dict = criterion(
                    boxes, classes, scores, evidence_map, uncertainty, targets
                )

            total_loss += loss.item()
            loss_components['evidence'] += loss_dict['evidence']
            loss_components['box'] += loss_dict['box']
            loss_components['class'] += loss_dict['class']

            # Decode predictions for metrics
            # Convert model outputs to detection format
            batch_size = boxes.shape[0]
            detections = []

            for b in range(batch_size):
                # Get predictions for this image
                pred_boxes = boxes[b]  # [N, 4]
                pred_scores = scores[b]  # [N]
                pred_classes = classes[b].argmax(dim=-1)  # [N]

                # Filter by confidence
                mask = pred_scores > conf_thres
                if mask.sum() > 0:
                    det = torch.cat([
                        pred_boxes[mask],
                        pred_scores[mask].unsqueeze(1),
                        pred_classes[mask].unsqueeze(1).float()
                    ], dim=1)
                    detections.append(det)
                else:
                    detections.append(torch.zeros((0, 6), device=device))

            # Process each image in batch
            for i, (det, target) in enumerate(zip(detections, targets)):
                if len(det) == 0:
                    pred_boxes = torch.zeros((0, 4), device=device)
                    pred_classes = torch.zeros((0,), device=device, dtype=torch.long)
                    pred_scores = torch.zeros((0,), device=device)
                else:
                    # Extract predictions: [x1, y1, x2, y2, score, class]
                    pred_boxes = det[:, :4]
                    pred_scores = det[:, 4]
                    pred_classes = det[:, 5].long()

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

    # Average losses
    val_loss = total_loss / len(dataloader)
    for key in loss_components:
        loss_components[key] /= len(dataloader)

    # Compute metrics
    metrics = metrics_calc.compute()
    metrics['val_loss'] = val_loss
    metrics['loss_components'] = loss_components

    return val_loss, metrics


def save_results_csv(results, save_path):
    """Save training results to CSV"""
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)


def main():
    args = parse_args()

    # Print banner
    print_banner()

    # Load configs
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)

    # Override batch size if provided
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
        config['val']['batch_size'] = args.batch_size

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Setup save directory
    save_dir = Path(config['train']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger('AURA-Net', log_file=save_dir / 'train.log')
    logger.info(f'Training AURA-Net with config: {args.config}')
    logger.info(f'Dataset config: {args.data}')

    # Set seed for reproducibility
    seed = config['train'].get('seed', 42)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    # Build model
    logger.info('Building model...')
    model = AURANetV3(
        num_classes=data_config['nc'],
        channels=[64, 128, 256, 512],
        num_evidence_regions=config['model'].get('num_evidence_regions', 100),
        num_decoder_layers=config['model'].get('num_decoder_layers', 3)
    )
    model = model.to(device)

    # Update config with actual num_classes from data
    config['model']['num_classes'] = data_config['nc']
    class_names = data_config['names']

    # Create dataloaders
    logger.info('Creating dataloaders...')
    train_loader = create_dataloader(
        data_config,
        img_size=config['model']['img_size'],
        batch_size=config['train']['batch_size'],
        split='train',
        shuffle=True,
        num_workers=4
    )

    val_loader = create_dataloader(
        data_config,
        img_size=config['model']['img_size'],
        batch_size=config['val']['batch_size'],
        split='val',
        shuffle=False,
        num_workers=4
    )

    # Print hyperparameters
    print_hyperparameters(config, data_config, device, train_loader, val_loader)

    # Print model info
    print_model_info(model, config['model']['img_size'], device)

    # Setup loss
    criterion = AURANetLoss(
        num_classes=config['model']['num_classes']
    )

    # Setup optimizer
    optimizer_name = config['train']['optimizer'].lower()
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['train']['lr'],
            momentum=0.9,
            weight_decay=config['train']['weight_decay']
        )
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    # Setup scheduler
    scheduler_name = config['train']['scheduler'].lower()
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train']['epochs']
        )
    else:
        scheduler = None

    # Setup AMP scaler
    scaler = GradScaler(enabled=(device.type == 'cuda' and config['train']['amp']))

    # Resume from checkpoint
    start_epoch = 0
    best_map = 0.0

    if args.resume or config['train'].get('resume'):
        resume_path = args.resume or config['train']['resume']
        logger.info(f'Resuming from checkpoint: {resume_path}')
        start_epoch, _ = load_checkpoint(model, resume_path, optimizer, scheduler, device)
        start_epoch += 1

    # Collect labels for distribution plot
    logger.info('Collecting labels for distribution plot...')
    all_labels = []
    for _, labels, _ in train_loader:
        all_labels.extend(labels)
    plot_labels_distribution(all_labels, class_names, save_dir / 'labels.jpg')

    # Plot first training batch
    logger.info('Plotting training batch samples...')
    for batch_idx, (imgs, labels, _) in enumerate(train_loader):
        if batch_idx < 3:
            plot_batch_images(imgs, labels, class_names, save_dir / f'train_batch{batch_idx}.jpg')
        else:
            break

    # Training loop
    print("\n" + "="*100)
    print("TRAINING".center(100))
    print("="*100)

    # YOLO-style header
    print(f"\n{'Epoch':<10}{'GPU_mem':<12}{'box_loss':<12}{'cls_loss':<12}{'evd_loss':<12}{'Instances':<12}{'Size':<10}")

    results = []

    for epoch in range(start_epoch, config['train']['epochs']):
        # Train
        train_loss, train_loss_dict, avg_instances = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, config['train']['epochs'], logger, config['model']['img_size']
        )

        # Validation
        val_loss, metrics = validate(model, val_loader, criterion, device, config, class_names)

        # Extract metrics
        map50 = metrics['mAP@0.5']
        map50_95 = metrics['mAP@0.5:0.95']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']

        # Print validation summary
        print(f"\nEpoch {epoch + 1}/{config['train']['epochs']} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"  mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map50_95:.4f}")

        # Check if best model
        is_best = map50_95 > best_map
        if is_best:
            best_map = map50_95
            print(f"  *** NEW BEST MODEL *** (mAP@0.5:0.95: {map50_95:.4f})")
        print()

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss,
            weights_dir / 'last.pt', is_best,
            config=config, class_names=class_names, metrics=metrics
        )

        # Save results
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mAP@0.5': map50,
            'mAP@0.5:0.95': map50_95,
            'box_loss': train_loss_dict['box'],
            'cls_loss': train_loss_dict['class'],
            'evd_loss': train_loss_dict['evidence']
        })

        # Save results CSV
        save_results_csv(results, save_dir / 'results.csv')

        # Plot curves every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == config['train']['epochs'] - 1:
            plot_training_results(save_dir / 'results.csv', save_dir / 'results.png')

            # Plot confusion matrix
            if 'confusion_matrix' in metrics:
                plot_confusion_matrix(
                    metrics['confusion_matrix'], class_names,
                    save_dir / 'confusion_matrix.png', normalize=False
                )
                plot_confusion_matrix(
                    metrics['confusion_matrix'], class_names,
                    save_dir / 'confusion_matrix_normalized.png', normalize=True
                )

            # Plot PR curve
            if 'p_curve' in metrics and 'r_curve' in metrics:
                plot_pr_curve(
                    metrics['r_curve'].mean(0), metrics['p_curve'],
                    metrics['per_class_ap'], save_dir / 'PR_curve.png', class_names
                )

            # Plot F1 curve
            if 'f1_curve' in metrics:
                plot_f1_curve(metrics['px'], metrics['f1_curve'], save_dir / 'F1_curve.png')

            # Plot P curve
            if 'p_curve' in metrics:
                plot_p_curve(metrics['px'], metrics['p_curve'], save_dir / 'P_curve.png')

            # Plot R curve
            if 'r_curve' in metrics:
                plot_r_curve(metrics['px'], metrics['r_curve'], save_dir / 'R_curve.png')

    # Training completed
    print("\n" + "="*70)
    print("TRAINING COMPLETED".center(70))
    print("="*70)
    print(f"\nBest mAP@0.5:0.95: {best_map:.4f}")
    print(f"Results saved to: {save_dir}")
    print(f"  - Best model: {weights_dir / 'best.pt'}")
    print(f"  - Last model: {weights_dir / 'last.pt'}")
    print(f"  - Training log: {save_dir / 'train.log'}")
    print(f"  - Results CSV: {save_dir / 'results.csv'}")
    print(f"  - Results plot: {save_dir / 'results.png'}")
    print("="*70 + "\n")

    logger.info('Training completed!')
    logger.info(f'Best mAP@0.5:0.95: {best_map:.4f}')


if __name__ == '__main__':
    main()
