"""
Advanced plotting utilities for training visualization
Includes: confusion matrix, PR curve, F1 curve, P curve, R curve, etc.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_confusion_matrix(confusion_matrix, class_names, save_path, normalize=False, title='Confusion Matrix'):
    """
    Plot confusion matrix

    Args:
        confusion_matrix: Confusion matrix array (N+1, N+1) including background
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize
        title: Plot title
    """
    # Add background class
    all_classes = ['background'] + list(class_names)

    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-6)
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=all_classes,
        yticklabels=all_classes,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()


def plot_pr_curve(px, py, ap, save_path, class_names=None):
    """
    Plot Precision-Recall curve

    Args:
        px: Confidence thresholds (1000,)
        py: Precision values per class (nc, 1000)
        ap: AP values per class (nc,)
        save_path: Path to save plot
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))

    # Plot per-class curves
    for i in range(len(ap)):
        class_name = class_names[i] if class_names and i < len(class_names) else f'Class {i}'
        plt.plot(px, py[i], linewidth=2, label=f'{class_name} AP={ap[i]:.3f}')

    # Plot mean
    plt.plot(px, py.mean(0), linewidth=3, color='blue', label=f'All Classes mAP@0.5={ap.mean():.3f}')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()


def plot_f1_curve(px, f1, save_path):
    """
    Plot F1-Confidence curve

    Args:
        px: Confidence thresholds (1000,)
        f1: F1 scores per class (nc, 1000)
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))

    # Plot per-class curves
    for i in range(f1.shape[0]):
        plt.plot(px, f1[i], linewidth=1, alpha=0.5)

    # Plot mean
    plt.plot(px, f1.mean(0), linewidth=3, color='blue', label=f'All Classes F1={f1.mean():.3f}')

    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1-Confidence Curve', fontsize=16, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()


def plot_p_curve(px, p, save_path):
    """
    Plot Precision-Confidence curve

    Args:
        px: Confidence thresholds (1000,)
        p: Precision values per class (nc, 1000)
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))

    # Plot per-class curves
    for i in range(p.shape[0]):
        plt.plot(px, p[i], linewidth=1, alpha=0.5)

    # Plot mean
    plt.plot(px, p.mean(0), linewidth=3, color='blue', label=f'All Classes P={p.mean():.3f}')

    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Confidence Curve', fontsize=16, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()


def plot_r_curve(px, r, save_path):
    """
    Plot Recall-Confidence curve

    Args:
        px: Confidence thresholds (1000,)
        r: Recall values per class (nc, 1000)
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))

    # Plot per-class curves
    for i in range(r.shape[0]):
        plt.plot(px, r[i], linewidth=1, alpha=0.5)

    # Plot mean
    plt.plot(px, r.mean(0), linewidth=3, color='blue', label=f'All Classes R={r.mean():.3f}')

    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Recall-Confidence Curve', fontsize=16, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()


def plot_labels_distribution(labels_list, class_names, save_path):
    """
    Plot labels distribution

    Args:
        labels_list: List of label arrays, each (N, 5) with [class, x, y, w, h]
        class_names: List of class names
        save_path: Path to save plot
    """
    # Count instances per class
    class_counts = np.zeros(len(class_names))

    for labels in labels_list:
        if len(labels) > 0:
            classes = labels[:, 0].astype(int)
            for c in classes:
                if c < len(class_names):
                    class_counts[c] += 1

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Class distribution
    ax = axes[0, 0]
    ax.bar(range(len(class_names)), class_counts, color='steelblue')
    ax.set_xlabel('Class', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Class Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Box width distribution
    ax = axes[0, 1]
    all_widths = []
    for labels in labels_list:
        if len(labels) > 0:
            all_widths.extend(labels[:, 3].tolist())
    if all_widths:
        ax.hist(all_widths, bins=50, color='coral', alpha=0.7)
    ax.set_xlabel('Box Width (normalized)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Box Width Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Box height distribution
    ax = axes[1, 0]
    all_heights = []
    for labels in labels_list:
        if len(labels) > 0:
            all_heights.extend(labels[:, 4].tolist())
    if all_heights:
        ax.hist(all_heights, bins=50, color='lightgreen', alpha=0.7)
    ax.set_xlabel('Box Height (normalized)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Box Height Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Box center distribution
    ax = axes[1, 1]
    all_x = []
    all_y = []
    for labels in labels_list:
        if len(labels) > 0:
            all_x.extend(labels[:, 1].tolist())
            all_y.extend(labels[:, 2].tolist())
    if all_x and all_y:
        ax.hist2d(all_x, all_y, bins=50, cmap='YlOrRd')
        ax.set_xlabel('X Center (normalized)', fontsize=10)
        ax.set_ylabel('Y Center (normalized)', fontsize=10)
        ax.set_title('Box Center Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()


def plot_training_results(results_csv, save_path):
    """
    Plot training results from CSV

    Args:
        results_csv: Path to results.csv
        save_path: Path to save plot
    """
    import pandas as pd

    try:
        df = pd.read_csv(results_csv)
    except:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss curves
    ax = axes[0, 0]
    if 'train_loss' in df.columns:
        ax.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in df.columns:
        ax.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. mAP curves
    ax = axes[0, 1]
    if 'mAP@0.5' in df.columns:
        ax.plot(df['epoch'], df['mAP@0.5'], label='mAP@0.5', linewidth=2)
    if 'mAP@0.5:0.95' in df.columns:
        ax.plot(df['epoch'], df['mAP@0.5:0.95'], label='mAP@0.5:0.95', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('mAP', fontsize=10)
    ax.set_title('mAP Curves', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Precision & Recall
    ax = axes[1, 0]
    if 'precision' in df.columns:
        ax.plot(df['epoch'], df['precision'], label='Precision', linewidth=2)
    if 'recall' in df.columns:
        ax.plot(df['epoch'], df['recall'], label='Recall', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Precision & Recall', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. F1 Score
    ax = axes[1, 1]
    if 'f1' in df.columns:
        ax.plot(df['epoch'], df['f1'], label='F1 Score', linewidth=2, color='green')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('F1 Score', fontsize=10)
    ax.set_title('F1 Score', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()


def plot_batch_images(images, labels, class_names, save_path, max_images=16):
    """
    Plot a batch of images with labels

    Args:
        images: Batch of images (B, 3, H, W) or (B, H, W, 3)
        labels: List of label arrays, each (N, 5) with [class, x, y, w, h]
        class_names: List of class names
        save_path: Path to save plot
        max_images: Maximum number of images to plot
    """
    import torch

    # Convert to numpy if needed
    if torch.is_tensor(images):
        if images.shape[1] == 3:
            images = images.permute(0, 2, 3, 1)
        images = images.cpu().numpy()

    # Limit number of images
    n_images = min(len(images), max_images)
    images = images[:n_images]
    labels = labels[:n_images]

    # Calculate grid size
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get image and labels
        img = images[idx]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        h, w = img.shape[:2]

        # Draw boxes
        img_draw = img.copy()
        if len(labels[idx]) > 0:
            for label in labels[idx]:
                cls_id = int(label[0])
                x_center, y_center, box_w, box_h = label[1:5]

                # Convert to pixel coordinates
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)

                # Draw box
                color = tuple(int(c * 255) for c in plt.cm.tab10(cls_id % 10)[:3])
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

                # Draw label
                class_name = class_names[cls_id] if cls_id < len(class_names) else f'C{cls_id}'
                cv2.putText(img_draw, class_name, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ax.imshow(img_draw)
        ax.axis('off')
        ax.set_title(f'Image {idx}', fontsize=10)

    # Hide empty subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
