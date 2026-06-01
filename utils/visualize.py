"""
Visualization utilities
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def visualize_detections(img, detections, class_names=None, conf_thres=0.25, 
                        save_path=None, show_evidence=True):
    """
    Visualize detection results
    
    Args:
        img: Image array (H, W, 3) in RGB or tensor (3, H, W)
        detections: Tensor (N, 9) containing [x1, y1, x2, y2, obj_conf, class_conf, class_id, evidence, uncertainty]
        class_names: List of class names
        conf_thres: Confidence threshold for display
        save_path: Path to save visualization
        show_evidence: Whether to show evidence and uncertainty scores
    
    Returns:
        img_vis: Visualized image
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(img):
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        img = (img.cpu().numpy() * 255).astype(np.uint8)
    
    img_vis = img.copy()
    h, w = img_vis.shape[:2]
    
    if len(detections) == 0:
        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
        return img_vis
    
    # Filter by confidence
    conf = detections[:, 4] * detections[:, 5]
    mask = conf >= conf_thres
    detections = detections[mask]
    
    # Draw boxes
    for det in detections:
        x1, y1, x2, y2 = det[:4].cpu().numpy()
        obj_conf = det[4].item()
        cls_conf = det[5].item()
        cls_id = int(det[6].item())
        evidence = det[7].item()
        uncertainty = det[8].item()
        
        # Scale to image size
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        
        # Color based on class
        color = tuple(int(c) for c in plt.cm.tab10(cls_id % 10)[:3] * 255)
        
        # Draw box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        if class_names and cls_id < len(class_names):
            label = f'{class_names[cls_id]}'
        else:
            label = f'Class {cls_id}'
        
        label += f' {obj_conf * cls_conf:.2f}'
        
        if show_evidence:
            label += f' E:{evidence:.2f} U:{uncertainty:.2f}'
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_vis, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    
    return img_vis


def visualize_evidence_map(img, evidence_map, save_path=None, alpha=0.5):
    """
    Visualize evidence map as heatmap overlay
    
    Args:
        img: Image array (H, W, 3) in RGB or tensor (3, H, W)
        evidence_map: Evidence map tensor (1, H, W) or (H, W)
        save_path: Path to save visualization
        alpha: Overlay transparency
    
    Returns:
        img_vis: Visualized image with evidence heatmap
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(img):
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        img = (img.cpu().numpy() * 255).astype(np.uint8)
    
    if torch.is_tensor(evidence_map):
        evidence_map = evidence_map.squeeze().cpu().numpy()
    
    # Resize evidence map to match image size
    h, w = img.shape[:2]
    evidence_map = cv2.resize(evidence_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to 0-255
    evidence_map = (evidence_map * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(evidence_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    img_vis = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    
    return img_vis


def plot_training_curves(train_losses, val_losses=None, save_path=None):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
