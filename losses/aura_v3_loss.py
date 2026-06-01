"""
AURA-NET - Loss Functions
==========================

Novel loss design for evidence-centric detection.

Components:
1. Evidence Loss - supervises evidence map generation
2. Box Loss - CIoU loss for box regression
3. Class Loss - Focal loss for classification
4. Uncertainty Loss - supervises uncertainty estimation
5. Iterative Refinement Loss - supervises all decoder iterations

Author: AURA-NET Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple


class EvidenceLoss(nn.Module):
    """
    Evidence Loss

    Supervises evidence map to highlight object centers.
    Uses Gaussian heatmap as target.
    """
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_evidence, target_boxes, image_size=(640, 640)):
        """
        Args:
            pred_evidence: (B, 1, H, W) - predicted evidence map
            target_boxes: List of (N, 5) tensors - [cx, cy, w, h, class_id] normalized
            image_size: (H, W) - original image size

        Returns:
            loss: scalar
        """
        B, _, H, W = pred_evidence.shape
        device = pred_evidence.device

        total_loss = 0.0

        for i in range(B):
            # Generate target evidence map
            target_evidence = self.generate_evidence_target(
                target_boxes[i], H, W, device
            )

            # Focal loss for evidence
            pos_mask = target_evidence == 1
            neg_mask = target_evidence < 1

            pos_loss = -((1 - pred_evidence[i]) ** self.alpha) * torch.log(pred_evidence[i] + 1e-6)
            neg_loss = -((1 - target_evidence) ** self.beta) * (pred_evidence[i] ** self.alpha) * torch.log(1 - pred_evidence[i] + 1e-6)

            loss = pos_loss[pos_mask].sum() + neg_loss[neg_mask].sum()
            total_loss += loss

        return total_loss / B

    def generate_evidence_target(self, boxes, H, W, device):
        """
        Generate Gaussian heatmap for evidence target.

        Args:
            boxes: (N, 5) - [cx, cy, w, h, class_id] normalized
            H, W: evidence map size
            device: torch device

        Returns:
            target: (1, H, W) - evidence target
        """
        target = torch.zeros(1, H, W, device=device)

        if len(boxes) == 0:
            return target

        for box in boxes:
            cx, cy, w, h = box[:4]

            # Convert to evidence map coordinates
            cx_map = cx * W
            cy_map = cy * H
            w_map = w * W
            h_map = h * H

            # Gaussian radius
            radius = max(1, int(min(w_map, h_map) / 6))

            # Generate Gaussian
            diameter = 2 * radius + 1
            gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

            # Get integer coordinates
            x = int(cx_map)
            y = int(cy_map)

            # Compute bounds
            left = min(x, radius)
            right = min(W - x, radius + 1)
            top = min(y, radius)
            bottom = min(H - y, radius + 1)

            # Place Gaussian
            target[0, y - top:y + bottom, x - left:x + right] = torch.maximum(
                target[0, y - top:y + bottom, x - left:x + right],
                torch.from_numpy(gaussian[radius - top:radius + bottom, radius - left:radius + right]).to(device)
            )

        return target

    @staticmethod
    def gaussian2D(shape, sigma=1):
        """Generate 2D Gaussian kernel."""
        import numpy as np
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h


class CIoULoss(nn.Module):
    """
    Complete IoU Loss

    Better than IoU, GIoU, DIoU for box regression.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: (N, 4) - [cx, cy, w, h] normalized
            target_boxes: (N, 4) - [cx, cy, w, h] normalized

        Returns:
            loss: scalar
        """
        # Convert to xyxy
        pred_xyxy = self.cxcywh_to_xyxy(pred_boxes)
        target_xyxy = self.cxcywh_to_xyxy(target_boxes)

        # Intersection
        x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
        y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
        x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
        y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Union
        pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
        target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
        union = pred_area + target_area - intersection

        # IoU
        iou = intersection / (union + 1e-6)

        # Enclosing box
        c_x1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
        c_y1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
        c_x2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
        c_y2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])

        c_diag = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2

        # Center distance
        center_dist = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

        # Aspect ratio
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + 1e-6)) -
            torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + 1e-6)), 2
        )

        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-6)

        # CIoU
        ciou = iou - (center_dist / (c_diag + 1e-6)) - alpha * v

        loss = 1 - ciou

        return loss.mean()

    @staticmethod
    def cxcywh_to_xyxy(boxes):
        """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)


class FocalLoss(nn.Module):
    """
    Focal Loss for classification

    Addresses class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target_labels):
        """
        Args:
            pred_logits: (N, num_classes) - class logits
            target_labels: (N,) - class indices

        Returns:
            loss: scalar
        """
        pred_probs = torch.softmax(pred_logits, dim=-1)
        target_probs = torch.zeros_like(pred_probs)
        target_probs.scatter_(1, target_labels.unsqueeze(1), 1)

        ce_loss = -torch.log(pred_probs + 1e-6) * target_probs
        focal_weight = (1 - pred_probs) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        return focal_loss.sum(dim=-1).mean()


class UncertaintyLoss(nn.Module):
    """
    Uncertainty Loss

    Supervises uncertainty estimation.
    High uncertainty for incorrect predictions, low for correct.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_uncertainty, pred_boxes, target_boxes, pred_classes, target_classes):
        """
        Args:
            pred_uncertainty: (N, 2) - [box_unc, class_unc]
            pred_boxes: (N, 4)
            target_boxes: (N, 4)
            pred_classes: (N, num_classes)
            target_classes: (N,)

        Returns:
            loss: scalar
        """
        # Box uncertainty should correlate with box error
        box_iou = self.compute_iou(pred_boxes, target_boxes)
        box_error = 1 - box_iou
        box_unc_loss = F.mse_loss(pred_uncertainty[:, 0], box_error)

        # Class uncertainty should correlate with class error
        pred_class_probs = torch.softmax(pred_classes, dim=-1)
        correct_class_probs = pred_class_probs[torch.arange(len(target_classes)), target_classes]
        class_error = 1 - correct_class_probs
        class_unc_loss = F.mse_loss(pred_uncertainty[:, 1], class_error)

        return box_unc_loss + class_unc_loss

    @staticmethod
    def compute_iou(boxes1, boxes2):
        """Compute IoU between boxes."""
        # Convert to xyxy
        boxes1_xyxy = torch.zeros_like(boxes1)
        boxes1_xyxy[:, 0] = boxes1[:, 0] - boxes1[:, 2] / 2
        boxes1_xyxy[:, 1] = boxes1[:, 1] - boxes1[:, 3] / 2
        boxes1_xyxy[:, 2] = boxes1[:, 0] + boxes1[:, 2] / 2
        boxes1_xyxy[:, 3] = boxes1[:, 1] + boxes1[:, 3] / 2

        boxes2_xyxy = torch.zeros_like(boxes2)
        boxes2_xyxy[:, 0] = boxes2[:, 0] - boxes2[:, 2] / 2
        boxes2_xyxy[:, 1] = boxes2[:, 1] - boxes2[:, 3] / 2
        boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2] / 2
        boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3] / 2

        # Intersection
        x1 = torch.max(boxes1_xyxy[:, 0], boxes2_xyxy[:, 0])
        y1 = torch.max(boxes1_xyxy[:, 1], boxes2_xyxy[:, 1])
        x2 = torch.min(boxes1_xyxy[:, 2], boxes2_xyxy[:, 2])
        y2 = torch.min(boxes1_xyxy[:, 3], boxes2_xyxy[:, 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Union
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
        union = area1 + area2 - intersection

        iou = intersection / (union + 1e-6)
        return iou


class AURANetLoss(nn.Module):
    """
    Complete Loss for AURA-NET

    Combines:
    1. Evidence Loss
    2. Box Loss (CIoU)
    3. Class Loss (Focal)
    4. Uncertainty Loss
    5. Iterative Refinement Loss

    Args:
        num_classes: Number of classes
        lambda_evidence: Weight for evidence loss (default: 1.0)
        lambda_box: Weight for box loss (default: 5.0)
        lambda_class: Weight for class loss (default: 1.0)
        lambda_uncertainty: Weight for uncertainty loss (default: 0.5)
        lambda_iterative: Weight for iterative loss (default: 0.3)
    """
    def __init__(
        self,
        num_classes=80,
        lambda_evidence=1.0,
        lambda_box=5.0,
        lambda_class=1.0,
        lambda_uncertainty=0.5,
        lambda_iterative=0.3
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lambda_evidence = lambda_evidence
        self.lambda_box = lambda_box
        self.lambda_class = lambda_class
        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_iterative = lambda_iterative

        self.evidence_loss = EvidenceLoss()
        self.box_loss = CIoULoss()
        self.class_loss = FocalLoss()
        self.uncertainty_loss = UncertaintyLoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict from model forward pass
            targets: List of (N, 5) tensors - [cx, cy, w, h, class_id] normalized

        Returns:
            loss_dict: dict with all loss components
        """
        # Evidence loss
        loss_evidence = self.evidence_loss(
            predictions['evidence_map'],
            targets
        )

        # Match predictions to targets
        matched_pred_boxes, matched_target_boxes, matched_pred_classes, matched_target_classes = self.match_predictions(
            predictions, targets
        )

        if len(matched_pred_boxes) == 0:
            # No matches, return only evidence loss
            return {
                'loss': loss_evidence * self.lambda_evidence,
                'loss_evidence': loss_evidence,
                'loss_box': torch.tensor(0.0, device=loss_evidence.device),
                'loss_class': torch.tensor(0.0, device=loss_evidence.device),
                'loss_uncertainty': torch.tensor(0.0, device=loss_evidence.device),
                'loss_iterative': torch.tensor(0.0, device=loss_evidence.device),
            }

        # Box loss
        loss_box = self.box_loss(matched_pred_boxes, matched_target_boxes)

        # Class loss
        loss_class = self.class_loss(matched_pred_classes, matched_target_classes)

        # Uncertainty loss
        loss_uncertainty = self.uncertainty_loss(
            predictions['uncertainty'][matched_pred_boxes.size(0)],
            matched_pred_boxes,
            matched_target_boxes,
            matched_pred_classes,
            matched_target_classes
        )

        # Iterative refinement loss (if intermediate predictions available)
        loss_iterative = torch.tensor(0.0, device=loss_evidence.device)
        if 'intermediate' in predictions:
            intermediate_boxes = predictions['intermediate']['decoder_intermediate']['boxes']
            for iter_boxes in intermediate_boxes[:-1]:  # Exclude final iteration
                iter_matched_boxes = iter_boxes.view(-1, 4)[matched_pred_boxes.size(0)]
                loss_iterative += self.box_loss(iter_matched_boxes, matched_target_boxes)
            loss_iterative /= len(intermediate_boxes) - 1

        # Total loss
        total_loss = (
            self.lambda_evidence * loss_evidence +
            self.lambda_box * loss_box +
            self.lambda_class * loss_class +
            self.lambda_uncertainty * loss_uncertainty +
            self.lambda_iterative * loss_iterative
        )

        return {
            'loss': total_loss,
            'loss_evidence': loss_evidence,
            'loss_box': loss_box,
            'loss_class': loss_class,
            'loss_uncertainty': loss_uncertainty,
            'loss_iterative': loss_iterative,
        }

    def match_predictions(self, predictions, targets):
        """
        Match predictions to ground truth targets.

        Simple matching based on IoU (can be improved with Hungarian matching).

        Args:
            predictions: dict from model
            targets: List of (N, 5) tensors

        Returns:
            matched_pred_boxes: (M, 4)
            matched_target_boxes: (M, 4)
            matched_pred_classes: (M, num_classes)
            matched_target_classes: (M,)
        """
        pred_boxes = predictions['boxes']  # (B, K, 4)
        pred_classes = predictions['classes']  # (B, K, num_classes)
        B, K, _ = pred_boxes.shape

        all_matched_pred_boxes = []
        all_matched_target_boxes = []
        all_matched_pred_classes = []
        all_matched_target_classes = []

        for i in range(B):
            if len(targets[i]) == 0:
                continue

            target_boxes = targets[i][:, :4]  # (N, 4)
            target_classes = targets[i][:, 4].long()  # (N,)

            # Compute IoU matrix
            iou_matrix = self.compute_iou_matrix(pred_boxes[i], target_boxes)  # (K, N)

            # Match: for each target, find best prediction
            max_ious, matched_pred_indices = iou_matrix.max(dim=0)  # (N,)

            # Filter by IoU threshold
            valid_mask = max_ious > 0.5

            if valid_mask.sum() == 0:
                continue

            matched_pred_idx = matched_pred_indices[valid_mask]
            matched_target_idx = torch.where(valid_mask)[0]

            all_matched_pred_boxes.append(pred_boxes[i][matched_pred_idx])
            all_matched_target_boxes.append(target_boxes[matched_target_idx])
            all_matched_pred_classes.append(pred_classes[i][matched_pred_idx])
            all_matched_target_classes.append(target_classes[matched_target_idx])

        if len(all_matched_pred_boxes) == 0:
            device = pred_boxes.device
            return (
                torch.empty(0, 4, device=device),
                torch.empty(0, 4, device=device),
                torch.empty(0, self.num_classes, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            )

        return (
            torch.cat(all_matched_pred_boxes, dim=0),
            torch.cat(all_matched_target_boxes, dim=0),
            torch.cat(all_matched_pred_classes, dim=0),
            torch.cat(all_matched_target_classes, dim=0)
        )

    @staticmethod
    def compute_iou_matrix(boxes1, boxes2):
        """
        Compute IoU matrix between two sets of boxes.

        Args:
            boxes1: (N, 4) - [cx, cy, w, h]
            boxes2: (M, 4) - [cx, cy, w, h]

        Returns:
            iou: (N, M)
        """
        # Convert to xyxy
        boxes1_xyxy = torch.zeros_like(boxes1)
        boxes1_xyxy[:, 0] = boxes1[:, 0] - boxes1[:, 2] / 2
        boxes1_xyxy[:, 1] = boxes1[:, 1] - boxes1[:, 3] / 2
        boxes1_xyxy[:, 2] = boxes1[:, 0] + boxes1[:, 2] / 2
        boxes1_xyxy[:, 3] = boxes1[:, 1] + boxes1[:, 3] / 2

        boxes2_xyxy = torch.zeros_like(boxes2)
        boxes2_xyxy[:, 0] = boxes2[:, 0] - boxes2[:, 2] / 2
        boxes2_xyxy[:, 1] = boxes2[:, 1] - boxes2[:, 3] / 2
        boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2] / 2
        boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3] / 2

        # Intersection
        x1 = torch.max(boxes1_xyxy[:, None, 0], boxes2_xyxy[None, :, 0])
        y1 = torch.max(boxes1_xyxy[:, None, 1], boxes2_xyxy[None, :, 1])
        x2 = torch.min(boxes1_xyxy[:, None, 2], boxes2_xyxy[None, :, 2])
        y2 = torch.min(boxes1_xyxy[:, None, 3], boxes2_xyxy[None, :, 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Union
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
        union = area1[:, None] + area2[None, :] - intersection

        iou = intersection / (union + 1e-6)
        return iou


if __name__ == '__main__':
    # Test
    loss_fn = AURANetLoss(num_classes=80)

    # Dummy predictions
    predictions = {
        'boxes': torch.rand(2, 200, 4),
        'classes': torch.randn(2, 200, 80),
        'evidence_map': torch.rand(2, 1, 80, 80),
        'uncertainty': torch.rand(2, 200, 2),
    }

    # Dummy targets
    targets = [
        torch.tensor([[0.5, 0.5, 0.2, 0.3, 0], [0.3, 0.7, 0.1, 0.2, 1]]),
        torch.tensor([[0.6, 0.4, 0.15, 0.25, 2]])
    ]

    loss_dict = loss_fn(predictions, targets)

    print("Loss Components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
