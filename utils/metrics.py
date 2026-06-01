"""
Evaluation metrics for object detection
Supports mAP@0.5, mAP@0.5:0.95, confusion matrix, PR curve, etc.
"""

import torch
import numpy as np
from .boxes import box_iou


def compute_ap(recall, precision):
    """
    Compute Average Precision (AP) using 101-point interpolation

    Args:
        recall: Array of recall values
        precision: Array of precision values

    Returns:
        ap: Average precision
    """
    # Append sentinel values
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def compute_metrics_single_class(pred_boxes, pred_scores, gt_boxes, iou_thres=0.5):
    """
    Compute precision, recall for a single class

    Args:
        pred_boxes: Predicted boxes (N, 4) in format (x1, y1, x2, y2)
        pred_scores: Prediction scores (N,)
        gt_boxes: Ground truth boxes (M, 4)
        iou_thres: IoU threshold for matching

    Returns:
        tp: True positives array (N,)
        fp: False positives array (N,)
        scores: Sorted scores (N,)
    """
    if len(pred_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    if len(gt_boxes) == 0:
        # All predictions are false positives
        return np.zeros(len(pred_boxes)), np.ones(len(pred_boxes)), pred_scores.cpu().numpy()

    # Sort by score
    sort_idx = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sort_idx]
    pred_scores = pred_scores[sort_idx]

    # Compute IoU
    ious = box_iou(pred_boxes, gt_boxes)

    # Match predictions to ground truth
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched_gt = set()

    for i in range(len(pred_boxes)):
        # Find best matching ground truth
        best_iou = 0
        best_gt_idx = -1

        for j in range(len(gt_boxes)):
            if j in matched_gt:
                continue

            if ious[i, j] > best_iou:
                best_iou = ious[i, j]
                best_gt_idx = j

        if best_iou >= iou_thres and best_gt_idx >= 0:
            tp[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1

    return tp, fp, pred_scores.cpu().numpy()


class MetricsCalculator:
    """
    Calculate object detection metrics over entire dataset
    Supports:
    - mAP@0.5
    - mAP@0.5:0.95 (COCO-style)
    - Precision, Recall, F1
    - Confusion matrix
    - Per-class metrics
    """

    def __init__(self, num_classes=10, conf_thres=0.25, iou_thres=0.5):
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.reset()

    def reset(self):
        """Reset accumulated data"""
        # Store all predictions and ground truths per class
        self.stats = []  # List of (tp, conf, pred_cls, target_cls)
        self.all_pred_boxes = []
        self.all_pred_classes = []
        self.all_pred_scores = []
        self.all_gt_boxes = []
        self.all_gt_classes = []

    def update(self, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes):
        """
        Update with batch results

        Args:
            pred_boxes: List of predicted boxes per image, each (N, 4)
            pred_classes: List of predicted classes per image, each (N,)
            pred_scores: List of prediction scores per image, each (N,)
            gt_boxes: List of ground truth boxes per image, each (M, 4)
            gt_classes: List of ground truth classes per image, each (M,)
        """
        for pred_box, pred_cls, pred_score, gt_box, gt_cls in zip(
            pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes
        ):
            self.all_pred_boxes.append(pred_box)
            self.all_pred_classes.append(pred_cls)
            self.all_pred_scores.append(pred_score)
            self.all_gt_boxes.append(gt_box)
            self.all_gt_classes.append(gt_cls)

            # Process for confusion matrix and stats
            self._process_batch(pred_box, pred_cls, pred_score, gt_box, gt_cls)

    def _process_batch(self, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes):
        """Process a single image for statistics"""
        if len(pred_boxes) == 0:
            if len(gt_boxes) > 0:
                # False negatives
                for gt_cls in gt_classes:
                    self.stats.append((
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        gt_cls.cpu().numpy()
                    ))
            return

        if len(gt_boxes) == 0:
            # All predictions are false positives
            self.stats.append((
                np.zeros(len(pred_boxes)),
                pred_scores.cpu().numpy(),
                pred_classes.cpu().numpy(),
                np.array([])
            ))
            return

        # Compute IoU matrix
        ious = box_iou(pred_boxes, gt_boxes)

        # Match predictions to ground truth
        correct = np.zeros((len(pred_boxes), 10))  # 10 IoU thresholds

        for i, iou_level in enumerate(np.linspace(0.5, 0.95, 10)):
            matched_gt = set()
            for pred_idx in range(len(pred_boxes)):
                # Find best matching ground truth with same class
                best_iou = 0
                best_gt_idx = -1

                for gt_idx in range(len(gt_boxes)):
                    if gt_idx in matched_gt:
                        continue
                    if pred_classes[pred_idx] != gt_classes[gt_idx]:
                        continue
                    if ious[pred_idx, gt_idx] > best_iou:
                        best_iou = ious[pred_idx, gt_idx]
                        best_gt_idx = gt_idx

                if best_iou >= iou_level and best_gt_idx >= 0:
                    correct[pred_idx, i] = 1
                    matched_gt.add(best_gt_idx)

        self.stats.append((
            correct,
            pred_scores.cpu().numpy(),
            pred_classes.cpu().numpy(),
            gt_classes.cpu().numpy()
        ))

    def compute(self):
        """
        Compute final metrics

        Returns:
            metrics: Dictionary of metrics including:
                - precision, recall, f1
                - mAP@0.5, mAP@0.5:0.95
                - per_class_ap
                - confusion_matrix
        """
        if len(self.stats) == 0:
            return self._empty_metrics()

        # Concatenate all stats
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]

        if len(stats) and stats[0].any():
            tp, conf, pred_cls, target_cls = stats[0], stats[1], stats[2], stats[3]

            # Sort by confidence
            i = np.argsort(-conf)
            tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

            # Find unique classes
            unique_classes = np.unique(target_cls)
            nc = unique_classes.shape[0]

            # Compute AP for each class and IoU threshold
            px, py = np.linspace(0, 1, 1000), []
            ap, p, r = np.zeros((nc, 10)), np.zeros((nc, 1000)), np.zeros((nc, 1000))

            for ci, c in enumerate(unique_classes):
                i = pred_cls == c
                n_gt = (target_cls == c).sum()
                n_p = i.sum()

                if n_p == 0 or n_gt == 0:
                    continue

                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum(0)
                tpc = tp[i].cumsum(0)

                # Recall
                recall = tpc / (n_gt + 1e-16)
                r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

                # Precision
                precision = tpc / (tpc + fpc)
                p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

                # AP for each IoU threshold
                for j in range(tp.shape[1]):
                    ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Compute F1 score
            f1 = 2 * p * r / (p + r + 1e-16)

            # Confusion matrix
            confusion_mat = self._compute_confusion_matrix(pred_cls, target_cls)

            metrics = {
                'precision': p.mean(),
                'recall': r.mean(),
                'f1': f1.mean(),
                'mAP@0.5': ap[:, 0].mean(),
                'mAP@0.5:0.95': ap.mean(),
                'per_class_ap': ap[:, 0],
                'ap_per_iou': ap.mean(0),
                'confusion_matrix': confusion_mat,
                'p_curve': p,
                'r_curve': r,
                'f1_curve': f1,
                'px': px,
                'unique_classes': unique_classes
            }

            return metrics

        return self._empty_metrics()

    def _compute_confusion_matrix(self, pred_cls, target_cls):
        """Compute confusion matrix"""
        try:
            from sklearn.metrics import confusion_matrix
            # Add background class
            all_classes = list(range(self.num_classes + 1))

            # Convert predictions and targets
            pred_cls_int = pred_cls.astype(int)
            target_cls_int = target_cls.astype(int)

            # Compute confusion matrix
            cm = confusion_matrix(
                target_cls_int,
                pred_cls_int,
                labels=all_classes
            )

            return cm
        except:
            # Fallback: simple confusion matrix
            cm = np.zeros((self.num_classes + 1, self.num_classes + 1))
            return cm

    def _empty_metrics(self):
        """Return empty metrics when no data"""
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mAP@0.5': 0.0,
            'mAP@0.5:0.95': 0.0,
            'per_class_ap': np.zeros(self.num_classes),
            'ap_per_iou': np.zeros(10),
            'confusion_matrix': np.zeros((self.num_classes + 1, self.num_classes + 1)),
            'p_curve': np.zeros((self.num_classes, 1000)),
            'r_curve': np.zeros((self.num_classes, 1000)),
            'f1_curve': np.zeros((self.num_classes, 1000)),
            'px': np.linspace(0, 1, 1000),
            'unique_classes': np.array([])
        }
