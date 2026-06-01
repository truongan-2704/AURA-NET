"""
Test Time Augmentation (TTA) for improved inference accuracy
"""

import torch
import torch.nn.functional as F
import numpy as np


class TTAWrapper:
    """
    Test Time Augmentation wrapper for AURA-Net

    Applies multiple augmentations during inference and averages predictions

    Augmentations:
    - Original
    - Horizontal flip
    - Vertical flip
    - Horizontal + Vertical flip
    - Multi-scale (optional)
    """

    def __init__(self, model, conf_thres=0.25, iou_thres=0.45, scales=None):
        """
        Args:
            model: AURA-Net model
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            scales: List of scales for multi-scale TTA (e.g., [0.8, 1.0, 1.2])
        """
        self.model = model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.scales = scales or [1.0]

    def __call__(self, x):
        """
        Forward pass with TTA

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            detections: List of detection tensors with TTA
        """
        return self.forward_tta(x)

    def forward_tta(self, x):
        """
        Forward pass with TTA

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            detections: List of detection tensors
        """
        all_detections = []

        # Original
        detections = self._forward_single(x)
        all_detections.append(detections)

        # Horizontal flip
        x_hflip = torch.flip(x, dims=[3])
        detections_hflip = self._forward_single(x_hflip)
        detections_hflip = self._unflip_horizontal(detections_hflip)
        all_detections.append(detections_hflip)

        # Vertical flip
        x_vflip = torch.flip(x, dims=[2])
        detections_vflip = self._forward_single(x_vflip)
        detections_vflip = self._unflip_vertical(detections_vflip)
        all_detections.append(detections_vflip)

        # Horizontal + Vertical flip
        x_hvflip = torch.flip(x, dims=[2, 3])
        detections_hvflip = self._forward_single(x_hvflip)
        detections_hvflip = self._unflip_horizontal(detections_hvflip)
        detections_hvflip = self._unflip_vertical(detections_hvflip)
        all_detections.append(detections_hvflip)

        # Multi-scale (if enabled)
        if len(self.scales) > 1:
            for scale in self.scales:
                if scale == 1.0:
                    continue

                # Resize
                h, w = x.shape[2:]
                new_h, new_w = int(h * scale), int(w * scale)
                x_scaled = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)

                # Forward
                detections_scaled = self._forward_single(x_scaled)
                all_detections.append(detections_scaled)

        # Merge all detections
        merged_detections = self._merge_detections(all_detections)

        return merged_detections

    def _forward_single(self, x):
        """
        Single forward pass

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            detections: List of detection tensors
        """
        with torch.no_grad():
            predictions = self.model(x)
            detections = self.model.decoder.decode_predictions(predictions, self.conf_thres)

        return detections

    def _unflip_horizontal(self, detections):
        """
        Unflip horizontally flipped detections

        Args:
            detections: List of detection tensors

        Returns:
            detections: Unflipped detections
        """
        unflipped = []
        for det in detections:
            if len(det) > 0:
                det = det.clone()
                # Flip x coordinates
                det[:, 0] = 1.0 - det[:, 0]  # x1
                det[:, 2] = 1.0 - det[:, 2]  # x2
                # Swap x1 and x2
                det[:, [0, 2]] = det[:, [2, 0]]
            unflipped.append(det)
        return unflipped

    def _unflip_vertical(self, detections):
        """
        Unflip vertically flipped detections

        Args:
            detections: List of detection tensors

        Returns:
            detections: Unflipped detections
        """
        unflipped = []
        for det in detections:
            if len(det) > 0:
                det = det.clone()
                # Flip y coordinates
                det[:, 1] = 1.0 - det[:, 1]  # y1
                det[:, 3] = 1.0 - det[:, 3]  # y2
                # Swap y1 and y2
                det[:, [1, 3]] = det[:, [3, 1]]
            unflipped.append(det)
        return unflipped

    def _merge_detections(self, all_detections):
        """
        Merge detections from multiple augmentations using NMS

        Args:
            all_detections: List of detection lists

        Returns:
            merged: Merged detections
        """
        from utils.boxes import non_max_suppression

        batch_size = len(all_detections[0])
        merged = []

        for i in range(batch_size):
            # Collect all detections for this image
            img_dets = []
            for dets in all_detections:
                if len(dets[i]) > 0:
                    img_dets.append(dets[i])

            if len(img_dets) == 0:
                merged.append(torch.zeros((0, 9), device=all_detections[0][i].device))
                continue

            # Concatenate
            img_dets = torch.cat(img_dets, dim=0)

            # Apply NMS
            img_dets = non_max_suppression([img_dets], self.iou_thres)[0]

            merged.append(img_dets)

        return merged


def apply_tta(model, x, conf_thres=0.25, iou_thres=0.45, scales=None):
    """
    Apply TTA to model inference

    Args:
        model: AURA-Net model
        x: Input tensor (B, 3, H, W)
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        scales: List of scales for multi-scale TTA

    Returns:
        detections: List of detection tensors with TTA
    """
    tta_wrapper = TTAWrapper(model, conf_thres, iou_thres, scales)
    return tta_wrapper(x)
