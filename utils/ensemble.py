"""
Model Ensemble for improved inference accuracy
"""

import torch
from pathlib import Path
from utils.boxes import non_max_suppression


class ModelEnsemble:
    """
    Ensemble multiple models for better predictions

    Methods:
    - Average predictions
    - Weighted average
    - NMS fusion
    """

    def __init__(self, models, weights=None, method='nms', conf_thres=0.25, iou_thres=0.45):
        """
        Args:
            models: List of models
            weights: List of weights for each model (for weighted average)
            method: Ensemble method ('average', 'weighted', 'nms')
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
        """
        self.models = models
        self.method = method
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Set all models to eval mode
        for model in self.models:
            model.eval()

        # Weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def __call__(self, x):
        """
        Forward pass with ensemble

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            detections: List of detection tensors
        """
        return self.forward(x)

    def forward(self, x):
        """
        Forward pass with ensemble

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            detections: List of detection tensors
        """
        if self.method == 'nms':
            return self._forward_nms(x)
        elif self.method == 'average':
            return self._forward_average(x)
        elif self.method == 'weighted':
            return self._forward_weighted(x)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def _forward_nms(self, x):
        """
        Ensemble using NMS fusion

        Collect all predictions from all models and apply NMS

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            detections: List of detection tensors
        """
        all_detections = []

        # Get predictions from all models
        with torch.no_grad():
            for model in self.models:
                predictions = model(x)
                detections = model.decoder.decode_predictions(predictions, self.conf_thres)
                all_detections.append(detections)

        # Merge detections
        batch_size = len(all_detections[0])
        merged = []

        for i in range(batch_size):
            # Collect all detections for this image
            img_dets = []
            for dets in all_detections:
                if len(dets[i]) > 0:
                    img_dets.append(dets[i])

            if len(img_dets) == 0:
                merged.append(torch.zeros((0, 9), device=x.device))
                continue

            # Concatenate
            img_dets = torch.cat(img_dets, dim=0)

            # Apply NMS
            img_dets = non_max_suppression([img_dets], self.iou_thres)[0]

            merged.append(img_dets)

        return merged

    def _forward_average(self, x):
        """
        Ensemble using average predictions

        Average the raw predictions before decoding

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            detections: List of detection tensors
        """
        all_predictions = []

        # Get predictions from all models
        with torch.no_grad():
            for model in self.models:
                predictions = model(x)
                all_predictions.append(predictions)

        # Average predictions
        avg_predictions = {}
        for key in all_predictions[0].keys():
            avg_predictions[key] = torch.stack([p[key] for p in all_predictions]).mean(dim=0)

        # Decode averaged predictions
        detections = self.models[0].decoder.decode_predictions(avg_predictions, self.conf_thres)

        # Apply NMS
        detections = non_max_suppression(detections, self.iou_thres)

        return detections

    def _forward_weighted(self, x):
        """
        Ensemble using weighted average predictions

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            detections: List of detection tensors
        """
        all_predictions = []

        # Get predictions from all models
        with torch.no_grad():
            for model in self.models:
                predictions = model(x)
                all_predictions.append(predictions)

        # Weighted average predictions
        avg_predictions = {}
        for key in all_predictions[0].keys():
            weighted_sum = sum(w * p[key] for w, p in zip(self.weights, all_predictions))
            avg_predictions[key] = weighted_sum

        # Decode averaged predictions
        detections = self.models[0].decoder.decode_predictions(avg_predictions, self.conf_thres)

        # Apply NMS
        detections = non_max_suppression(detections, self.iou_thres)

        return detections


def load_ensemble(weight_paths, config, device='cuda'):
    """
    Load multiple models for ensemble

    Args:
        weight_paths: List of paths to model weights
        config: Model configuration
        device: Device to load models

    Returns:
        ensemble: ModelEnsemble instance
    """
    from models.aura_net import build_aura_net
    from utils.checkpoint import load_model_weights

    models = []

    for weight_path in weight_paths:
        print(f'Loading model from {weight_path}')
        model = build_aura_net(config)
        model = load_model_weights(model, weight_path, device)
        model = model.to(device)
        model.eval()
        models.append(model)

    ensemble = ModelEnsemble(models)

    return ensemble
