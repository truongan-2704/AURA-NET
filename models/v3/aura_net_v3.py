"""
AURA-NET - Main Model
======================

Evidence-Centric Object Detection with Adaptive Computation

Complete integration of all novel components:
1. Dual-Stream Feature Extractor (DSFE)
2. Evidence Proposal Network (EPN)
3. Adaptive Evidence Router (AER)
4. Evidence-to-Object Decoder (EOD)

Author: AURA-NET Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .dual_stream_extractor import DualStreamFeatureExtractor
from .evidence_proposal_network import EvidenceProposalNetwork
from .adaptive_evidence_router import AdaptiveEvidenceRouter
from .evidence_to_object_decoder import EvidenceToObjectDecoder


class AURANet(nn.Module):
    """
    AURA-NET

    Evidence-Centric Object Detection with Adaptive Computation

    Novel Architecture:
        Input (B, 3, 640, 640)
        ↓
        Dual-Stream Feature Extractor → (B, 128, 80, 80)
        ↓
        Evidence Proposal Network → evidence_map (B, 1, 80, 80), features (B, 256, 80, 80)
        ↓
        Adaptive Evidence Router → routed_features (B, K, 256), coords, scores
        ↓
        Evidence-to-Object Decoder → boxes, classes, uncertainty
        ↓
        Evidence-Guided NMS → Final detections

    Args:
        num_classes: Number of object classes (default: 80 for COCO)
        feature_dim: Feature dimension (default: 256)
        min_K: Minimum evidence regions (default: 100)
        max_K: Maximum evidence regions (default: 300)
        num_iterations: Decoder iterations (default: 3)
        conf_threshold: Confidence threshold for inference (default: 0.25)
        iou_threshold: IoU threshold for NMS (default: 0.5)
    """
    def __init__(
        self,
        num_classes: int = 80,
        feature_dim: int = 256,
        min_K: int = 100,
        max_K: int = 300,
        num_iterations: int = 3,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Stage 1: Dual-Stream Feature Extraction
        self.feature_extractor = DualStreamFeatureExtractor(
            in_channels=3,
            out_channels=128
        )

        # Stage 2: Evidence Proposal Network
        self.evidence_network = EvidenceProposalNetwork(
            in_channels=128,
            hidden_channels=feature_dim
        )

        # Stage 3: Adaptive Evidence Router
        self.evidence_router = AdaptiveEvidenceRouter(
            feature_dim=feature_dim,
            min_K=min_K,
            max_K=max_K
        )

        # Stage 4: Evidence-to-Object Decoder
        self.object_decoder = EvidenceToObjectDecoder(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_iterations=num_iterations
        )

    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> Dict:
        """
        Forward pass

        Args:
            x: (B, 3, H, W) - input images
            return_intermediate: Whether to return intermediate outputs

        Returns:
            output: dict with keys:
                - boxes: (B, K, 4) - predicted boxes (cx, cy, w, h) normalized
                - classes: (B, K, num_classes) - class logits
                - scores: (B, K) - confidence scores
                - evidence_map: (B, 1, 80, 80) - evidence heatmap
                - uncertainty: (B, K, 2) - prediction uncertainty
                - routing_info: dict - routing statistics
                - intermediate: dict (if return_intermediate=True)
        """
        B, C, H, W = x.shape

        # Stage 1: Dual-Stream Feature Extraction
        features = self.feature_extractor(x)  # (B, 128, 80, 80)

        # Stage 2: Evidence Proposal
        evidence_map, enhanced_features = self.evidence_network(features)  # (B, 1, 80, 80), (B, 256, 80, 80)

        # Stage 3: Adaptive Evidence Routing
        routed_features, coords, evidence_scores, routing_info = self.evidence_router(
            enhanced_features, evidence_map
        )  # (B, K, 256), (B, K, 2), (B, K, 1)

        # Stage 4: Evidence-to-Object Decoding
        decoder_output = self.object_decoder(
            routed_features, coords, evidence_scores
        )

        # Extract predictions
        boxes = decoder_output['boxes']  # (B, K, 4)
        class_logits = decoder_output['classes']  # (B, K, num_classes)
        uncertainty = decoder_output['uncertainty']  # (B, K, 2)

        # Compute confidence scores
        class_probs = torch.softmax(class_logits, dim=-1)  # (B, K, num_classes)
        max_class_probs, class_ids = class_probs.max(dim=-1)  # (B, K)

        # Combine class confidence with evidence scores
        confidence_scores = max_class_probs * evidence_scores.squeeze(-1)  # (B, K)

        # Prepare output
        output = {
            'boxes': boxes,
            'classes': class_logits,
            'class_ids': class_ids,
            'scores': confidence_scores,
            'evidence_map': evidence_map,
            'evidence_scores': evidence_scores,
            'uncertainty': uncertainty,
            'routing_info': routing_info,
        }

        if return_intermediate:
            output['intermediate'] = {
                'features': features,
                'enhanced_features': enhanced_features,
                'routed_features': routed_features,
                'coords': coords,
                'decoder_intermediate': decoder_output['intermediate']
            }

        return output

    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Inference with NMS

        Args:
            x: (B, 3, H, W) - input images
            conf_threshold: Confidence threshold (default: self.conf_threshold)
            iou_threshold: IoU threshold for NMS (default: self.iou_threshold)

        Returns:
            predictions: List of dicts (one per image) with keys:
                - boxes: (N, 4) - filtered boxes (cx, cy, w, h) normalized
                - scores: (N,) - confidence scores
                - class_ids: (N,) - class IDs
                - evidence_scores: (N,) - evidence scores
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        # Forward pass
        with torch.no_grad():
            output = self.forward(x)

        boxes = output['boxes']  # (B, K, 4)
        scores = output['scores']  # (B, K)
        class_ids = output['class_ids']  # (B, K)
        evidence_scores = output['evidence_scores'].squeeze(-1)  # (B, K)

        B = boxes.size(0)
        predictions = []

        for i in range(B):
            # Filter by confidence
            mask = scores[i] > conf_threshold
            filtered_boxes = boxes[i][mask]
            filtered_scores = scores[i][mask]
            filtered_class_ids = class_ids[i][mask]
            filtered_evidence = evidence_scores[i][mask]

            if len(filtered_boxes) == 0:
                predictions.append({
                    'boxes': torch.empty(0, 4, device=boxes.device),
                    'scores': torch.empty(0, device=boxes.device),
                    'class_ids': torch.empty(0, dtype=torch.long, device=boxes.device),
                    'evidence_scores': torch.empty(0, device=boxes.device),
                })
                continue

            # Evidence-Guided NMS
            keep_indices = self.evidence_guided_nms(
                filtered_boxes,
                filtered_scores,
                filtered_evidence,
                filtered_class_ids,
                iou_threshold
            )

            predictions.append({
                'boxes': filtered_boxes[keep_indices],
                'scores': filtered_scores[keep_indices],
                'class_ids': filtered_class_ids[keep_indices],
                'evidence_scores': filtered_evidence[keep_indices],
            })

        return predictions

    def evidence_guided_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        evidence_scores: torch.Tensor,
        class_ids: torch.Tensor,
        iou_threshold: float
    ) -> torch.Tensor:
        """
        Evidence-Guided NMS

        Novel: Incorporates evidence scores into NMS decision.
        High-evidence detections are preserved even with lower confidence.

        Args:
            boxes: (N, 4) - boxes (cx, cy, w, h) normalized
            scores: (N,) - confidence scores
            evidence_scores: (N,) - evidence scores
            class_ids: (N,) - class IDs
            iou_threshold: IoU threshold

        Returns:
            keep_indices: (M,) - indices to keep
        """
        # Combine confidence and evidence
        combined_scores = scores * 0.7 + evidence_scores * 0.3

        # Per-class NMS
        keep_indices = []
        unique_classes = class_ids.unique()

        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = combined_scores[cls_mask]
            cls_evidence = evidence_scores[cls_mask]
            cls_indices = torch.where(cls_mask)[0]

            # Sort by combined score
            sorted_idx = torch.argsort(cls_scores, descending=True)

            keep = []
            while len(sorted_idx) > 0:
                # Keep highest scoring box
                idx = sorted_idx[0]
                keep.append(cls_indices[idx].item())

                if len(sorted_idx) == 1:
                    break

                # Compute IoU with remaining boxes
                ious = self.box_iou(
                    cls_boxes[idx:idx+1],
                    cls_boxes[sorted_idx[1:]]
                )[0]

                # Adaptive threshold based on evidence
                adaptive_threshold = iou_threshold * (1 - cls_evidence[idx] * 0.3)

                # Keep boxes with IoU < threshold OR high evidence
                mask = (ious < adaptive_threshold) | (cls_evidence[sorted_idx[1:]] > 0.8)
                sorted_idx = sorted_idx[1:][mask]

            keep_indices.extend(keep)

        return torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)

    @staticmethod
    def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes

        Args:
            boxes1: (N, 4) - boxes (cx, cy, w, h)
            boxes2: (M, 4) - boxes (cx, cy, w, h)

        Returns:
            iou: (N, M) - IoU matrix
        """
        # Convert to (x1, y1, x2, y2)
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

        # Compute intersection
        x1 = torch.max(boxes1_xyxy[:, None, 0], boxes2_xyxy[None, :, 0])
        y1 = torch.max(boxes1_xyxy[:, None, 1], boxes2_xyxy[None, :, 1])
        x2 = torch.min(boxes1_xyxy[:, None, 2], boxes2_xyxy[None, :, 2])
        y2 = torch.min(boxes1_xyxy[:, None, 3], boxes2_xyxy[None, :, 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Compute union
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
        union = area1[:, None] + area2[None, :] - intersection

        # Compute IoU
        iou = intersection / (union + 1e-6)

        return iou


if __name__ == '__main__':
    # Test
    model = AURANet(
        num_classes=80,
        feature_dim=256,
        min_K=100,
        max_K=300,
        num_iterations=3
    )

    x = torch.randn(2, 3, 640, 640)

    print("=" * 60)
    print("AURA-NET - Forward Pass Test")
    print("=" * 60)

    with torch.no_grad():
        # Training mode
        output = model(x, return_intermediate=True)

        print("\nTraining Output:")
        print(f"  Boxes: {output['boxes'].shape}")
        print(f"  Classes: {output['classes'].shape}")
        print(f"  Scores: {output['scores'].shape}")
        print(f"  Evidence map: {output['evidence_map'].shape}")
        print(f"  Uncertainty: {output['uncertainty'].shape}")
        print(f"\nRouting Info:")
        for k, v in output['routing_info'].items():
            print(f"  {k}: {v}")

        # Inference mode
        predictions = model.predict(x, conf_threshold=0.25, iou_threshold=0.5)

        print(f"\nInference Output (batch size {len(predictions)}):")
        for i, pred in enumerate(predictions):
            print(f"  Image {i}: {len(pred['boxes'])} detections")

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {params / 1e6:.2f}M")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
