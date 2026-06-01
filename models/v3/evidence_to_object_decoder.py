"""
AURA-NET v3.0 - Evidence-to-Object Decoder (EOD)
=================================================

Novel Component #4: Iteratively refines detections from evidence.

Key innovations:
1. Iterative refinement (2-3 iterations)
2. Each iteration refines box + class + evidence
3. Uncertainty estimation per iteration
4. No anchor, no grid - pure evidence-based

Author: AURA-NET Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RefinementLayer(nn.Module):
    """
    Single Refinement Layer

    Refines box and class predictions given current estimates.
    """
    def __init__(self, feature_dim=256, num_classes=80):
        super().__init__()

        self.num_classes = num_classes

        # Self-attention for feature enhancement
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Box refinement head
        self.box_head = nn.Sequential(
            nn.Linear(feature_dim + 4, 256),  # +4 for current box
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 4)  # Delta box (dx, dy, dw, dh)
        )

        # Class head
        self.class_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, features, boxes, classes):
        """
        Args:
            features: (B, K, feature_dim)
            boxes: (B, K, 4) - current box estimates (cx, cy, w, h) normalized
            classes: (B, K, num_classes) - current class logits

        Returns:
            delta_boxes: (B, K, 4) - box refinements
            class_logits: (B, K, num_classes) - refined class predictions
        """
        # Self-attention
        attn_out, _ = self.self_attn(features, features, features)
        features = self.norm1(features + attn_out)

        # FFN
        ffn_out = self.ffn(features)
        features = self.norm2(features + ffn_out)

        # Box refinement (conditioned on current box)
        box_input = torch.cat([features, boxes], dim=-1)
        delta_boxes = self.box_head(box_input)
        delta_boxes = torch.tanh(delta_boxes) * 0.5  # Limit refinement range

        # Class prediction
        class_logits = self.class_head(features)

        return delta_boxes, class_logits


class UncertaintyEstimator(nn.Module):
    """
    Uncertainty Estimator

    Estimates prediction uncertainty for each detection.
    """
    def __init__(self, feature_dim=256):
        super().__init__()

        self.uncertainty_net = nn.Sequential(
            nn.Linear(feature_dim + 4 + 1, 128),  # +4 box, +1 max class score
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 2),  # Box uncertainty, class uncertainty
            nn.Sigmoid()
        )

    def forward(self, features, boxes, classes):
        """
        Args:
            features: (B, K, feature_dim)
            boxes: (B, K, 4)
            classes: (B, K, num_classes)

        Returns:
            uncertainty: (B, K, 2) - [box_uncertainty, class_uncertainty]
        """
        # Get max class score
        max_class_score = torch.softmax(classes, dim=-1).max(dim=-1, keepdim=True)[0]

        # Concatenate
        combined = torch.cat([features, boxes, max_class_score], dim=-1)

        # Estimate uncertainty
        uncertainty = self.uncertainty_net(combined)

        return uncertainty


class EvidenceToObjectDecoder(nn.Module):
    """
    Evidence-to-Object Decoder (EOD)

    Novel: Iteratively refines detections from evidence regions.
    Each iteration improves box, class, and evidence score.

    Architecture:
        Input: region_features (B, K, 256), coords (B, K, 2), evidence_scores (B, K, 1)
        ↓
        Initialize boxes from coords
        ↓
        Iteration 1: Refine boxes + classes
        ↓
        Iteration 2: Refine boxes + classes
        ↓
        Iteration 3: Refine boxes + classes
        ↓
        Uncertainty estimation
        ↓
        Output: boxes, classes, uncertainty, evidence_scores

    Args:
        feature_dim: Feature dimension (default: 256)
        num_classes: Number of classes (default: 80)
        num_iterations: Number of refinement iterations (default: 3)
    """
    def __init__(self, feature_dim=256, num_classes=80, num_iterations=3):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_iterations = num_iterations

        # Iterative refinement modules
        self.refinement_layers = nn.ModuleList([
            RefinementLayer(feature_dim, num_classes)
            for _ in range(num_iterations)
        ])

        # Uncertainty estimator
        self.uncertainty_net = UncertaintyEstimator(feature_dim)

        # Initial box predictor
        self.init_box_net = nn.Sequential(
            nn.Linear(feature_dim + 2 + 1, 128),  # +2 coords, +1 evidence
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 4),  # (cx, cy, w, h)
            nn.Sigmoid()  # Normalized [0, 1]
        )

    def init_boxes(self, region_features, coords, evidence_scores):
        """
        Initialize boxes from evidence coordinates.

        Args:
            region_features: (B, K, feature_dim)
            coords: (B, K, 2) - normalized coordinates
            evidence_scores: (B, K, 1)

        Returns:
            boxes: (B, K, 4) - initial box estimates (cx, cy, w, h)
        """
        # Concatenate features, coords, and evidence
        combined = torch.cat([region_features, coords, evidence_scores], dim=-1)

        # Predict initial boxes
        boxes = self.init_box_net(combined)

        return boxes

    def forward(self, region_features, coords, evidence_scores):
        """
        Args:
            region_features: (B, K, feature_dim)
            coords: (B, K, 2) - normalized coordinates [0, 1]
            evidence_scores: (B, K, 1)

        Returns:
            output: dict with keys:
                - boxes: (B, K, 4) - final box predictions (cx, cy, w, h) normalized
                - classes: (B, K, num_classes) - final class logits
                - uncertainty: (B, K, 2) - [box_unc, class_unc]
                - evidence_scores: (B, K, 1) - evidence scores
                - intermediate: dict with all intermediate predictions
        """
        B, K, _ = region_features.shape

        # Initialize predictions from evidence
        boxes = self.init_boxes(region_features, coords, evidence_scores)
        classes = torch.zeros(B, K, self.num_classes, device=region_features.device)

        # Store intermediate predictions
        all_boxes = [boxes]
        all_classes = [classes]

        # Iterative refinement
        for i, refiner in enumerate(self.refinement_layers):
            # Refine predictions
            delta_boxes, class_logits = refiner(region_features, boxes, classes)

            # Update
            boxes = boxes + delta_boxes
            boxes = torch.clamp(boxes, 0, 1)  # Keep normalized
            classes = class_logits

            all_boxes.append(boxes)
            all_classes.append(classes)

        # Estimate uncertainty
        uncertainty = self.uncertainty_net(region_features, boxes, classes)

        # Return final iteration + all intermediate
        output = {
            'boxes': boxes,
            'classes': classes,
            'uncertainty': uncertainty,
            'evidence_scores': evidence_scores,
            'intermediate': {
                'boxes': all_boxes,
                'classes': all_classes,
            }
        }

        return output


if __name__ == '__main__':
    # Test
    model = EvidenceToObjectDecoder(
        feature_dim=256,
        num_classes=80,
        num_iterations=3
    )

    B, K = 2, 200
    region_features = torch.randn(B, K, 256)
    coords = torch.rand(B, K, 2)
    evidence_scores = torch.rand(B, K, 1)

    with torch.no_grad():
        output = model(region_features, coords, evidence_scores)

    print(f"Region features shape: {region_features.shape}")
    print(f"Coords shape: {coords.shape}")
    print(f"Evidence scores shape: {evidence_scores.shape}")
    print(f"\nOutput:")
    print(f"  Boxes shape: {output['boxes'].shape}")
    print(f"  Classes shape: {output['classes'].shape}")
    print(f"  Uncertainty shape: {output['uncertainty'].shape}")
    print(f"  Evidence scores shape: {output['evidence_scores'].shape}")
    print(f"\nIntermediate predictions:")
    print(f"  Number of iterations: {len(output['intermediate']['boxes'])}")

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params / 1e6:.2f}M")
