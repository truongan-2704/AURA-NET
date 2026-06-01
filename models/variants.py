"""
AURA-NET Model Variants
=======================

Different model sizes: S (Small), M (Medium), L (Large)

Scaling strategy:
- Feature dimensions
- Number of evidence regions (K)
- Decoder iterations
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.v3.dual_stream_extractor import DualStreamFeatureExtractor
from models.v3.evidence_proposal_network import EvidenceProposalNetwork
from models.v3.adaptive_evidence_router import AdaptiveEvidenceRouter
from models.v3.evidence_to_object_decoder import EvidenceToObjectDecoder


class AURANetBase(nn.Module):
    """Base AURA-NET architecture"""

    def __init__(
        self,
        num_classes=80,
        feature_dim=256,
        min_K=50,
        max_K=150,
        num_iterations=3,
        extractor_channels=128
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.min_K = min_K
        self.max_K = max_K

        # 1. Dual-Stream Feature Extractor
        self.feature_extractor = DualStreamFeatureExtractor(
            in_channels=3,
            out_channels=extractor_channels
        )

        # 2. Evidence Proposal Network
        self.evidence_network = EvidenceProposalNetwork(
            in_channels=extractor_channels,
            hidden_channels=feature_dim
        )

        # 3. Adaptive Evidence Router
        self.evidence_router = AdaptiveEvidenceRouter(
            feature_dim=feature_dim,
            min_K=min_K,
            max_K=max_K
        )

        # 4. Evidence-to-Object Decoder
        self.object_decoder = EvidenceToObjectDecoder(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_iterations=num_iterations
        )

    def forward(self, x, return_intermediate=False):
        # Extract features
        features = self.feature_extractor(x)

        # Generate evidence map
        evidence_map, evidence_features = self.evidence_network(features)

        # Route evidence regions
        region_features, coords, evidence_scores, routing_info = \
            self.evidence_router(evidence_features, evidence_map)

        # Decode to objects
        output = self.object_decoder(region_features, coords, evidence_scores)

        # Add evidence map to output
        output['evidence_map'] = evidence_map
        output['routing_info'] = routing_info

        if return_intermediate:
            output['features'] = features
            output['evidence_features'] = evidence_features

        return output

    @torch.no_grad()
    def predict(self, x, conf_threshold=0.25, iou_threshold=0.5):
        """Inference mode with NMS"""
        self.eval()
        outputs = self.forward(x)

        batch_size = x.size(0)
        predictions = []

        for i in range(batch_size):
            boxes = outputs['boxes'][i]
            classes = outputs['classes'][i]
            scores = outputs['scores'][i]

            # Filter by confidence
            mask = scores >= conf_threshold
            boxes = boxes[mask]
            classes = classes[mask]
            scores = scores[mask]

            if len(boxes) == 0:
                predictions.append({
                    'boxes': torch.zeros((0, 4)),
                    'scores': torch.zeros(0),
                    'class_ids': torch.zeros(0, dtype=torch.long)
                })
                continue

            # Get class predictions
            class_probs = torch.softmax(classes, dim=-1)
            class_scores, class_ids = class_probs.max(dim=-1)

            # Apply NMS
            keep = self.nms(boxes, class_scores, iou_threshold)

            predictions.append({
                'boxes': boxes[keep],
                'scores': class_scores[keep],
                'class_ids': class_ids[keep]
            })

        return predictions

    @staticmethod
    def nms(boxes, scores, iou_threshold):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return torch.zeros(0, dtype=torch.long)

        # Convert to xyxy
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])

            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            # Keep boxes with IoU < threshold
            mask = iou <= iou_threshold
            order = order[1:][mask]

        return torch.tensor(keep, dtype=torch.long)


# ============================================================================
# Model Variants
# ============================================================================

def AURANet_Nano(num_classes=80):
    """
    AURA-NET Nano
    - Ultra-lightweight for MCU/embedded devices
    - ~0.8M parameters
    - Maximum speed, minimal memory
    - Target: Raspberry Pi, Jetson Nano, mobile devices
    """
    return AURANetBase(
        num_classes=num_classes,
        feature_dim=64,           # Minimal
        min_K=15,                 # Very few evidence regions
        max_K=40,
        num_iterations=1,         # Single iteration
        extractor_channels=32     # Minimal channels
    )


def AURANet_Tiny(num_classes=80):
    """
    AURA-NET Tiny
    - Very lightweight for mobile devices
    - ~1.5M parameters
    - Very fast inference
    - Target: Mobile phones, tablets, edge devices
    """
    return AURANetBase(
        num_classes=num_classes,
        feature_dim=96,           # Very small
        min_K=20,
        max_K=60,
        num_iterations=2,
        extractor_channels=48
    )


def AURANet_S(num_classes=80):
    """
    AURA-NET Small
    - Lightweight for modern mobile devices
    - ~2.5M parameters
    - Fast inference with good accuracy
    - Target: Modern smartphones, tablets, edge servers
    """
    return AURANetBase(
        num_classes=num_classes,
        feature_dim=128,
        min_K=30,
        max_K=80,
        num_iterations=2,
        extractor_channels=64
    )


def AURANet_M(num_classes=80):
    """
    AURA-NET Medium (Default)
    - Balanced speed and accuracy
    - ~9.6M parameters
    - Good for most applications
    - Target: Standard GPUs, workstations
    """
    return AURANetBase(
        num_classes=num_classes,
        feature_dim=256,
        min_K=50,
        max_K=150,
        num_iterations=3,
        extractor_channels=128
    )


def AURANet_L(num_classes=80):
    """
    AURA-NET Large
    - High accuracy for edge servers
    - ~18M parameters (optimized from 25M)
    - Good accuracy with reasonable speed
    - Target: Edge servers, high-end workstations
    """
    return AURANetBase(
        num_classes=num_classes,
        feature_dim=320,          # Reduced from 384
        min_K=60,                 # Reduced from 80
        max_K=200,                # Reduced from 250
        num_iterations=3,         # Reduced from 4
        extractor_channels=160    # Reduced from 192
    )


def AURANet_X(num_classes=80):
    """
    AURA-NET Extra Large
    - Research/competition grade
    - ~45M parameters
    - Best accuracy, slowest
    - Target: High-end GPUs, cloud servers
    """
    return AURANetBase(
        num_classes=num_classes,
        feature_dim=512,
        min_K=100,
        max_K=300,
        num_iterations=5,
        extractor_channels=256
    )


# Aliases for backward compatibility and easier usage
AURANet = AURANet_M

# Aliases without underscore (easier to type)
AURANetNano = AURANet_Nano
AURANetTiny = AURANet_Tiny
AURANetS = AURANet_S
AURANetM = AURANet_M
AURANetL = AURANet_L
AURANetX = AURANet_X


# ============================================================================
# Model Info
# ============================================================================

def get_model_info():
    """Get information about all model variants"""
    import torch

    variants = {
        'Nano': AURANet_Nano,
        'Tiny': AURANet_Tiny,
        'S': AURANet_S,
        'M': AURANet_M,
        'L': AURANet_L,
        'X': AURANet_X
    }

    print("=" * 90)
    print("AURA-NET Model Variants (Optimized for End Devices)")
    print("=" * 90)
    print(f"\n{'Variant':<10} {'Params (M)':<12} {'Feature':<10} {'K Range':<12} {'Iter':<6} {'Target Device':<30}")
    print("-" * 90)

    targets = {
        'Nano': 'MCU, Raspberry Pi, IoT',
        'Tiny': 'Mobile phones, tablets',
        'S': 'Modern smartphones, edge',
        'M': 'Standard GPUs, workstations',
        'L': 'Edge servers, high-end PC',
        'X': 'Cloud servers, research'
    }

    for name, model_fn in variants.items():
        model = model_fn(num_classes=80)
        params = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"{name:<10} {params:<12.2f} {model.feature_dim:<10} "
              f"{model.min_K}-{model.max_K:<7} {model.object_decoder.num_iterations:<6} {targets[name]:<30}")

    print("=" * 90)

    # Test forward pass
    print("\nTesting forward pass (640x640 image)...")
    x = torch.randn(1, 3, 640, 640)

    for name, model_fn in variants.items():
        model = model_fn(num_classes=80)
        model.eval()

        with torch.no_grad():
            import time
            start = time.time()
            output = model(x)
            elapsed = (time.time() - start) * 1000

        print(f"  {name:<10}: {elapsed:>6.1f}ms")

    print("=" * 90)


if __name__ == '__main__':
    get_model_info()
