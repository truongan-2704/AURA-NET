"""
AURA-NET - Forward Pass Test
=============================

Test all modules and compute model complexity.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.v3 import AURANet
from losses.aura_v3_loss import AURANetLoss


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model, input_size=(1, 3, 640, 640)):
    """
    Estimate FLOPs (simplified).
    For accurate measurement, use thop or fvcore.
    """
    try:
        from thop import profile, clever_format
        input_tensor = torch.randn(*input_size)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    except ImportError:
        return "N/A (install thop)", "N/A"


def test_forward_pass():
    """Test forward pass with different batch sizes."""
    print("=" * 80)
    print("AURA-NET v3.0 - Forward Pass Test")
    print("=" * 80)

    # Create model    print("\n[1] Creating model...")
    model = AURANet(
        num_classes=80,
        feature_dim=256,
        min_K=100,
        max_K=300,
        num_iterations=3
    )
    model.eval()

    # Count parameters
    params = count_parameters(model)
    print(f"    Total parameters: {params / 1e6:.2f}M")

    # Test different batch sizes
    batch_sizes = [1, 2, 4]
    image_size = (640, 640)

    for bs in batch_sizes:
        print(f"\n[2] Testing batch size {bs}...")
        x = torch.randn(bs, 3, *image_size)

        try:
            with torch.no_grad():
                # Training mode output
                output = model(x, return_intermediate=True)

                print(f"    [OK] Forward pass successful")
                print(f"      - Boxes: {output['boxes'].shape}")
                print(f"      - Classes: {output['classes'].shape}")
                print(f"      - Scores: {output['scores'].shape}")
                print(f"      - Evidence map: {output['evidence_map'].shape}")
                print(f"      - Uncertainty: {output['uncertainty'].shape}")
                print(f"      - Routing K: {output['routing_info']['K']}")

                # Inference mode
                predictions = model.predict(x, conf_threshold=0.25, iou_threshold=0.5)
                print(f"    [OK] Inference successful")
                for i, pred in enumerate(predictions):
                    print(f"      - Image {i}: {len(pred['boxes'])} detections")

        except Exception as e:
            print(f"    [ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Test loss
    print(f"\n[3] Testing loss function...")
    loss_fn = AURANetLoss(num_classes=80)

    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        output = model(x)

    # Dummy targets
    targets = [
        torch.tensor([[0.5, 0.5, 0.2, 0.3, 0], [0.3, 0.7, 0.1, 0.2, 1]]),
        torch.tensor([[0.6, 0.4, 0.15, 0.25, 2]])
    ]

    try:
        loss_dict = loss_fn(output, targets)
        print(f"    [OK] Loss computation successful")
        print(f"      Loss components:")
        for k, v in loss_dict.items():
            print(f"        - {k}: {v.item():.4f}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compute FLOPs
    print(f"\n[4] Computing FLOPs...")
    flops, params_str = compute_flops(model, input_size=(1, 3, 640, 640))
    print(f"    FLOPs: {flops}")
    print(f"    Params: {params_str}")

    # Test gradient flow
    print(f"\n[5] Testing gradient flow...")
    model.train()
    x = torch.randn(1, 3, 640, 640, requires_grad=True)
    output = model(x)
    targets = [torch.tensor([[0.5, 0.5, 0.2, 0.3, 0]])]

    try:
        loss_dict = loss_fn(output, targets)
        loss = loss_dict['loss']
        loss.backward()
        print(f"    [OK] Gradient flow successful")
        print(f"      - Loss: {loss.item():.4f}")
        print(f"      - Input grad: {x.grad is not None}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("[OK] All tests passed!")
    print("=" * 80)

    return True


def test_individual_modules():
    """Test each module individually."""
    print("\n" + "=" * 80)
    print("Testing Individual Modules")
    print("=" * 80)

    from models.v3.dual_stream_extractor import DualStreamFeatureExtractor
    from models.v3.evidence_proposal_network import EvidenceProposalNetwork
    from models.v3.adaptive_evidence_router import AdaptiveEvidenceRouter
    from models.v3.evidence_to_object_decoder import EvidenceToObjectDecoder

    # Test Dual-Stream Extractor
    print("\n[1] Dual-Stream Feature Extractor...")
    dsfe = DualStreamFeatureExtractor(in_channels=3, out_channels=128)
    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        out = dsfe(x)
    print(f"    Input: {x.shape} -> Output: {out.shape}")
    print(f"    Parameters: {count_parameters(dsfe) / 1e6:.2f}M")

    # Test Evidence Proposal Network
    print("\n[2] Evidence Proposal Network...")
    epn = EvidenceProposalNetwork(in_channels=128, hidden_channels=256)
    x = torch.randn(2, 128, 80, 80)
    with torch.no_grad():
        evidence_map, features = epn(x)
    print(f"    Input: {x.shape}")
    print(f"    Evidence map: {evidence_map.shape}")
    print(f"    Features: {features.shape}")
    print(f"    Parameters: {count_parameters(epn) / 1e6:.2f}M")

    # Test Adaptive Evidence Router
    print("\n[3] Adaptive Evidence Router...")
    aer = AdaptiveEvidenceRouter(feature_dim=256, min_K=100, max_K=300)
    features = torch.randn(2, 256, 80, 80)
    evidence_map = torch.rand(2, 1, 80, 80)
    with torch.no_grad():
        output, coords, scores, routing_info = aer(features, evidence_map)
    print(f"    Features: {features.shape}")
    print(f"    Evidence map: {evidence_map.shape}")
    print(f"    Output: {output.shape}")
    print(f"    Coords: {coords.shape}")
    print(f"    Routing K: {routing_info['K']}")
    print(f"    Parameters: {count_parameters(aer) / 1e6:.2f}M")

    # Test Evidence-to-Object Decoder
    print("\n[4] Evidence-to-Object Decoder...")
    eod = EvidenceToObjectDecoder(feature_dim=256, num_classes=80, num_iterations=3)
    region_features = torch.randn(2, 200, 256)
    coords = torch.rand(2, 200, 2)
    evidence_scores = torch.rand(2, 200, 1)
    with torch.no_grad():
        output = eod(region_features, coords, evidence_scores)
    print(f"    Region features: {region_features.shape}")
    print(f"    Boxes: {output['boxes'].shape}")
    print(f"    Classes: {output['classes'].shape}")
    print(f"    Uncertainty: {output['uncertainty'].shape}")
    print(f"    Parameters: {count_parameters(eod) / 1e6:.2f}M")

    print("\n" + "=" * 80)
    print("[OK] All module tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    # Test individual modules first
    test_individual_modules()

    # Test complete model
    success = test_forward_pass()

    if success:
        print("\n[OK] AURA-NET v3.0 is ready for training!")
    else:
        print("\n[ERROR] Some tests failed. Please fix errors before training.")
        sys.exit(1)
