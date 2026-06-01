"""
Test AURA-NET v3.0 with optimized GFLOPs
"""

import torch
from models.v3 import AURANetV3

def test_gflops():
    """Test different channel configurations for GFLOPs"""

    configs = [
        # Original
        {'name': 'Original', 'feature_dim': 256, 'min_K': 100, 'max_K': 300},
        # Optimized 1: Reduce feature dim
        {'name': 'Opt-1 (Reduce feature_dim)', 'feature_dim': 192, 'min_K': 100, 'max_K': 300},
        # Optimized 2: Reduce regions
        {'name': 'Opt-2 (Reduce regions)', 'feature_dim': 256, 'min_K': 64, 'max_K': 200},
        # Optimized 3: Combined
        {'name': 'Opt-3 (Combined)', 'feature_dim': 192, 'min_K': 64, 'max_K': 200},
        # Optimized 4: Aggressive
        {'name': 'Opt-4 (Aggressive)', 'feature_dim': 160, 'min_K': 64, 'max_K': 150},
        # Optimized 5: Very Aggressive
        {'name': 'Opt-5 (Very Aggressive)', 'feature_dim': 128, 'min_K': 50, 'max_K': 150},
    ]

    print("="*80)
    print("AURA-NET v3.0 - GFLOPs Optimization Test")
    print("="*80)

    for cfg in configs:
        model = AURANetV3(
            num_classes=80,
            feature_dim=cfg['feature_dim'],
            min_K=cfg['min_K'],
            max_K=cfg['max_K']
        )

        # Count parameters
        params = sum(p.numel() for p in model.parameters())

        # Compute FLOPs
        from thop import profile
        dummy_input = torch.randn(1, 3, 640, 640)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

        print(f"\n{cfg['name']}:")
        print(f"  Feature dim: {cfg['feature_dim']}")
        print(f"  Min K: {cfg['min_K']}, Max K: {cfg['max_K']}")
        print(f"  Parameters: {params/1e6:.2f}M")
        print(f"  GFLOPs: {flops/1e9:.3f}G")

        # Test forward pass
        with torch.no_grad():
            outputs = model(dummy_input)
            print(f"  Output boxes: {outputs['boxes'].shape}")
            print(f"  ✅ Forward pass OK")

if __name__ == '__main__':
    test_gflops()
