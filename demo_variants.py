"""
AURA-NET Variants Demo
======================

Test and compare all model variants.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models import (
    AURANet_Nano, AURANet_Tiny, AURANet_S,
    AURANet_M, AURANet_L, AURANet_X,
    get_model_info
)


def test_variants():
    """Test all model variants"""
    print("\n" + "="*90)
    print("AURA-NET Model Variants Test - Optimized for End Devices")
    print("="*90)

    variants = {
        'Nano': AURANet_Nano,
        'Tiny': AURANet_Tiny,
        'S (Small)': AURANet_S,
        'M (Medium)': AURANet_M,
        'L (Large)': AURANet_L,
        'X (Extra)': AURANet_X
    }

    num_classes = 9  # Example: BCCD dataset

    print(f"\n{'Variant':<15} {'Params (M)':<12} {'Feature':<10} {'K Range':<12} {'Iter':<6} {'Target Device':<30}")
    print("-"*90)

    targets = {
        'Nano': 'MCU, Raspberry Pi, IoT',
        'Tiny': 'Mobile phones, tablets',
        'S (Small)': 'Modern smartphones, edge',
        'M (Medium)': 'Standard GPUs, workstations',
        'L (Large)': 'Edge servers, high-end PC',
        'X (Extra)': 'Cloud servers, research'
    }

    for name, model_fn in variants.items():
        model = model_fn(num_classes=num_classes)
        params = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"{name:<15} {params:<12.2f} {model.feature_dim:<10} "
              f"{model.min_K}-{model.max_K:<7} {model.object_decoder.num_iterations:<6} {targets[name]:<30}")

    print("="*90)

    # Test forward pass
    print("\nTesting forward pass (640x640 image)...")
    print(f"\n{'Variant':<15} {'Time (ms)':<12} {'Memory (MB)':<12} {'FPS':<10} {'Status':<15}")
    print("-"*90)

    x = torch.randn(1, 3, 640, 640)

    for name, model_fn in variants.items():
        model = model_fn(num_classes=num_classes)
        model.eval()

        try:
            import time
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start = time.time()
            with torch.no_grad():
                output = model(x)
            elapsed = (time.time() - start) * 1000

            # Estimate memory
            mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            fps = 1000 / elapsed if elapsed > 0 else 0

            print(f"{name:<15} {elapsed:<12.1f} {mem:<12.1f} {fps:<10.1f} {'OK':<15}")

        except Exception as e:
            print(f"{name:<15} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'Error':<15}")
            print(f"  Error: {e}")

    print("="*90)

    # Test inference
    print("\nTesting inference with NMS...")
    print(f"\n{'Variant':<15} {'Detections':<12} {'Time (ms)':<12} {'FPS':<10}")
    print("-"*90)

    for name, model_fn in variants.items():
        model = model_fn(num_classes=num_classes)
        model.eval()

        try:
            import time
            start = time.time()
            with torch.no_grad():
                predictions = model.predict(x, conf_threshold=0.25)
            elapsed = (time.time() - start) * 1000
            fps = 1000 / elapsed if elapsed > 0 else 0

            num_dets = len(predictions[0]['boxes'])
            print(f"{name:<15} {num_dets:<12} {elapsed:<12.1f} {fps:<10.1f}")

        except Exception as e:
            print(f"{name:<15} {'Error':<12} {'N/A':<12} {'N/A':<10}")

    print("="*90)


def compare_batch_sizes():
    """Test different batch sizes"""
    print("\n" + "="*90)
    print("Batch Size Comparison (End Device Focus)")
    print("="*90)

    variants = {
        'Nano': AURANet_Nano,
        'Tiny': AURANet_Tiny,
        'S': AURANet_S,
        'M': AURANet_M,
        'L': AURANet_L,
    }

    batch_sizes = [1, 2, 4, 8]

    for name, model_fn in variants.items():
        print(f"\nAURA-NET-{name}:")
        print(f"{'Batch Size':<15} {'Time (ms)':<15} {'Time/Image (ms)':<20}")
        print("-"*50)

        model = model_fn(num_classes=9)
        model.eval()

        for bs in batch_sizes:
            try:
                x = torch.randn(bs, 3, 640, 640)

                import time
                start = time.time()
                with torch.no_grad():
                    output = model(x)
                elapsed = (time.time() - start) * 1000

                per_image = elapsed / bs
                print(f"{bs:<15} {elapsed:<15.1f} {per_image:<20.1f}")

            except Exception as e:
                print(f"{bs:<15} {'OOM/Error':<15} {'N/A':<20}")

    print("="*90)


def main():
    """Main function"""
    print("\n" + "="*90)
    print("AURA-NET Model Variants Demo - Optimized for End Devices")
    print("="*90)

    # Test all variants
    test_variants()

    # Compare batch sizes
    compare_batch_sizes()

    # Show model info
    print("\n")
    get_model_info()

    print("\n" + "="*90)
    print("Demo completed!")
    print("="*90)
    print("\nTo use a specific variant:")
    print("  from models import AURANet_Nano, AURANet_Tiny, AURANet_S, AURANet_M, AURANet_L, AURANet_X")
    print("\n  # For end devices:")
    print("  model = AURANet_Nano(num_classes=9)  # MCU, Raspberry Pi")
    print("  model = AURANet_Tiny(num_classes=9)  # Mobile phones")
    print("  model = AURANet_S(num_classes=9)     # Smartphones")
    print("\n  # For standard devices:")
    print("  model = AURANet_M(num_classes=9)     # Default (balanced)")
    print("  model = AURANet_L(num_classes=9)     # Edge servers (optimized)")
    print("  model = AURANet_X(num_classes=9)     # Cloud servers")
    print("="*90 + "\n")


if __name__ == '__main__':
    main()
