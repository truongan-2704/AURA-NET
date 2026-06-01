"""
Quick Demo - Test Apple Leaf Training Setup
============================================

Test if everything is set up correctly before full training.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models.v3 import AURANet
from losses.aura_v3_loss import AURANetLoss
from train_apple_leaf import AppleLeafDataset, collate_fn


def test_dataset():
    """Test dataset loading."""
    print("=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)

    dataset = AppleLeafDataset(
        'datasets/apple_leaft_detection',
        split='train',
        img_size=640,
        augment=False
    )

    print(f"\nDataset size: {len(dataset)}")

    # Load one sample
    image, boxes = dataset[0]

    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Boxes shape: {boxes.shape}")
    print(f"  Number of objects: {len(boxes)}")

    if len(boxes) > 0:
        print(f"\n  First box:")
        print(f"    cx={boxes[0, 0]:.3f}, cy={boxes[0, 1]:.3f}")
        print(f"    w={boxes[0, 2]:.3f}, h={boxes[0, 3]:.3f}")
        print(f"    class={int(boxes[0, 4])}")

    return True


def test_model():
    """Test model forward pass."""
    print("\n" + "=" * 60)
    print("Testing Model")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create model
    model = AURANet(
        num_classes=4,
        feature_dim=256,
        min_K=50,
        max_K=150,
        num_iterations=3
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 640, 640).to(device)

    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"\nOutput shapes:")
    print(f"  Boxes: {output['boxes'].shape}")
    print(f"  Classes: {output['classes'].shape}")
    print(f"  Scores: {output['scores'].shape}")
    print(f"  Evidence map: {output['evidence_map'].shape}")

    return True


def test_loss():
    """Test loss computation."""
    print("\n" + "=" * 60)
    print("Testing Loss Function")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model and loss
    model = AURANet(num_classes=4).to(device)
    criterion = AURANetLoss(num_classes=4)

    # Create dummy batch
    batch_size = 2
    images = torch.randn(batch_size, 3, 640, 640).to(device)

    # Create dummy targets
    targets = [
        torch.tensor([[0.5, 0.5, 0.2, 0.2, 0]], dtype=torch.float32).to(device),
        torch.tensor([[0.3, 0.3, 0.1, 0.1, 1],
                     [0.7, 0.7, 0.15, 0.15, 2]], dtype=torch.float32).to(device),
    ]

    print(f"\nBatch size: {batch_size}")
    print(f"Target 0: {len(targets[0])} objects")
    print(f"Target 1: {len(targets[1])} objects")

    # Forward pass
    with torch.no_grad():
        outputs = model(images)

    # Compute loss
    loss_dict = criterion(outputs, targets)

    print(f"\nLoss values:")
    print(f"  Total loss: {loss_dict['loss'].item():.4f}")
    print(f"  Evidence loss: {loss_dict['loss_evidence'].item():.4f}")
    print(f"  Box loss: {loss_dict['loss_box'].item():.4f}")
    print(f"  Class loss: {loss_dict['loss_class'].item():.4f}")

    return True


def test_dataloader():
    """Test dataloader with real data."""
    print("\n" + "=" * 60)
    print("Testing DataLoader with Real Data")
    print("=" * 60)

    from torch.utils.data import DataLoader

    dataset = AppleLeafDataset(
        'datasets/apple_leaft_detection',
        split='train',
        img_size=640,
        augment=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Use 0 for testing
        collate_fn=collate_fn
    )

    print(f"\nDataLoader created")
    print(f"  Batch size: 4")
    print(f"  Total batches: {len(dataloader)}")

    # Load one batch
    images, targets = next(iter(dataloader))

    print(f"\nFirst batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Number of targets: {len(targets)}")

    for i, target in enumerate(targets):
        print(f"  Image {i}: {len(target)} objects")

    return True


def main():
    print("\n" + "=" * 60)
    print("AURA-NET - Apple Leaf Training Setup Test")
    print("=" * 60)

    tests = [
        ("Dataset Loading", test_dataset),
        ("Model Forward Pass", test_model),
        ("Loss Computation", test_loss),
        ("DataLoader", test_dataloader),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
            print(f"\n[OK] {test_name} passed")
        except Exception as e:
            results.append((test_name, "FAIL"))
            print(f"\n[ERROR] {test_name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, result in results:
        status = "[OK]" if result == "PASS" else "[FAIL]"
        print(f"{status} {test_name}")

    all_passed = all(r[1] == "PASS" for r in results)

    if all_passed:
        print("\n" + "=" * 60)
        print("[SUCCESS] All tests passed!")
        print("You can now start training:")
        print("  python train_apple_leaf.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("[WARNING] Some tests failed. Please fix the issues before training.")
        print("=" * 60)


if __name__ == '__main__':
    main()
