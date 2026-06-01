"""
Quick test for AURA-NET variants import
"""

import torch

print("Testing AURA-NET variants import...")
print("=" * 80)

# Test import with underscore
print("\n1. Testing import with underscore:")
try:
    from models import AURANet_Nano, AURANet_Tiny, AURANet_S, AURANet_M, AURANet_L, AURANet_X
    print("   ✓ AURANet_Nano, AURANet_Tiny, AURANet_S, AURANet_M, AURANet_L, AURANet_X")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test import without underscore
print("\n2. Testing import without underscore:")
try:
    from models import AURANetNano, AURANetTiny, AURANetS, AURANetM, AURANetL, AURANetX
    print("   ✓ AURANetNano, AURANetTiny, AURANetS, AURANetM, AURANetL, AURANetX")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test model creation
print("\n3. Testing model creation:")
try:
    model_nano = AURANetNano(num_classes=9)
    print(f"   ✓ AURANetNano: {sum(p.numel() for p in model_nano.parameters()) / 1e6:.2f}M params")

    model_tiny = AURANetTiny(num_classes=9)
    print(f"   ✓ AURANetTiny: {sum(p.numel() for p in model_tiny.parameters()) / 1e6:.2f}M params")

    model_s = AURANetS(num_classes=9)
    print(f"   ✓ AURANetS: {sum(p.numel() for p in model_s.parameters()) / 1e6:.2f}M params")

    model_m = AURANetM(num_classes=9)
    print(f"   ✓ AURANetM: {sum(p.numel() for p in model_m.parameters()) / 1e6:.2f}M params")

    model_l = AURANetL(num_classes=9)
    print(f"   ✓ AURANetL: {sum(p.numel() for p in model_l.parameters()) / 1e6:.2f}M params")

    model_x = AURANetX(num_classes=9)
    print(f"   ✓ AURANetX: {sum(p.numel() for p in model_x.parameters()) / 1e6:.2f}M params")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test forward pass
print("\n4. Testing forward pass:")
try:
    x = torch.randn(1, 3, 640, 640)

    model_tiny = AURANetTiny(num_classes=9)
    model_tiny.eval()

    with torch.no_grad():
        output = model_tiny(x)

    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Output boxes shape: {output['boxes'].shape}")
    print(f"   ✓ Output classes shape: {output['classes'].shape}")
    print(f"   ✓ Output scores shape: {output['scores'].shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test predict method
print("\n5. Testing predict method:")
try:
    x = torch.randn(1, 3, 640, 640)

    model_tiny = AURANetTiny(num_classes=9)
    model_tiny.eval()

    predictions = model_tiny.predict(x, conf_threshold=0.25)

    print(f"   ✓ Predict method successful")
    print(f"   ✓ Number of detections: {len(predictions[0]['boxes'])}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)

print("\n✅ You can now use:")
print("   from models import AURANetTiny")
print("   model = AURANetTiny(num_classes=9)")
print("\nOr with underscore:")
print("   from models import AURANet_Tiny")
print("   model = AURANet_Tiny(num_classes=9)")
