"""
AURA-NET v2.0 - Complete Architecture

Evidence-Guided Adaptive Multi-Scale Object Detection

Novel Components:
1. EACS (Edge-Aware Compact Stem)
2. CFPBlock (Context-Frequency Progressive Block) Backbone
3. ASMNeck (Adaptive Scale Mixing Neck)
4. CUPHead (Compact Unified Prediction Head)

This is a complete redesign with genuine novel contributions,
NOT copying YOLO, EfficientDet, or other existing architectures.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.eacs_stem import EACS
from models.cfp_block import CFPBlock
from models.asm_neck import ASMNeck
from models.cup_head import MultiScaleCUPHead


class AURANetV2(nn.Module):
    """
    AURA-NET v2.0

    Evidence-Guided Adaptive Multi-Scale Object Detection

    Architecture:
    1. EACS Stem: Dual-path edge-semantic separation
    2. CFPBlock Backbone: Context-frequency progressive fusion
    3. ASMNeck: Evidence-driven adaptive scale mixing
    4. CUPHead: Unified prediction with task embeddings

    Args:
        num_classes: Number of object classes
        variant: Model variant ('nano', 'small', 'medium')
        img_size: Input image size (default: 640)
    """
    def __init__(self, num_classes=80, variant='small', img_size=640):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant
        self.img_size = img_size

        # Get architecture config
        config = self._get_config(variant)
        self.config = config

        # 1. EACS Stem
        self.stem = EACS(
            in_channels=3,
            edge_channels=config['stem']['edge_channels'],
            semantic_channels=config['stem']['semantic_channels'],
            out_channels=config['stem']['out_channels']
        )

        # 2. CFPBlock Backbone
        self.backbone = self._build_backbone(config)

        # 3. ASMNeck
        backbone_channels = config['backbone']['channels']
        neck_channels = [backbone_channels[1], backbone_channels[2], backbone_channels[3]]
        target_size = (img_size // 8, img_size // 8)

        self.neck = ASMNeck(
            channels_list=neck_channels,
            target_size=target_size,
            out_channels=config['neck']['out_channels']
        )

        # 4. CUPHead
        self.head = MultiScaleCUPHead(
            in_channels=config['neck']['out_channels'],
            num_classes=num_classes,
            num_scales=3,
            num_anchors=1
        )

    def _get_config(self, variant):
        """Get architecture configuration for variant"""
        configs = {
            'nano': {
                'stem': {
                    'edge_channels': 8,
                    'semantic_channels': 16,
                    'out_channels': 32
                },
                'backbone': {
                    'channels': [32, 64, 128, 256],
                    'num_blocks': [2, 2, 3, 2],
                    'strides': [1, 2, 2, 2]
                },
                'neck': {
                    'out_channels': 128
                }
            },
            'small': {
                'stem': {
                    'edge_channels': 16,
                    'semantic_channels': 32,
                    'out_channels': 64
                },
                'backbone': {
                    'channels': [64, 128, 256, 512],
                    'num_blocks': [2, 3, 4, 2],
                    'strides': [1, 2, 2, 2]
                },
                'neck': {
                    'out_channels': 256
                }
            },
            'medium': {
                'stem': {
                    'edge_channels': 24,
                    'semantic_channels': 48,
                    'out_channels': 96
                },
                'backbone': {
                    'channels': [96, 192, 384, 768],
                    'num_blocks': [3, 4, 6, 3],
                    'strides': [1, 2, 2, 2]
                },
                'neck': {
                    'out_channels': 384
                }
            }
        }

        if variant not in configs:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")

        return configs[variant]

    def _build_backbone(self, config):
        """Build CFPBlock backbone"""
        channels = config['backbone']['channels']
        num_blocks = config['backbone']['num_blocks']
        strides = config['backbone']['strides']

        layers = []
        in_channels = channels[0]

        for stage_idx, (out_channels, num_block, stride) in enumerate(zip(channels, num_blocks, strides)):
            # First block in stage (with stride)
            layers.append(
                CFPBlock(in_channels, out_channels, stage_idx=stage_idx, stride=stride)
            )

            # Remaining blocks in stage
            for _ in range(num_block - 1):
                layers.append(
                    CFPBlock(out_channels, out_channels, stage_idx=stage_idx, stride=1)
                )

            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (B, 3, H, W) input images
        Returns:
            cls_preds: List of classification predictions for each scale
            box_preds: List of box predictions for each scale
            unc_preds: List of uncertainty predictions for each scale
        """
        # 1. Stem
        x = self.stem(x)  # (B, C0, H/2, W/2)

        # 2. Backbone - extract multi-scale features
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # Collect features at specific stages for neck
            # Stage 1 (after 2 blocks): H/4, W/4
            # Stage 2 (after 2+3 blocks): H/8, W/8
            # Stage 3 (after 2+3+4 blocks): H/16, W/16
            if self.variant == 'small':
                if i == 1:  # After stage 1
                    features.append(x)
                elif i == 4:  # After stage 2
                    features.append(x)
                elif i == 8:  # After stage 3
                    features.append(x)
            elif self.variant == 'nano':
                if i == 1:  # After stage 1
                    features.append(x)
                elif i == 3:  # After stage 2
                    features.append(x)
                elif i == 6:  # After stage 3
                    features.append(x)
            elif self.variant == 'medium':
                if i == 2:  # After stage 1
                    features.append(x)
                elif i == 6:  # After stage 2
                    features.append(x)
                elif i == 12:  # After stage 3
                    features.append(x)

        # 3. Neck - adaptive scale mixing
        neck_outputs, evidence_maps = self.neck(features)

        # 4. Head - unified prediction
        cls_preds, box_preds, unc_preds = self.head(neck_outputs)

        return cls_preds, box_preds, unc_preds

    def forward_with_evidence(self, x):
        """
        Forward pass with evidence maps for visualization

        Args:
            x: (B, 3, H, W) input images
        Returns:
            cls_preds: List of classification predictions
            box_preds: List of box predictions
            unc_preds: List of uncertainty predictions
            evidence_maps: List of evidence maps
        """
        # Stem
        x = self.stem(x)

        # Backbone
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if self.variant == 'small':
                if i in [1, 4, 8]:
                    features.append(x)
            elif self.variant == 'nano':
                if i in [1, 3, 6]:
                    features.append(x)
            elif self.variant == 'medium':
                if i in [2, 6, 12]:
                    features.append(x)

        # Neck
        neck_outputs, evidence_maps = self.neck(features)

        # Head
        cls_preds, box_preds, unc_preds = self.head(neck_outputs)

        return cls_preds, box_preds, unc_preds, evidence_maps


def build_aura_net_v2(config):
    """
    Build AURA-NET v2 from config

    Args:
        config: Dict with keys:
            - num_classes: Number of classes
            - variant: 'nano', 'small', or 'medium'
            - img_size: Input image size
    Returns:
        model: AURANetV2 instance
    """
    num_classes = config.get('num_classes', 80)
    variant = config.get('variant', 'small')
    img_size = config.get('img_size', 640)

    model = AURANetV2(
        num_classes=num_classes,
        variant=variant,
        img_size=img_size
    )

    return model


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_info(model, img_size=640):
    """Get model information"""
    total_params, trainable_params = count_parameters(model)

    # Estimate FLOPs (rough estimate)
    # Will use thop library if available
    try:
        from thop import profile, clever_format
        input_tensor = torch.randn(1, 3, img_size, img_size)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
    except:
        flops = "N/A (install thop: pip install thop)"
        params = f"{total_params / 1e6:.2f}M"

    info = {
        'variant': model.variant,
        'num_classes': model.num_classes,
        'img_size': model.img_size,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_str': params,
        'flops_str': flops
    }

    return info


if __name__ == '__main__':
    print('='*80)
    print('AURA-NET v2.0 - Architecture Test')
    print('='*80)

    # Test all variants
    variants = ['nano', 'small', 'medium']
    img_size = 640
    num_classes = 80

    for variant in variants:
        print(f'\n--- Testing {variant.upper()} variant ---')

        model = AURANetV2(num_classes=num_classes, variant=variant, img_size=img_size)
        model.eval()

        # Forward pass
        x = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            cls_preds, box_preds, unc_preds = model(x)

        # Print output shapes
        print(f'\nInput: {x.shape}')
        print(f'Number of detection scales: {len(cls_preds)}')
        for i, (cls, box, unc) in enumerate(zip(cls_preds, box_preds, unc_preds)):
            print(f'  Scale {i}: cls={cls.shape}, box={box.shape}, unc={unc.shape}')

        # Model info
        info = get_model_info(model, img_size)
        print(f'\nModel Info:')
        print(f'  Parameters: {info["params_str"]}')
        print(f'  FLOPs: {info["flops_str"]}')

    # Test with evidence maps
    print(f'\n--- Testing with Evidence Maps ---')
    model = AURANetV2(num_classes=80, variant='small', img_size=640)
    model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        cls_preds, box_preds, unc_preds, evidence_maps = model.forward_with_evidence(x)

    print(f'Evidence maps: {len(evidence_maps)}')
    for i, ev in enumerate(evidence_maps):
        print(f'  Evidence {i}: {ev.shape}')

    print('\n' + '='*80)
    print('AURA-NET v2.0 - All tests passed!')
    print('='*80)
