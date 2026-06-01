"""
AURA-NET v3.0 - Dual-Stream Feature Extractor
==============================================

Novel Component #1: Separates edge and semantic information from input.

Key innovations:
1. Learnable Sobel filters (not fixed)
2. Bidirectional cross-stream gating
3. Frequency-aware feature enhancement
4. Edge-semantic fusion with learnable weights

Author: AURA-NET Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnableSobelConv(nn.Module):
    """
    Learnable Sobel Convolution

    Initialized with Sobel kernels but learnable during training.
    Extracts edge information from input images.
    """
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        # Sobel kernels for initialization
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        # Create learnable conv
        self.edge_conv_x = nn.Conv2d(in_channels, out_channels // 2, 3, 1, 1, bias=False)
        self.edge_conv_y = nn.Conv2d(in_channels, out_channels // 2, 3, 1, 1, bias=False)

        # Initialize with Sobel
        with torch.no_grad():
            for i in range(in_channels):
                for j in range(out_channels // 2):
                    self.edge_conv_x.weight[j, i] = sobel_x[0, 0]
                    self.edge_conv_y.weight[j, i] = sobel_y[0, 0]

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        edge_x = self.edge_conv_x(x)
        edge_y = self.edge_conv_y(x)
        edge = torch.cat([edge_x, edge_y], dim=1)
        edge = self.bn(edge)
        edge = self.act(edge)
        return edge


class EdgeEnhancementBlock(nn.Module):
    """
    Edge Enhancement Block

    Enhances edge features with residual connections and attention.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Spatial attention for edges
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_channels, 1, 7, 1, 3, bias=False),
            nn.Sigmoid()
        )

        # Downsample if needed
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.act2 = nn.GELU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply spatial attention
        attn = self.spatial_attn(out)
        out = out * attn

        # Residual
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.act2(out)

        return out


class SemanticBlock(nn.Module):
    """
    Semantic Block

    Extracts semantic/content features with standard convolutions.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

        # Downsample if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.act2 = nn.GELU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply channel attention
        attn = self.channel_attn(out)
        out = out * attn

        # Residual
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.act2(out)

        return out


class BidirectionalGate(nn.Module):
    """
    Bidirectional Cross-Stream Gate

    Novel: Uses gating mechanism (not simple addition/concat) to fuse
    edge and semantic streams. Each stream can modulate the other.
    """
    def __init__(self, edge_channels, semantic_channels):
        super().__init__()

        # Edge -> Semantic gate
        self.edge_to_semantic = nn.Sequential(
            nn.Conv2d(edge_channels, semantic_channels, 1, bias=False),
            nn.BatchNorm2d(semantic_channels),
            nn.Sigmoid()
        )

        # Semantic -> Edge gate
        self.semantic_to_edge = nn.Sequential(
            nn.Conv2d(semantic_channels, edge_channels, 1, bias=False),
            nn.BatchNorm2d(edge_channels),
            nn.Sigmoid()
        )

        # Self-gates
        self.edge_self_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(edge_channels, edge_channels, 1),
            nn.Sigmoid()
        )

        self.semantic_self_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(semantic_channels, semantic_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, edge_feat, semantic_feat):
        """
        Args:
            edge_feat: (B, C_edge, H, W)
            semantic_feat: (B, C_semantic, H, W)

        Returns:
            edge_gated: (B, C_edge, H, W)
            semantic_gated: (B, C_semantic, H, W)
        """
        # Cross-stream gates
        edge_gate = self.semantic_to_edge(semantic_feat)
        semantic_gate = self.edge_to_semantic(edge_feat)

        # Self-gates
        edge_self = self.edge_self_gate(edge_feat)
        semantic_self = self.semantic_self_gate(semantic_feat)

        # Apply gates (multiplicative)
        edge_gated = edge_feat * edge_gate * edge_self
        semantic_gated = semantic_feat * semantic_gate * semantic_self

        return edge_gated, semantic_gated


class DualStreamFeatureExtractor(nn.Module):
    """
    Dual-Stream Feature Extractor (DSFE)

    Novel architecture that separates edge and semantic processing
    from the input, then fuses them with bidirectional gating.

    Architecture:
        Input (3, 640, 640)
        ├─ Edge Stream: Sobel → EdgeBlocks → (128, 80, 80)
        └─ Semantic Stream: Conv → SemanticBlocks → (128, 80, 80)

        Cross-Stream Gating → Fusion → (128, 80, 80)

    Args:
        in_channels: Input channels (default: 3)
        out_channels: Output channels (default: 128)
    """
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()

        # Edge stream
        self.edge_conv = LearnableSobelConv(in_channels, 64)
        self.edge_blocks = nn.Sequential(
            EdgeEnhancementBlock(64, 64),
            nn.MaxPool2d(2, 2),  # 320x320
            EdgeEnhancementBlock(64, 96),
            nn.MaxPool2d(2, 2),  # 160x160
            EdgeEnhancementBlock(96, out_channels),
            nn.MaxPool2d(2, 2),  # 80x80
        )

        # Semantic stream
        self.semantic_conv = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.semantic_bn = nn.BatchNorm2d(64)
        self.semantic_act = nn.GELU()
        self.semantic_blocks = nn.Sequential(
            SemanticBlock(64, 64),
            nn.MaxPool2d(2, 2),  # 160x160
            SemanticBlock(64, 96),
            nn.MaxPool2d(2, 2),  # 80x80
            SemanticBlock(96, out_channels),
        )

        # Cross-stream gating
        self.cross_gate = BidirectionalGate(out_channels, out_channels)

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 640, 640)

        Returns:
            fused: (B, 128, 80, 80)
        """
        # Edge stream
        edge_feat = self.edge_conv(x)
        edge_feat = self.edge_blocks(edge_feat)

        # Semantic stream
        semantic_feat = self.semantic_conv(x)
        semantic_feat = self.semantic_bn(semantic_feat)
        semantic_feat = self.semantic_act(semantic_feat)
        semantic_feat = self.semantic_blocks(semantic_feat)

        # Cross-stream gating
        edge_gated, semantic_gated = self.cross_gate(edge_feat, semantic_feat)

        # Fusion
        fused = torch.cat([edge_gated, semantic_gated], dim=1)
        fused = self.fusion(fused)

        return fused


if __name__ == '__main__':
    # Test
    model = DualStreamFeatureExtractor(in_channels=3, out_channels=128)
    x = torch.randn(2, 3, 640, 640)

    with torch.no_grad():
        out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")
