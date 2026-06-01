"""
AURA-Net: Adaptive Unified Region-Aware Network
Evidence-Guided Adaptive Region Architecture (EGAR)

Main model implementation
"""

import torch
import torch.nn as nn
from .stem import MultiResolutionVisualStem
from .evidence_encoder import LocalTextureEncoder
from .global_memory import GlobalSceneMemory
from .frequency_detail import FrequencyDetailModule
from .region_evidence import RegionEvidenceGenerator, MultiScaleEvidenceGenerator
from .adaptive_router import AdaptiveRegionProcessor
from .detail_inspector import DetailEvidenceInspector
from .fusion_decoder import EvidenceFusionDecoder


class AURANet(nn.Module):
    """
    AURA-Net: Adaptive Unified Region-Aware Network
    
    Architecture Pipeline:
    Input Image (B, 3, 640, 640)
    → Multi-Resolution Visual Stem (B, 64, 160, 160)
    → Local Texture Encoder (B, 160, 80, 80)
    → Global Scene Memory (B, 192, 80, 80)
    → Frequency Detail Encoder (B, 192, 80, 80)
    → Region Evidence Generator (evidence_map + evidence_feature)
    → Adaptive Region Processor (routed_feature)
    → Detail Evidence Inspector (detail_feature)
    → Evidence Fusion Decoder (predictions)
    
    Args:
        num_classes: Number of object classes
        img_size: Input image size (default: 640)
        channels: Dictionary of channel configurations
        top_k: Number of top regions to inspect
        use_topk_inspector: Whether to use top-k mode in inspector
        multi_scale_evidence: Whether to use multi-scale evidence generator
    """
    
    def __init__(self, 
                 num_classes=10,
                 img_size=640,
                 channels=None,
                 top_k=16,
                 use_topk_inspector=False,
                 multi_scale_evidence=False):
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Default channel configuration
        if channels is None:
            channels = {
                'stem': 64,
                'local': 160,
                'global': 192,
                'decoder': 192
            }
        
        self.channels = channels
        
        # Stage 0: Multi-Resolution Visual Stem
        self.stem = MultiResolutionVisualStem(
            in_channels=3,
            out_channels=channels['stem'],
            anti_alias=True
        )
        
        # Stage 1: Local Texture Encoder (Evidence Encoder - Part 1)
        self.local_encoder = LocalTextureEncoder(
            in_channels=channels['stem'],
            mid_channels=96,
            out_channels=channels['local'],
            drop_path=0.1
        )
        
        # Stage 2: Global Scene Memory (Evidence Encoder - Part 2)
        self.global_memory = GlobalSceneMemory(
            in_channels=channels['local'],
            out_channels=channels['global'],
            num_blocks=2
        )
        
        # Stage 3: Frequency Detail Encoder (Evidence Encoder - Part 3)
        self.frequency_detail = FrequencyDetailModule(
            in_channels=channels['global'],
            out_channels=channels['global'],
            num_layers=2
        )
        
        # Stage 4: Region Evidence Generator
        if multi_scale_evidence:
            self.evidence_generator = MultiScaleEvidenceGenerator(
                in_channels=channels['global'],
                hidden_channels=128
            )
        else:
            self.evidence_generator = RegionEvidenceGenerator(
                in_channels=channels['global'],
                hidden_channels=128
            )
        
        # Stage 5: Adaptive Region Processor
        self.adaptive_router = AdaptiveRegionProcessor(
            channels=channels['global']
        )
        
        # Stage 6: Detail Evidence Inspector
        self.detail_inspector = DetailEvidenceInspector(
            channels=channels['global'],
            top_k=top_k,
            use_topk=use_topk_inspector
        )
        
        # Stage 7: Evidence Fusion Decoder
        self.decoder = EvidenceFusionDecoder(
            in_channels=channels['decoder'],
            num_classes=num_classes,
            num_anchors=1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_evidence=False):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            return_evidence: Whether to return intermediate evidence maps
        
        Returns:
            predictions: Dictionary containing detection outputs
            evidence_info: (Optional) Dictionary containing evidence maps and features
        """
        # Stage 0: Visual Stem
        x = self.stem(x)  # (B, 64, 160, 160)
        
        # Stage 1: Local Texture Encoder
        x = self.local_encoder(x)  # (B, 160, 80, 80)
        
        # Stage 2: Global Scene Memory
        x = self.global_memory(x)  # (B, 192, 80, 80)
        
        # Stage 3: Frequency Detail Encoder
        x = self.frequency_detail(x)  # (B, 192, 80, 80)
        
        # Stage 4: Region Evidence Generator
        evidence_map, evidence_feature = self.evidence_generator(x)
        # evidence_map: (B, 1, 80, 80)
        # evidence_feature: (B, 192, 80, 80)
        
        # Stage 5: Adaptive Region Processor
        routed_feature, route_weights = self.adaptive_router(evidence_feature, evidence_map)
        # routed_feature: (B, 192, 80, 80)
        # route_weights: (B, 3)
        
        # Stage 6: Detail Evidence Inspector
        detail_feature = self.detail_inspector(routed_feature, evidence_map)
        # detail_feature: (B, 192, 80, 80)
        
        # Stage 7: Evidence Fusion Decoder
        predictions = self.decoder(routed_feature, detail_feature, evidence_map)
        
        if return_evidence:
            evidence_info = {
                'evidence_map': evidence_map,
                'evidence_feature': evidence_feature,
                'routed_feature': routed_feature,
                'route_weights': route_weights,
                'detail_feature': detail_feature
            }
            return predictions, evidence_info
        
        return predictions
    
    def predict(self, x, conf_thres=0.25, iou_thres=0.45):
        """
        Inference with post-processing

        Args:
            x: Input tensor (B, 3, H, W)
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS

        Returns:
            detections: List of detection tensors
        """
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from utils.boxes import non_max_suppression

        predictions = self.forward(x)
        detections = self.decoder.decode_predictions(predictions, conf_thres)

        # Apply NMS
        detections = non_max_suppression(detections, iou_thres)

        return detections


def build_aura_net(config):
    """
    Build AURA-Net from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        model: AURANet instance
    """
    model_config = config.get('model', {})
    
    # Extract parameters
    num_classes = model_config.get('num_classes', 10)
    img_size = model_config.get('img_size', 640)
    width_mult = model_config.get('width_mult', 1.0)
    top_k = model_config.get('top_k', 16)
    use_topk_inspector = model_config.get('use_topk_inspector', False)
    multi_scale_evidence = model_config.get('multi_scale_evidence', False)
    
    # Channel configuration with width multiplier
    channels_config = model_config.get('channels', {})
    channels = {
        'stem': int(channels_config.get('stem', 64) * width_mult),
        'local': int(channels_config.get('local', 160) * width_mult),
        'global': int(channels_config.get('global', 192) * width_mult),
        'decoder': int(channels_config.get('decoder', 192) * width_mult)
    }
    
    # Build model
    model = AURANet(
        num_classes=num_classes,
        img_size=img_size,
        channels=channels,
        top_k=top_k,
        use_topk_inspector=use_topk_inspector,
        multi_scale_evidence=multi_scale_evidence
    )
    
    return model
