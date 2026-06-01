"""
AURA-NET Models Package
Evidence-Centric Object Detection
"""

from .v3 import (
    AURANet,
    DualStreamFeatureExtractor,
    EvidenceProposalNetwork,
    AdaptiveEvidenceRouter,
    EvidenceToObjectDecoder
)

__all__ = [
    'AURANet',
    'DualStreamFeatureExtractor',
    'EvidenceProposalNetwork',
    'AdaptiveEvidenceRouter',
    'EvidenceToObjectDecoder',
]

__version__ = '1.0.0'
