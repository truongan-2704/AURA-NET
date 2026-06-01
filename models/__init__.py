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

from .variants import (
    AURANet_Nano,
    AURANet_Tiny,
    AURANet_S,
    AURANet_M,
    AURANet_L,
    AURANet_X,
    # Aliases without underscore
    AURANetNano,
    AURANetTiny,
    AURANetS,
    AURANetM,
    AURANetL,
    AURANetX,
    get_model_info
)

__all__ = [
    # Main model (Medium variant)
    'AURANet',

    # Model variants (with underscore)
    'AURANet_Nano',
    'AURANet_Tiny',
    'AURANet_S',
    'AURANet_M',
    'AURANet_L',
    'AURANet_X',

    # Model variants (without underscore - easier to type)
    'AURANetNano',
    'AURANetTiny',
    'AURANetS',
    'AURANetM',
    'AURANetL',
    'AURANetX',

    # Components
    'DualStreamFeatureExtractor',
    'EvidenceProposalNetwork',
    'AdaptiveEvidenceRouter',
    'EvidenceToObjectDecoder',

    # Utils
    'get_model_info',
]

__version__ = '1.0.0'
