"""
AURA-NET Loss Functions
Evidence-Centric Object Detection
"""

from .aura_v3_loss import (
    AURANetLoss,
    EvidenceLoss,
    CIoULoss,
    FocalLoss,
    UncertaintyLoss
)

__all__ = [
    'AURANetLoss',
    'EvidenceLoss',
    'CIoULoss',
    'FocalLoss',
    'UncertaintyLoss',
]

__version__ = '1.0.0'

