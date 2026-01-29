"""
Classification Heads for Multi-Label Image Classification

Available heads:
- MLDecoder: Transformer-based decoder with query embeddings
- CSRA: Class-Specific Residual Attention
- Q2L: Query2Label with label correlation modeling
"""

from .ml_decoder import MLDecoder
from .csra import CSRA
from .q2l import Q2L

__all__ = ["MLDecoder", "CSRA", "Q2L"]
