# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Transformer implementation.

@ 2025, Meta
"""

from .architecture import Transformer, TransformerConfig
from .inference import build_pretrain_mask, generate, prefill, pretrain
