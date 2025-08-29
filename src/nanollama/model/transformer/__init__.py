# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Transformer implementation.

@ 2025, Ambroise Odonnat
"""

from .architecture import Transformer, TransformerConfig
from .inference import InferenceContext, generate, prefill, pretrain
