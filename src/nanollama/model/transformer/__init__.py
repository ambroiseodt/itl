# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Transformer implementation.

Copyright (c) 2025 by the authors
"""

from .architecture import Transformer, TransformerConfig
from .inference import InferenceContext, generate, prefill, pretrain
