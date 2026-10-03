"""A minimal Triton-Ascend submission package for CANN Bench."""

from .fused_add_relu import fused_add_relu

__all__ = ["fused_add_relu"]
