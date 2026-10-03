"""Correctness reference only; never import this from the submitted package."""

import torch


def fused_add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.float() + y.float()).clamp_min(0).to(x.dtype)
