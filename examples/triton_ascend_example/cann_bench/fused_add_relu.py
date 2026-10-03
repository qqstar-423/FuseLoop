"""Elementwise add and ReLU in one Triton-Ascend kernel.

Inputs have identical shapes and one shared NPU device/dtype. Broadcasting is
deliberately outside this example's interface. No input-dependent data is cached.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_relu_kernel(x_ptr, y_ptr, output_ptr, n_elements,
                           BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = tl.maximum(x + y, 0.0)
    tl.store(output_ptr + offsets, result, mask=mask)


def fused_add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return max(float32(x) + float32(y), 0), cast to the input dtype."""
    if x.device.type != "npu" or y.device.type != "npu":
        raise ValueError("fused_add_relu requires Ascend NPU inputs")
    if x.device != y.device or x.dtype != y.dtype or x.shape != y.shape:
        raise ValueError("inputs must have the same device, dtype and shape")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("supported dtypes: float16, bfloat16, float32")
    # Layout copies, when required, belong to this call and its measured cost.
    x = x.contiguous()
    y = y.contiguous()
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n_elements = x.numel()
    if n_elements:
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _fused_add_relu_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    return output
