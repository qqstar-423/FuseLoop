"""Real NPU/JIT smoke; invoke explicitly on a compatible Linux Ascend machine."""

import argparse
import importlib.metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    import torch
    import torch_npu
    from triton.runtime import driver
    from cann_bench import fused_add_relu

    torch_npu.npu.set_device(args.device_id)
    target = driver.active.get_current_target()
    if target.backend != "npu":
        raise RuntimeError(f"expected Triton-Ascend NPU backend, got {target.backend}")
    print("triton-ascend:", importlib.metadata.version("triton-ascend"), "target:", target)
    device = f"npu:{args.device_id}"
    checked = 0

    def check(x, y):
        nonlocal checked
        output = fused_add_relu(x, y)
        torch_npu.npu.synchronize()
        expected = (x.float() + y.float()).clamp_min(0).to(x.dtype)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert output.shape == x.shape and output.device == x.device and output.dtype == x.dtype
        if x.numel():
            assert output.data_ptr() not in (x.data_ptr(), y.data_ptr())
        checked += 1
        return output

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for shape in ((0,), (), (1,), (1023,), (1024,), (1025,), (4099,), (17, 63)):
            # Identical shapes with different data must never reuse old results.
            for _ in range(2):
                x = torch.randn(shape, dtype=dtype, device=device)
                y = torch.randn(shape, dtype=dtype, device=device)
                check(x, y)
        x = torch.randn((17, 63), dtype=dtype, device=device).transpose(0, 1)
        y = torch.randn((17, 63), dtype=dtype, device=device).transpose(0, 1)
        check(x, y)
        x = torch.full((1025,), -2, dtype=dtype, device=device)
        y = torch.full_like(x, 1)
        first = check(x, y).clone()
        x.fill_(2)  # Same allocation and shape; content changed between calls.
        second = check(x, y)
        torch.testing.assert_close(first, torch.zeros_like(first), rtol=0, atol=0)
        torch.testing.assert_close(second, torch.full_like(second, 3), rtol=0, atol=0)
    print(f"PASS: {checked} actual NPU invocations; no performance claim")


if __name__ == "__main__":
    main()
