"""Validate the submission contract without importing a real torch/Triton runtime.

Mock launches verify wrapper behavior only. Numerical correctness and JIT require
examples/triton_ascend_example/self_test.py on a real Ascend NPU.
"""

import ast
import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import yaml


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "triton_ascend_example"
KERNEL_SOURCE = EXAMPLE / "cann_bench" / "fused_add_relu.py"


class TensorStub:
    def __init__(self, shape, *, dtype="float32", device=None, token=None, contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or SimpleNamespace(type="npu", index=0)
        self.token = token
        self.is_contiguous = contiguous

    def numel(self):
        return math.prod(self.shape)

    def contiguous(self):
        if self.is_contiguous:
            return self
        return TensorStub(self.shape, dtype=self.dtype, device=self.device, token=self.token)


class KernelStub:
    def __init__(self, function):
        self.function = function
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append({"grid": grid, "args": args, "kwargs": kwargs,
                                  "tokens_at_launch": (args[0].token, args[1].token)})
        return launch


def load_with_runtime_stubs():
    torch = ModuleType("torch")
    torch.Tensor = TensorStub
    torch.float16, torch.bfloat16, torch.float32 = "float16", "bfloat16", "float32"
    torch.empty = lambda shape, dtype, device: TensorStub(shape, dtype=dtype, device=device)
    triton = ModuleType("triton")
    triton.jit = KernelStub
    triton.cdiv = lambda size, block: (size + block - 1) // block
    language = ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language
    spec = importlib.util.spec_from_file_location("_triton_example_under_test", KERNEL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"torch": torch, "triton": triton, "triton.language": language}):
        spec.loader.exec_module(module)
    return module


class TritonExampleTests(unittest.TestCase):
    def setUp(self):
        self.module = load_with_runtime_stubs()

    def test_submission_exports_match_proto_and_golden(self):
        proto = yaml.safe_load((EXAMPLE / "task/proto.yaml").read_text(encoding="utf-8"))
        self.assertEqual(proto["operator"]["schema"],
                         "fused_add_relu(Tensor x, Tensor y) -> Tensor output")
        self.assertEqual([row["name"] for row in proto["operator"]["inputs"]], ["x", "y"])
        for relative in ("cann_bench/fused_add_relu.py", "task/golden.py"):
            tree = ast.parse((EXAMPLE / relative).read_text(encoding="utf-8"))
            wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                           and node.name == "fused_add_relu")
            self.assertEqual([arg.arg for arg in wrapper.args.args], ["x", "y"])
        export = ast.parse((EXAMPLE / "cann_bench/__init__.py").read_text(encoding="utf-8"))
        imports = [node for node in export.body if isinstance(node, ast.ImportFrom)]
        self.assertTrue(any(node.module == "fused_add_relu" and
                            [alias.name for alias in node.names] == ["fused_add_relu"]
                            for node in imports))

    def test_tail_reads_and_writes_are_masked_and_wrapper_has_no_builtin_compute(self):
        tree = ast.parse(KERNEL_SOURCE.read_text(encoding="utf-8"))
        kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_fused_add_relu_kernel")
        accesses = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "tl" and node.func.attr in {"load", "store"}]
        self.assertEqual(len(accesses), 3)
        self.assertTrue(all(any(keyword.arg == "mask" for keyword in node.keywords)
                            for node in accesses))
        wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "fused_add_relu")
        torch_calls = [node.func.attr for node in ast.walk(wrapper) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute)
                       and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"]
        self.assertEqual(torch_calls, ["empty"])

    def test_launch_covers_nondivisible_inputs_without_truncating(self):
        for size, expected_grid in ((1, (1,)), (1024, (1,)), (1025, (2,)), (4099, (5,))):
            with self.subTest(size=size):
                x, y = TensorStub((size,)), TensorStub((size,))
                output = self.module.fused_add_relu(x, y)
                launch = self.module._fused_add_relu_kernel.launches[-1]
                self.assertEqual(launch["grid"], expected_grid)
                self.assertEqual(launch["args"][3], size)
                self.assertIs(launch["args"][2], output)
                self.assertEqual(output.shape, x.shape)
                self.assertEqual(output.device, x.device)

    def test_empty_input_allocates_empty_output_without_launch(self):
        x, y = TensorStub((0,)), TensorStub((0,))
        output = self.module.fused_add_relu(x, y)
        self.assertEqual(output.shape, (0,))
        self.assertEqual(self.module._fused_add_relu_kernel.launches, [])

    def test_repeated_calls_read_current_inputs_and_allocate_new_outputs(self):
        x, y = TensorStub((1025,), token="first-x"), TensorStub((1025,), token="first-y")
        first = self.module.fused_add_relu(x, y)
        x.token = "modified-in-place"
        second = self.module.fused_add_relu(x, y)
        other_x, other_y = TensorStub((1025,), token="new-x"), TensorStub((1025,), token="new-y")
        third = self.module.fused_add_relu(other_x, other_y)
        launches = self.module._fused_add_relu_kernel.launches
        self.assertEqual([row["tokens_at_launch"] for row in launches],
                         [("first-x", "first-y"), ("modified-in-place", "first-y"), ("new-x", "new-y")])
        self.assertIs(launches[1]["args"][0], x)
        self.assertIs(launches[2]["args"][0], other_x)
        self.assertEqual(len({id(first), id(second), id(third), id(x), id(y)}), 5)

    def test_noncontiguous_inputs_are_materialized_for_each_call(self):
        x = TensorStub((63, 17), token="strided-x", contiguous=False)
        y = TensorStub((63, 17), token="strided-y", contiguous=False)
        self.module.fused_add_relu(x, y)
        launch = self.module._fused_add_relu_kernel.launches[-1]
        self.assertIsNot(launch["args"][0], x)
        self.assertIsNot(launch["args"][1], y)
        self.assertEqual(launch["tokens_at_launch"], ("strided-x", "strided-y"))

    def test_invalid_contract_is_rejected_before_launch(self):
        cpu = SimpleNamespace(type="cpu", index=0)
        other_device = SimpleNamespace(type="npu", index=1)
        pairs = [
            (TensorStub((2,), device=cpu), TensorStub((2,), device=cpu), ValueError),
            (TensorStub((2,)), TensorStub((2,), device=other_device), ValueError),
            (TensorStub((2,)), TensorStub((1,)), ValueError),
            (TensorStub((2,)), TensorStub((2,), dtype="float16"), ValueError),
            (TensorStub((2,), dtype="int32"), TensorStub((2,), dtype="int32"), TypeError),
        ]
        for x, y, error in pairs:
            with self.subTest(x=x.dtype, y=y.dtype, shape=y.shape):
                with self.assertRaises(error):
                    self.module.fused_add_relu(x, y)
        self.assertEqual(self.module._fused_add_relu_kernel.launches, [])


if __name__ == "__main__":
    unittest.main()
