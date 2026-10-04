# FusedAddRelu Teaching Task

Inputs x and y must match in shape, dtype and NPU device; broadcasting is not supported. Inputs are finite values.
The result is `max(float32(x) + float32(y), 0)`, finally cast back to the input dtype.

This task is for interface and precision checking; no performance baseline or fabricated speedups are pre-filled.
In actual use, the task files stay read-only; the scope of modification is only the submitted implementation.
