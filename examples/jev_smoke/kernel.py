"""Synthetic source-reading fixture, not a compiled or benchmarked NPU kernel.

The function names model two serial stages and an HBM intermediate buffer.
The private smoke marker is amber_orchid_7291.
"""


def kernel_1_conv_to_hbm(x, weight, intermediate_hbm):
    intermediate_hbm[:] = conv2d(x, weight)


def kernel_2_sigmoid_from_hbm(intermediate_hbm, output):
    output[:] = sigmoid(intermediate_hbm)


def fused_conv_sigmoid(x, weight, intermediate_hbm, output):
    kernel_1_conv_to_hbm(x, weight, intermediate_hbm)
    kernel_2_sigmoid_from_hbm(intermediate_hbm, output)
