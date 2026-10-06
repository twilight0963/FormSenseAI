"""Building blocks: Conv-BN-ReLU6, inverted residual, depthwise-separable conv."""

from __future__ import annotations

from keras import layers

_BN = {"momentum": 0.99, "epsilon": 1e-3}


def make_divisible(v: float, divisor: int = 8) -> int:
    """Round channel counts to a multiple of ``divisor`` (hardware-friendly)."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_bn_act(x, filters: int, kernel: int = 3, stride: int = 1, act: bool = True, name: str = ""):
    x = layers.Conv2D(filters, kernel, strides=stride, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(**_BN, name=f"{name}_bn")(x)
    if act:
        x = layers.ReLU(max_value=6.0, name=f"{name}_relu")(x)
    return x


def depthwise_bn_act(x, stride: int = 1, name: str = ""):
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(**_BN, name=f"{name}_dw_bn")(x)
    return layers.ReLU(max_value=6.0, name=f"{name}_dw_relu")(x)


def inverted_residual(x, filters: int, stride: int, expansion: int, name: str):
    """MobileNetV2 block: 1x1 expand → 3x3 depthwise → 1x1 linear projection."""
    in_ch = x.shape[-1]
    shortcut = x
    if expansion != 1:
        x = conv_bn_act(x, in_ch * expansion, kernel=1, name=f"{name}_expand")
    x = depthwise_bn_act(x, stride=stride, name=name)
    x = conv_bn_act(x, filters, kernel=1, act=False, name=f"{name}_project")
    if stride == 1 and in_ch == filters:
        x = layers.Add(name=f"{name}_add")([shortcut, x])
    return x


def separable_block(x, filters: int, name: str):
    """3x3 depthwise + 1x1 pointwise, each followed by BN + ReLU6."""
    x = depthwise_bn_act(x, name=name)
    return conv_bn_act(x, filters, kernel=1, name=f"{name}_pw")
