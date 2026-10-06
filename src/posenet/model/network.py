"""PoseNet: MobileNetV2-style encoder + FPN-style decoder → stride-4 heatmaps.

Fully convolutional by design: keypoint heatmaps keep spatial structure that a
Dense regression head would throw away, and give better accuracy at lower cost.
"""

from __future__ import annotations

import keras
from keras import layers

from posenet.config import ModelConfig
from posenet.keypoints import NUM_KEYPOINTS
from posenet.model.layers import conv_bn_act, inverted_residual, make_divisible, separable_block

# (expansion, channels, repeats, first stride) — MobileNetV2 layout.
_STAGES = (
    (1, 16, 1, 1),
    (6, 24, 2, 2),  # stride 4
    (6, 32, 3, 2),  # stride 8
    (6, 64, 4, 2),
    (6, 96, 3, 1),  # stride 16
    (6, 160, 3, 2),
    (6, 320, 1, 1),  # stride 32
)
# Stage indices whose outputs feed the decoder (strides 4, 8, 16, 32).
_TAPS = (1, 2, 4, 6)


def build_posenet(cfg: ModelConfig, num_keypoints: int = NUM_KEYPOINTS) -> keras.Model:
    """Input: (B, S, S, 3) RGB in [0, 255]. Output: (B, S/4, S/4, K) float32 heatmaps."""
    if cfg.heatmap_stride != 4:
        raise ValueError("The decoder upsamples to stride 4; heatmap_stride must be 4.")
    alpha = cfg.width_multiplier
    dec = cfg.decoder_channels

    inputs = keras.Input((cfg.input_size, cfg.input_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="normalize")(inputs)
    x = conv_bn_act(x, make_divisible(32 * alpha), stride=2, name="stem")

    features = []
    for s, (t, c, n, stride) in enumerate(_STAGES):
        filters = make_divisible(c * alpha)
        for i in range(n):
            x = inverted_residual(x, filters, stride if i == 0 else 1, t, name=f"stage{s}_block{i}")
        if s in _TAPS:
            features.append(x)

    # Top-down pathway: stride 32 → 16 → 8 → 4.
    y = conv_bn_act(features[-1], dec, kernel=1, name="lateral32")
    for f, stride in zip(reversed(features[:-1]), (16, 8, 4), strict=True):
        # float32: under mixed precision, XLA on GPU (TF 2.20) fails to compile
        # the fp16 bilinear-resize gradient ("Seen floating point types of
        # different precisions"). The merge Add casts back to the compute dtype.
        y = layers.UpSampling2D(2, interpolation="bilinear", dtype="float32", name=f"up{stride}")(y)
        lateral = conv_bn_act(f, dec, kernel=1, name=f"lateral{stride}")
        y = layers.Add(name=f"merge{stride}")([y, lateral])
        y = separable_block(y, dec, name=f"refine{stride}")

    y = separable_block(y, dec, name="head")
    heatmaps = layers.Conv2D(num_keypoints, 1, padding="same", dtype="float32", name="heatmaps")(y)
    return keras.Model(inputs, heatmaps, name="posenet")
