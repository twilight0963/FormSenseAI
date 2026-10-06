"""Visibility-weighted heatmap MSE."""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="posenet")
class HeatmapLoss(keras.losses.Loss):
    """MSE over heatmaps, with unlabeled / out-of-frame keypoints masked out.

    ``y_true`` packs ``[heatmaps (K), weights (K)]`` on the channel axis.
    """

    def __init__(self, num_keypoints: int, name: str = "heatmap_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_keypoints = num_keypoints

    def call(self, y_true, y_pred):
        k = self.num_keypoints
        target, weights = y_true[..., :k], y_true[..., k:]
        y_pred = ops.cast(y_pred, target.dtype)
        return ops.mean(weights * ops.square(y_pred - target), axis=(1, 2, 3))

    def get_config(self):
        return {**super().get_config(), "num_keypoints": self.num_keypoints}
