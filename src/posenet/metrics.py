"""Training metric (PCK on heatmaps) and COCO OKS utilities."""

from __future__ import annotations

import keras
import numpy as np
from keras import ops

from posenet.keypoints import OKS_SIGMAS


@keras.saving.register_keras_serializable(package="posenet")
class HeatmapPCK(keras.metrics.Metric):
    """Fraction of valid keypoints whose predicted heatmap peak lies within
    ``threshold * heatmap_size`` cells of the target peak.

    With the default 0.05 on 48×48 heatmaps this is ~2.4 cells ≈ 9.6 input pixels,
    i.e. about 6% of the person box side. It is a fast training signal only;
    the reported benchmark number is COCO OKS AP from ``posenet.evaluate``.
    """

    def __init__(self, num_keypoints: int, threshold: float = 0.05, name: str = "pck", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_keypoints = num_keypoints
        self.threshold = threshold
        self.correct = self.add_variable(shape=(), initializer="zeros", name="correct")
        self.total = self.add_variable(shape=(), initializer="zeros", name="total")

    @staticmethod
    def _argmax_xy(heat):
        shape = ops.shape(heat)
        b, h, w, k = shape[0], shape[1], shape[2], shape[3]
        flat = ops.reshape(ops.transpose(heat, (0, 3, 1, 2)), (b, k, h * w))
        idx = ops.cast(ops.argmax(flat, axis=-1), "int32")
        x = ops.cast(idx % w, "float32")
        y = ops.cast(idx // w, "float32")
        return x, y

    def update_state(self, y_true, y_pred, sample_weight=None):
        k = self.num_keypoints
        target = y_true[..., :k]
        weights = ops.cast(y_true[:, 0, 0, k:], "float32")
        px, py = self._argmax_xy(ops.cast(y_pred, "float32"))
        tx, ty = self._argmax_xy(target)
        dist = ops.sqrt(ops.square(px - tx) + ops.square(py - ty))
        limit = self.threshold * ops.cast(ops.shape(target)[1], "float32")
        hit = ops.cast(dist <= limit, "float32") * weights
        self.correct.assign_add(ops.sum(hit))
        self.total.assign_add(ops.sum(weights))

    def result(self):
        return ops.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self):
        return {
            **super().get_config(),
            "num_keypoints": self.num_keypoints,
            "threshold": self.threshold,
        }


def oks(pred: np.ndarray, gt: np.ndarray, area: float) -> float:
    """Object Keypoint Similarity between (K, 2) predictions and (K, 3) ground truth."""
    visible = gt[:, 2] > 0
    if not np.any(visible):
        return 0.0
    d2 = np.sum(np.square(pred[:, :2] - gt[:, :2]), axis=-1)
    var = np.square(2.0 * OKS_SIGMAS)
    e = d2 / (2.0 * var * (area + np.spacing(1)))
    return float(np.mean(np.exp(-e[visible])))
