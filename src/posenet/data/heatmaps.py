"""Vectorized Gaussian heatmap targets (TensorFlow ops)."""

from __future__ import annotations

import tensorflow as tf


def cell_scale(heatmap_len: int, stride: int, udp: bool) -> float:
    """Input pixels per heatmap cell along one axis.

    ``center``: cell ``j`` sits at continuous input coord ``(j + 0.5) * stride``.
    ``udp``: cell ``j`` sits at pixel index ``j * (N_in - 1) / (N_hm - 1)``, i.e. the
    continuous coord ``j * scale + 0.5``, so corner cells align with corner pixels.
    """
    if not udp:
        return float(stride)
    return (heatmap_len * stride - 1.0) / (heatmap_len - 1.0)


def to_cells(coords, heatmap_hw: tuple[int, int], stride: int, udp: bool):
    """Continuous input coords (..., 2) → heatmap cell coords (cell centre = integer)."""
    h, w = heatmap_hw
    if not udp:
        return coords / float(stride) - 0.5
    sx, sy = cell_scale(w, stride, True), cell_scale(h, stride, True)
    return (coords - 0.5) / tf.constant([sx, sy], coords.dtype)


def render_heatmaps(
    keypoints: tf.Tensor,
    weights: tf.Tensor,
    heatmap_hw: int | tuple[int, int],
    stride: int,
    sigma: float,
    udp: bool = False,
) -> tf.Tensor:
    """Render one Gaussian per keypoint.

    Args:
        keypoints: (B, K, 2) continuous input-image coordinates.
        weights: (B, K) 1 for valid keypoints, 0 otherwise.
        heatmap_hw: output (H, W), or a single side length for square heatmaps.
        stride: input pixels per heatmap cell.
        sigma: Gaussian std in heatmap cells.
        udp: use the UDP cell convention (see ``cell_scale``).

    Returns:
        (B, H, W, K) float32 heatmaps with peak 1 at each valid keypoint.
    """
    h, w = (heatmap_hw, heatmap_hw) if isinstance(heatmap_hw, int) else heatmap_hw
    centers = to_cells(tf.cast(keypoints, tf.float32), (h, w), stride, udp)
    inv = -1.0 / (2.0 * sigma * sigma)
    gx = tf.exp(
        inv * tf.square(tf.range(w, dtype=tf.float32)[None, None, :] - centers[..., 0:1])
    )  # (B, K, W)
    gy = tf.exp(
        inv * tf.square(tf.range(h, dtype=tf.float32)[None, None, :] - centers[..., 1:2])
    )  # (B, K, H)
    heat = gy[..., :, None] * gx[..., None, :]  # (B, K, H, W)
    heat = heat * tf.cast(weights, tf.float32)[..., None, None]
    return tf.transpose(heat, [0, 2, 3, 1])
