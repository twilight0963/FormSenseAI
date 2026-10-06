"""Vectorized Gaussian heatmap targets (TensorFlow ops)."""

from __future__ import annotations

import tensorflow as tf


def render_heatmaps(
    keypoints: tf.Tensor, weights: tf.Tensor, heatmap_size: int, stride: int, sigma: float
) -> tf.Tensor:
    """Render one Gaussian per keypoint.

    Args:
        keypoints: (B, K, 2) continuous input-image coordinates.
        weights: (B, K) 1 for valid keypoints, 0 otherwise.
        heatmap_size: output side length H (= W).
        stride: input pixels per heatmap cell.
        sigma: Gaussian std in heatmap cells.

    Returns:
        (B, H, H, K) float32 heatmaps with peak 1 at each valid keypoint.
    """
    # Cell j covers [j, j+1) in heatmap coords; its centre is j + 0.5.
    centers = tf.cast(keypoints, tf.float32) / float(stride) - 0.5
    grid = tf.range(heatmap_size, dtype=tf.float32)
    inv = -1.0 / (2.0 * sigma * sigma)
    gx = tf.exp(inv * tf.square(grid[None, None, :] - centers[..., 0:1]))  # (B, K, W)
    gy = tf.exp(inv * tf.square(grid[None, None, :] - centers[..., 1:2]))  # (B, K, H)
    heat = gy[..., :, None] * gx[..., None, :]  # (B, K, H, W)
    heat = heat * tf.cast(weights, tf.float32)[..., None, None]
    return tf.transpose(heat, [0, 2, 3, 1])
