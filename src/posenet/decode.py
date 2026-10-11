"""Heatmap → keypoint decoding with DARK-style sub-pixel refinement."""

from __future__ import annotations

import cv2
import numpy as np


def _refine(heat: np.ndarray, xs: np.ndarray, ys: np.ndarray, sigma: float):
    """Second-order Taylor refinement of integer peaks on the log heatmap.

    heat: (N, H, W, K). xs, ys: (N, K) integer peak locations.
    Returns sub-pixel offsets (dx, dy), each (N, K), clipped to [-1, 1].
    """
    n, h, w, k = heat.shape
    smooth = np.empty_like(heat)
    for i in range(n):
        # OpenCV filters support at most 4 channels per call on some builds; loop per keypoint.
        for j in range(k):
            # Zero padding: reflecting the border would pull near-edge peaks outward.
            smooth[i, :, :, j] = cv2.GaussianBlur(
                heat[i, :, :, j], (0, 0), sigma, borderType=cv2.BORDER_CONSTANT
            )
    pad = ((0, 0), (1, 1), (1, 1), (0, 0))
    log_smooth = np.pad(np.log(np.maximum(smooth, 1e-10)), pad, mode="edge")
    log_raw = np.pad(np.log(np.maximum(heat, 1e-10)), pad, mode="edge")

    ni, ki = np.meshgrid(np.arange(n), np.arange(k), indexing="ij")
    px, py = xs + 1, ys + 1  # account for padding
    # The blur window is truncated near the border, which biases the peak, so
    # peaks within ~1.5 sigma of an edge are refined on the unsmoothed heatmap.
    margin = int(np.ceil(1.5 * sigma))
    near_edge = (xs < margin) | (xs >= w - margin) | (ys < margin) | (ys >= h - margin)

    def at(dy, dx):
        return np.where(
            near_edge,
            log_raw[ni, py + dy, px + dx, ki],
            log_smooth[ni, py + dy, px + dx, ki],
        )

    c = at(0, 0)
    gx = 0.5 * (at(0, 1) - at(0, -1))
    gy = 0.5 * (at(1, 0) - at(-1, 0))
    hxx = at(0, 1) - 2.0 * c + at(0, -1)
    hyy = at(1, 0) - 2.0 * c + at(-1, 0)
    hxy = 0.25 * (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1))

    det = hxx * hyy - hxy * hxy
    # Refine only interior peaks with a negative-definite Hessian; keep the integer peak otherwise.
    interior = (xs > 0) & (xs < w - 1) & (ys > 0) & (ys < h - 1)
    ok = interior & (det > 1e-8) & (hxx < 0)
    safe = np.where(ok, det, 1.0)
    dx = np.where(ok, -(hyy * gx - hxy * gy) / safe, 0.0)
    dy = np.where(ok, -(hxx * gy - hxy * gx) / safe, 0.0)
    return np.clip(dx, -1.0, 1.0), np.clip(dy, -1.0, 1.0)


def decode_heatmaps(
    heat: np.ndarray, stride: int, sigma: float = 2.0, refine: bool = True, udp: bool = False
):
    """Decode (N, H, W, K) heatmaps.

    ``udp`` selects the cell convention the model was trained with (see
    ``posenet.data.heatmaps.cell_scale``).

    Returns:
        coords: (N, K, 2) continuous input-image coordinates.
        scores: (N, K) peak heatmap values.
    """
    heat = np.asarray(heat, dtype=np.float32)
    n, h, w, k = heat.shape
    flat = heat.transpose(0, 3, 1, 2).reshape(n, k, h * w)
    idx = np.argmax(flat, axis=-1)
    scores = np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]
    xs, ys = idx % w, idx // w

    fx, fy = xs.astype(np.float32), ys.astype(np.float32)
    if refine:
        dx, dy = _refine(heat, xs, ys, sigma)
        fx, fy = fx + dx, fy + dy

    if udp:
        sx, sy = (w * stride - 1.0) / (w - 1.0), (h * stride - 1.0) / (h - 1.0)
        coords = np.stack([fx * sx + 0.5, fy * sy + 0.5], axis=-1)
    else:
        # Heatmap cell centre j sits at continuous coord j + 0.5.
        coords = np.stack([(fx + 0.5) * stride, (fy + 0.5) * stride], axis=-1)
    return coords.astype(np.float32), scores.astype(np.float32)


def flip_heatmaps(heat: np.ndarray, flip_index) -> np.ndarray:
    """Map heatmaps predicted on a mirrored image back to the original frame."""
    return heat[:, :, ::-1, :][..., list(flip_index)]
