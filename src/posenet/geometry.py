"""Square-crop affine warps shared by preprocessing, evaluation and inference.

Coordinate convention: continuous pixel coordinates, where pixel ``i`` spans
``[i, i + 1)`` and its centre is at ``i + 0.5``. With this convention a
horizontal flip is exactly ``x -> W - x`` and a stride-``s`` downsample is
exactly ``x -> x / s``, so no half-pixel corrections leak into training
targets. OpenCV uses pixel-centre-at-integer coordinates, so matrices passed
to ``cv2.warpAffine`` are converted with ``_to_index_space``.
"""

from __future__ import annotations

import cv2
import numpy as np


def square_crop_affine(center: np.ndarray, side: float, out_size: int) -> np.ndarray:
    """2x3 affine (continuous coords) mapping a square region onto ``out_size``²."""
    k = out_size / side
    cx, cy = float(center[0]), float(center[1])
    return np.array(
        [[k, 0.0, out_size / 2.0 - k * cx], [0.0, k, out_size / 2.0 - k * cy]],
        dtype=np.float64,
    )


def invert_affine(m: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine matrix."""
    full = np.vstack([m, [0.0, 0.0, 1.0]])
    return np.linalg.inv(full)[:2]


def apply_affine(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to ``(..., 2)`` points."""
    return points @ m[:, :2].T + m[:, 2]


def _to_index_space(m: np.ndarray) -> np.ndarray:
    """Convert a continuous-coordinate affine to OpenCV's index coordinates."""
    out = m.copy()
    out[:, 2] = m[:, 2] + 0.5 * (m[0:2, 0] + m[0:2, 1]) - 0.5
    return out


def crop_square(
    image: np.ndarray, center: np.ndarray, side: float, out_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a square region around ``center`` and resample it to ``out_size``².

    Areas outside the image are zero-filled. When downscaling, the source patch
    is Gaussian pre-filtered to avoid aliasing.

    Returns:
        (crop, affine) where ``affine`` maps original continuous coords to crop
        continuous coords.
    """
    affine = square_crop_affine(center, side, out_size)
    k = out_size / side
    h, w = image.shape[:2]

    # Only touch the source pixels the crop needs (plus a filter margin).
    margin = 2 + int(np.ceil(3.0 / k))
    half = side / 2.0
    x0 = max(int(np.floor(center[0] - half)) - margin, 0)
    y0 = max(int(np.floor(center[1] - half)) - margin, 0)
    x1 = min(int(np.ceil(center[0] + half)) + margin, w)
    y1 = min(int(np.ceil(center[1] + half)) + margin, h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((out_size, out_size, image.shape[2]), image.dtype), affine

    patch = image[y0:y1, x0:x1]
    if k < 1.0:
        sigma = (1.0 / k - 1.0) / 2.0
        if sigma > 0.3:
            patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Affine relative to the patch origin (an exact integer shift).
    patch_affine = affine.copy()
    patch_affine[:, 2] += affine[:, :2] @ np.array([x0, y0], dtype=np.float64)
    crop = cv2.warpAffine(
        patch,
        _to_index_space(patch_affine),
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return crop, affine


def bbox_to_square(bbox: np.ndarray, scale: float) -> tuple[np.ndarray, float]:
    """COCO ``[x, y, w, h]`` box → (centre, side) of the enlarged enclosing square."""
    x, y, w, h = (float(v) for v in bbox)
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
    return center, max(w, h) * scale
