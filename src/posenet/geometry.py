"""Crop affine warps shared by preprocessing, evaluation and inference.

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


def crop_affine(center: np.ndarray, size: tuple[float, float], out_size: tuple[int, int]) -> np.ndarray:
    """2x3 affine (continuous coords) mapping a ``(w, h)`` region onto ``(out_w, out_h)``.

    The region should have the output's aspect ratio; otherwise the crop is stretched.
    """
    kx, ky = out_size[0] / size[0], out_size[1] / size[1]
    cx, cy = float(center[0]), float(center[1])
    return np.array(
        [[kx, 0.0, out_size[0] / 2.0 - kx * cx], [0.0, ky, out_size[1] / 2.0 - ky * cy]],
        dtype=np.float64,
    )


def square_crop_affine(center: np.ndarray, side: float, out_size: int) -> np.ndarray:
    """2x3 affine (continuous coords) mapping a square region onto ``out_size``²."""
    return crop_affine(center, (side, side), (out_size, out_size))


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
    """Crop a square region around ``center`` and resample it to ``out_size``² (see ``crop_region``)."""
    return crop_region(image, center, (side, side), (out_size, out_size))


def crop_region(
    image: np.ndarray, center: np.ndarray, size: tuple[float, float], out_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a ``(w, h)`` region around ``center`` and resample it to ``(out_w, out_h)``.

    Areas outside the image are zero-filled. When downscaling, the source patch
    is Gaussian pre-filtered to avoid aliasing.

    Returns:
        (crop, affine) where ``affine`` maps original continuous coords to crop
        continuous coords.
    """
    affine = crop_affine(center, size, out_size)
    out_w, out_h = out_size
    k = min(out_w / size[0], out_h / size[1])
    h, w = image.shape[:2]

    # Only touch the source pixels the crop needs (plus a filter margin).
    margin = 2 + int(np.ceil(3.0 / k))
    half_w, half_h = size[0] / 2.0, size[1] / 2.0
    x0 = max(int(np.floor(center[0] - half_w)) - margin, 0)
    y0 = max(int(np.floor(center[1] - half_h)) - margin, 0)
    x1 = min(int(np.ceil(center[0] + half_w)) + margin, w)
    y1 = min(int(np.ceil(center[1] + half_h)) + margin, h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((out_h, out_w, image.shape[2]), image.dtype), affine

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
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return crop, affine


def bbox_to_square(bbox: np.ndarray, scale: float) -> tuple[np.ndarray, float]:
    """COCO ``[x, y, w, h]`` box → (centre, side) of the enlarged enclosing square."""
    center, (side, _) = bbox_to_region(bbox, scale, 1.0)
    return center, side


def bbox_to_region(bbox: np.ndarray, scale: float, aspect: float) -> tuple[np.ndarray, tuple[float, float]]:
    """COCO ``[x, y, w, h]`` box → (centre, (w, h)) of the enlarged enclosing box with
    ``w / h == aspect`` (the shorter side is extended, as in MMPose's ``_box2cs``)."""
    x, y, w, h = (float(v) for v in bbox)
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return center, (w * scale, h * scale)
