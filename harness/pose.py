"""TFLite pose estimation: square crops, heatmap decoding and a two-pass smart crop.

The geometry and decoding code is copied from the ``training`` branch
(``src/posenet/geometry.py``, ``decode.py``, ``keypoints.py``, ``inference.py``)
so the harness runs the exported model without the training package.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Model constants from the training run's config.yaml.
INPUT_SIZE = 192
HEATMAP_STRIDE = 4
HEATMAP_SIGMA = 2.0
BBOX_SCALE = 1.25

KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
NUM_KEYPOINTS = len(KEYPOINT_NAMES)

# Index of the mirrored keypoint after a horizontal flip.
FLIP_INDEX = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


# --- geometry (training: src/posenet/geometry.py) ---


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


# --- decoding (training: src/posenet/decode.py) ---


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


def decode_heatmaps(heat: np.ndarray, stride: int, sigma: float = 2.0, refine: bool = True):
    """Decode (N, H, W, K) heatmaps.

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

    # Heatmap cell centre j sits at continuous coord j + 0.5.
    coords = np.stack([(fx + 0.5) * stride, (fy + 0.5) * stride], axis=-1)
    return coords.astype(np.float32), scores.astype(np.float32)


def flip_heatmaps(heat: np.ndarray, flip_index) -> np.ndarray:
    """Map heatmaps predicted on a mirrored image back to the original frame."""
    return heat[:, :, ::-1, :][..., list(flip_index)]


# --- estimator ---


def _make_interpreter(model_path: str, num_threads: int):
    """Prefer the standalone LiteRT runtime; fall back to TensorFlow's deprecated copy."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Interpreter(model_path=model_path, num_threads=num_threads)


@dataclass
class Pose:
    keypoints: np.ndarray  # (K, 2) original-image pixel coords
    scores: np.ndarray  # (K,)

    @property
    def score(self) -> float:
        return float(np.mean(self.scores))


class PoseEstimator:
    """Single-person pose estimator on an exported TFLite model.

    Without a previous pose it uses the two-pass smart crop from training:
    pass 1 on the whole frame, pass 2 re-cropped around the confident keypoints.
    With a previous pose (video tracking) it crops around that pose directly.
    """

    def __init__(
        self,
        model_path: str | Path,
        flip_test: bool = True,
        score_threshold: float = 0.3,
        num_threads: int = 4,
    ):
        self.flip_test = flip_test
        self.score_threshold = score_threshold
        self._interp = _make_interpreter(str(model_path), num_threads)
        self._interp.allocate_tensors()
        self._input = self._interp.get_input_details()[0]["index"]
        self._output = self._interp.get_output_details()[0]["index"]
        self._invoke(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.float32))  # warm-up

    def _invoke(self, crop: np.ndarray) -> np.ndarray:
        self._interp.set_tensor(self._input, np.ascontiguousarray(crop[None], np.float32))
        self._interp.invoke()
        return self._interp.get_tensor(self._output).copy()

    def heatmaps(self, crop: np.ndarray) -> np.ndarray:
        """(S, S, 3) RGB crop → (1, H, W, K) heatmaps (flip-averaged if enabled)."""
        heat = self._invoke(crop)
        if self.flip_test:
            mirrored = self._invoke(crop[:, ::-1])
            heat = 0.5 * (heat + flip_heatmaps(mirrored, FLIP_INDEX))
        return heat

    def _run_crop(self, image: np.ndarray, center: np.ndarray, side: float) -> Pose:
        crop, affine = crop_square(image, center, side, INPUT_SIZE)
        coords, scores = decode_heatmaps(self.heatmaps(crop), HEATMAP_STRIDE, HEATMAP_SIGMA)
        kps = apply_affine(coords[0].astype(np.float64), invert_affine(affine))
        return Pose(kps.astype(np.float32), scores[0])

    def _refit(self, image: np.ndarray, pose: Pose) -> Pose | None:
        """Re-crop around ``pose``'s confident keypoints; None if too few are confident."""
        confident = pose.scores >= self.score_threshold
        if confident.sum() < 4:
            return None
        h, w = image.shape[:2]
        pts = pose.keypoints[confident]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        # Keypoints sit inside the person box (head top / feet extend past them).
        side = float(np.max(hi - lo)) * BBOX_SCALE * 1.15
        side = max(side, 0.1 * max(h, w))
        return self._run_crop(image, (lo + hi) / 2.0, side)

    def predict(self, image: np.ndarray, prev: Pose | None = None) -> Pose:
        """Estimate the pose in an RGB frame, optionally tracking from the previous frame's pose."""
        if prev is not None:
            tracked = self._refit(image, prev)
            if tracked is not None and tracked.score >= self.score_threshold:
                return tracked

        h, w = image.shape[:2]
        pose = self._run_crop(image, np.array([w / 2.0, h / 2.0]), float(max(h, w)))
        refined = self._refit(image, pose)
        return refined if refined is not None and refined.score >= pose.score else pose
