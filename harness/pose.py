"""Pose estimation on an exported model: crops, heatmap decoding and a two-pass smart crop.

Runs either a TFLite model (CPU) or a SavedModel (GPU when available), e.g. the
ViTPose export from the ``vitpose`` branch. Input size, heatmap stride and the
heatmap cell convention are read from the model (and its run's ``config.yaml``).

The geometry and decoding code is copied from the ``vitpose`` branch
(``src/posenet/geometry.py``, ``decode.py``, ``keypoints.py``, ``inference.py``)
so the harness runs the exported model without the training package.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Constants shared by every model trained with the posenet package.
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


def crop_affine(center: np.ndarray, size: tuple[float, float], out_size: tuple[int, int]) -> np.ndarray:
    """2x3 affine (continuous coords) mapping a ``(w, h)`` region onto ``(out_w, out_h)``."""
    kx, ky = out_size[0] / size[0], out_size[1] / size[1]
    cx, cy = float(center[0]), float(center[1])
    return np.array(
        [[kx, 0.0, out_size[0] / 2.0 - kx * cx], [0.0, ky, out_size[1] / 2.0 - ky * cy]],
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


def bbox_to_region(bbox: np.ndarray, scale: float, aspect: float) -> tuple[np.ndarray, tuple[float, float]]:
    """``[x, y, w, h]`` box → (centre, (w, h)) of the enlarged enclosing box with ``w / h == aspect``."""
    x, y, w, h = (float(v) for v in bbox)
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return center, (w * scale, h * scale)


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


def decode_heatmaps(
    heat: np.ndarray, stride: int, sigma: float = 2.0, refine: bool = True, udp: bool = False
):
    """Decode (N, H, W, K) heatmaps.

    ``udp``: the model was trained with UDP cells (corner cells align with corner
    pixels, as ViTPose does) instead of cell ``j`` at ``(j + 0.5) * stride``.

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


# --- estimator ---


class TFLiteBackend:
    """TFLite interpreter on the CPU; batches run one image at a time."""

    def __init__(self, model_path: Path, num_threads: int):
        try:
            from ai_edge_litert.interpreter import Interpreter  # standalone LiteRT runtime
        except ImportError:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter  # TensorFlow's deprecated copy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
        self._interp.allocate_tensors()
        inp = self._interp.get_input_details()[0]
        self._input = inp["index"]
        self._output = self._interp.get_output_details()[0]["index"]
        self.input_hw = (int(inp["shape"][1]), int(inp["shape"][2]))
        self.label = f"TFLite CPU ({num_threads} threads)"

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        out = []
        for image in batch:
            self._interp.set_tensor(self._input, np.ascontiguousarray(image[None], np.float32))
            self._interp.invoke()
            out.append(self._interp.get_tensor(self._output).copy())
        return np.concatenate(out)


class SavedModelBackend:
    """SavedModel (``model.export()``) on the GPU if TensorFlow sees one, else the CPU."""

    def __init__(self, model_path: Path):
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self._tf = tf
        self._model = tf.saved_model.load(str(model_path))
        shape = self._model.serve.input_signature[0].shape
        self.input_hw = (int(shape[1]), int(shape[2]))
        self.label = "SavedModel GPU" if gpus else "SavedModel CPU"

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        return self._model.serve(self._tf.constant(batch, self._tf.float32)).numpy()


def heatmap_coords_for(model_path: Path) -> str:
    """Read ``model.heatmap_coords`` from the training run's config.yaml, searching a few
    directories up from the export (``<run>/export/saved_model``); "center" if absent."""
    import yaml

    for parent in list(model_path.resolve().parents)[:3]:
        cfg_file = parent / "config.yaml"
        if cfg_file.is_file():
            model_cfg = (yaml.safe_load(cfg_file.read_text()) or {}).get("model", {})
            return str(model_cfg.get("heatmap_coords", "center"))
    return "center"


@dataclass
class Pose:
    keypoints: np.ndarray  # (K, 2) original-image pixel coords
    scores: np.ndarray  # (K,)

    @property
    def score(self) -> float:
        return float(np.mean(self.scores))


class PoseEstimator:
    """Single-person pose estimator on an exported model (TFLite file or SavedModel directory).

    Without a previous pose it searches the frame with input-shaped windows
    (several across a wide frame), keeps the most confident, then re-crops around
    its keypoints. With a previous pose (video tracking) it crops around that pose.
    """

    def __init__(
        self,
        model_path: str | Path,
        flip_test: bool = True,
        score_threshold: float = 0.3,
        num_threads: int = 4,
        heatmap_coords: str = "auto",
    ):
        model_path = Path(model_path)
        self.flip_test = flip_test
        self.score_threshold = score_threshold
        self.backend = (
            SavedModelBackend(model_path) if model_path.is_dir() else TFLiteBackend(model_path, num_threads)
        )
        self.input_hw = self.backend.input_hw
        if heatmap_coords == "auto":
            heatmap_coords = heatmap_coords_for(model_path)
        if heatmap_coords not in ("center", "udp"):
            raise ValueError(f"heatmap_coords must be auto, center or udp, not {heatmap_coords!r}")
        self.udp = heatmap_coords == "udp"
        warm = self.backend(np.zeros((1, *self.input_hw, 3), np.float32))  # also compiles the graph
        self.stride = self.input_hw[0] // warm.shape[1]

    @property
    def aspect(self) -> float:
        """Input width / height."""
        return self.input_hw[1] / self.input_hw[0]

    def heatmaps(self, crop: np.ndarray) -> np.ndarray:
        """(H, W, 3) RGB crop → (1, H/s, W/s, K) heatmaps (flip-averaged if enabled)."""
        if not self.flip_test:
            return self.backend(crop[None])
        both = self.backend(np.stack([crop, crop[:, ::-1]]))
        return 0.5 * (both[:1] + flip_heatmaps(both[1:], FLIP_INDEX))

    def _run_crop(self, image: np.ndarray, center: np.ndarray, size: tuple[float, float]) -> Pose:
        h, w = self.input_hw
        crop, affine = crop_region(image, center, size, (w, h))
        coords, scores = decode_heatmaps(self.heatmaps(crop), self.stride, HEATMAP_SIGMA, udp=self.udp)
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
        size = np.maximum((hi - lo) * 1.15, 0.08 * max(h, w))
        center = (lo + hi) / 2.0
        return self._run_crop(
            image, *bbox_to_region(np.array([*(center - size / 2), *size]), BBOX_SCALE, self.aspect)
        )

    def search_windows(self, h: int, w: int) -> list[tuple[np.ndarray, tuple[float, float]]]:
        """Input-shaped (centre, size) windows covering the frame: one if the frame is no wider
        than the input, otherwise full-height windows spread evenly across it."""
        win_w = h * self.aspect
        if w <= win_w * 1.2:
            return [bbox_to_region(np.array([0.0, 0.0, w, h]), 1.0, self.aspect)]
        n = int(np.ceil(w / win_w))
        xs = np.linspace(win_w / 2.0, w - win_w / 2.0, n)
        return [(np.array([x, h / 2.0]), (win_w, float(h))) for x in xs]

    def predict(self, image: np.ndarray, prev: Pose | None = None) -> Pose:
        """Estimate the pose in an RGB frame, optionally tracking from the previous frame's pose."""
        if prev is not None:
            tracked = self._refit(image, prev)
            if tracked is not None and tracked.score >= self.score_threshold:
                return tracked

        h, w = image.shape[:2]
        pose = max((self._run_crop(image, c, s) for c, s in self.search_windows(h, w)), key=lambda p: p.score)
        refined = self._refit(image, pose)
        return refined if refined is not None and refined.score >= pose.score else pose
