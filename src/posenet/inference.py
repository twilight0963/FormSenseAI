"""Model loading and single-image pose prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from posenet.config import Config
from posenet.decode import decode_heatmaps, flip_heatmaps
from posenet.geometry import apply_affine, crop_square, invert_affine
from posenet.keypoints import FLIP_INDEX
from posenet.model.network import build_posenet


def load_config_for(model_path: str | Path) -> Config:
    """Use the run's config.yaml if present (it defines the architecture)."""
    run_cfg = Path(model_path).parent / "config.yaml"
    return Config.from_yaml(run_cfg) if run_cfg.exists() else Config()


def load_inference_model(model_path: str | Path, cfg: Config, dtype: str = "float32") -> keras.Model:
    """Rebuild the network with an inference dtype policy and load trained weights.

    Training may use ``mixed_float16``; rebuilding gives a clean float32 (or fp16)
    graph for export and inference regardless of how the checkpoint was saved.
    """
    trained = keras.models.load_model(model_path, compile=False)
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16" if dtype == "float16" else "float32")
    try:
        model = build_posenet(cfg.model)
    finally:
        keras.mixed_precision.set_global_policy(previous)
    model.set_weights(trained.get_weights())
    return model


@dataclass
class Pose:
    keypoints: np.ndarray  # (K, 2) original-image pixel coords
    scores: np.ndarray  # (K,)

    @property
    def score(self) -> float:
        return float(np.mean(self.scores))


class PoseEstimator:
    """Single-person pose estimator with a two-pass smart crop (no person detector).

    Pass 1 runs on the whole (letterboxed) image. Pass 2 re-crops around the
    confident keypoints from pass 1, matching the framing used in training.
    """

    def __init__(
        self,
        model: keras.Model,
        cfg: Config,
        flip_test: bool = True,
        score_threshold: float = 0.3,
    ):
        self.model = model
        self.cfg = cfg
        self.flip_test = flip_test
        self.score_threshold = score_threshold
        self._predict = tf.function(model, jit_compile=True, reduce_retracing=True)

    @classmethod
    def from_path(cls, model_path: str | Path, **kwargs) -> PoseEstimator:
        cfg = load_config_for(model_path)
        return cls(load_inference_model(model_path, cfg), cfg, **kwargs)

    def heatmaps(self, images: np.ndarray) -> np.ndarray:
        """(N, S, S, 3) RGB uint8/float → (N, H, W, K) heatmaps (flip-averaged if enabled)."""
        x = tf.convert_to_tensor(images, tf.float32)
        if self.flip_test:
            both = self._predict(tf.concat([x, x[:, :, ::-1, :]], axis=0)).numpy()
            n = images.shape[0]
            return 0.5 * (both[:n] + flip_heatmaps(both[n:], FLIP_INDEX))
        return self._predict(x).numpy()

    def _run_crop(self, image: np.ndarray, center: np.ndarray, side: float) -> Pose:
        size = self.cfg.model.input_size
        crop, affine = crop_square(image, center, side, size)
        heat = self.heatmaps(crop[None])
        coords, scores = decode_heatmaps(heat, self.cfg.model.heatmap_stride, self.cfg.train.heatmap_sigma)
        kps = apply_affine(coords[0].astype(np.float64), invert_affine(affine))
        return Pose(kps.astype(np.float32), scores[0])

    def predict(self, image: np.ndarray, bbox: np.ndarray | None = None) -> Pose:
        """Estimate the pose in an RGB image.

        Args:
            bbox: optional COCO ``[x, y, w, h]`` person box; skips the smart crop.
        """
        h, w = image.shape[:2]
        scale = self.cfg.data.bbox_scale
        if bbox is not None:
            x, y, bw, bh = (float(v) for v in bbox)
            return self._run_crop(image, np.array([x + bw / 2, y + bh / 2]), max(bw, bh) * scale)

        pose = self._run_crop(image, np.array([w / 2.0, h / 2.0]), float(max(h, w)))
        confident = pose.scores >= self.score_threshold
        if confident.sum() < 4:
            return pose
        pts = pose.keypoints[confident]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        # Keypoints sit inside the person box (head top / feet extend past them).
        side = float(np.max(hi - lo)) * scale * 1.15
        side = max(side, 0.1 * max(h, w))
        refined = self._run_crop(image, (lo + hi) / 2.0, side)
        return refined if refined.score >= pose.score else pose
