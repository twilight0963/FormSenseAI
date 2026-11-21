"""Exercise classifier: a small temporal CNN over a window of keypoint features.

A single frame can't tell a squat from standing, so the model looks at ``window`` frames
(48 at 30 fps ≈ 1.6 s). Each frame is described by ``skeleton.features``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from formcheck import skeleton


@dataclass
class ClassifierConfig:
    classes: list[str] = field(default_factory=list)
    window: int = 48  # frames per window
    fps: float = 30.0  # frame rate the window is resampled to
    num_features: int = skeleton.NUM_FEATURES
    score_threshold: float = skeleton.DEFAULT_THRESHOLD
    unknown_below: float = 0.5  # max probability under this → "unknown"
    width: int = 128
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.3

    @property
    def seconds(self) -> float:
        return self.window / self.fps


def build_model(cfg: ClassifierConfig):
    import keras
    from keras import layers

    inp = keras.Input((cfg.window, cfg.num_features), name="features")
    x = layers.Conv1D(cfg.width, 5, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for d in cfg.dilations:
        y = layers.Conv1D(cfg.width, 3, padding="same", dilation_rate=d, use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.SpatialDropout1D(0.1)(y)
        x = layers.Add()([x, y])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(len(cfg.classes), activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name="exercise_classifier")


def save(run_dir: Path, model, cfg: ClassifierConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model.keras")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))


def load_config(run_dir: Path) -> ClassifierConfig:
    data = json.loads((Path(run_dir) / "config.json").read_text())
    data["dilations"] = tuple(data.get("dilations", (1, 2, 4, 8)))
    return ClassifierConfig(**data)


class ExerciseClassifier:
    """Runtime wrapper: recent poses (with timestamps) → class probabilities."""

    def __init__(self, run_dir: str | Path):
        import keras

        run_dir = Path(run_dir)
        self.cfg = load_config(run_dir)
        self.model = keras.models.load_model(run_dir / "model.keras", compile=False)
        self.classes = list(self.cfg.classes)

    def window_features(self, kps: np.ndarray, scores: np.ndarray, times: np.ndarray) -> np.ndarray | None:
        """Resample the last ``cfg.seconds`` of a pose history to the model's frame rate."""
        if len(times) < 2 or times[-1] - times[0] < 0.5 * self.cfg.seconds:
            return None
        dst = times[-1] - (np.arange(self.cfg.window)[::-1]) / self.cfg.fps
        dst = np.clip(dst, times[0], times[-1])  # too-short history repeats the first frame
        k, s = skeleton.resample(kps, scores, np.asarray(times, np.float64), dst)
        return skeleton.features(k, s, self.cfg.score_threshold)

    def probabilities(self, kps: np.ndarray, scores: np.ndarray, times: np.ndarray) -> np.ndarray | None:
        feats = self.window_features(kps, scores, times)
        if feats is None:
            return None
        return np.asarray(self.model(feats[None], training=False))[0]
