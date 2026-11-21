"""Train the exercise classifier on keypoint sequences from ``formcheck.extract``.

Usage:
    python -m formcheck.train --data data/keypoints/penn [--epochs 40]

Writes ``runs/classifier-<timestamp>/`` with model.keras, config.json and report.txt
(window- and video-level accuracy plus a confusion matrix on the test split).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402

from formcheck import skeleton  # noqa: E402
from formcheck.classifier import ClassifierConfig, build_model, save  # noqa: E402


def load_videos(data: Path) -> list[dict]:
    videos = []
    for f in sorted(data.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        videos.append(
            {
                "name": f.stem,
                "kps": z["keypoints"].astype(np.float32),
                "scores": z["scores"].astype(np.float32),
                "action": str(z["action"]),
                "split": str(z["split"]),
                "fps": float(z["fps"]),
            }
        )
    return videos


def sample_window(
    video: dict, cfg: ClassifierConfig, rng: np.random.Generator | None, start: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a window resampled to ``cfg.fps``. With ``rng``, apply random time-stretch, start,
    mirroring, rotation, jitter and joint dropout; otherwise a plain window at ``start``."""
    kps, scores = video["kps"], video["scores"]
    n = len(kps)
    stretch = rng.uniform(0.7, 1.3) if rng is not None else 1.0
    span = (cfg.window - 1) * stretch * video["fps"] / cfg.fps  # in source frames
    if start is None:
        start = rng.uniform(0, max(n - 1 - span, 0)) if rng is not None else 0.0
    pos = np.clip(start + np.linspace(0, span, cfg.window), 0, n - 1)
    k, s = skeleton.resample(kps, scores, np.arange(n, dtype=np.float64), pos)

    if rng is not None:
        if rng.random() < 0.5:
            k, s = skeleton.mirror(k, s)
        vis = s >= cfg.score_threshold
        if vis.any():
            pts = k[vis]
            center = pts.mean(axis=0)
            extent = max(float(np.ptp(pts[:, 1])), float(np.ptp(pts[:, 0])), 1.0)
            theta = np.radians(rng.normal(0, 6))  # camera tilt
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            k = (k - center) @ rot.T * rng.uniform(0.9, 1.1, size=2) + center  # also aspect jitter
            k = k + rng.normal(0, 0.01 * extent, size=k.shape)
        drop = rng.random(s.shape[1]) < 0.06  # whole joint missing (occlusion, out of frame)
        s = np.where(drop[None], 0.0, s)
        frame_drop = rng.random(s.shape[0]) < 0.03
        s = np.where(frame_drop[:, None], 0.0, s)
    return k.astype(np.float32), s.astype(np.float32)


class WindowSequence(keras.utils.PyDataset):
    """Class-balanced random windows from the train split."""

    def __init__(self, videos, cfg: ClassifierConfig, batch_size: int, steps: int, seed: int):
        super().__init__()
        self.cfg, self.batch_size, self.steps = cfg, batch_size, steps
        self.rng = np.random.default_rng(seed)
        self.by_class = [[v for v in videos if v["action"] == c] for c in cfg.classes]

    def __len__(self):
        return self.steps

    def __getitem__(self, _):
        xs, ys = [], []
        for _ in range(self.batch_size):
            c = int(self.rng.integers(len(self.by_class)))
            v = self.by_class[c][int(self.rng.integers(len(self.by_class[c])))]
            k, s = sample_window(v, self.cfg, self.rng)
            xs.append(skeleton.features(k, s, self.cfg.score_threshold))
            ys.append(c)
        return np.stack(xs), np.array(ys)


def eval_windows(videos: list[dict], cfg: ClassifierConfig, stride: int = 8):
    """Every window (no augmentation) of every video, plus the video index of each."""
    xs, ys, owner = [], [], []
    for i, v in enumerate(videos):
        n = len(v["kps"])
        span = (cfg.window - 1) * v["fps"] / cfg.fps
        starts = np.arange(0, max(n - span, 1), stride * v["fps"] / cfg.fps)
        for st in starts:
            k, s = sample_window(v, cfg, None, start=float(st))
            xs.append(skeleton.features(k, s, cfg.score_threshold))
            ys.append(cfg.classes.index(v["action"]))
            owner.append(i)
    return np.stack(xs), np.array(ys), np.array(owner)


def report(model, videos: list[dict], cfg: ClassifierConfig, name: str) -> tuple[str, np.ndarray, np.ndarray]:
    x, y, owner = eval_windows(videos, cfg)
    probs = model.predict(x, batch_size=256, verbose=0)
    pred = probs.argmax(axis=1)
    win_acc = float((pred == y).mean())
    vid_true, vid_pred = [], []
    for i in np.unique(owner):
        m = owner == i
        vid_true.append(y[m][0])
        vid_pred.append(int(probs[m].mean(axis=0).argmax()))
    vid_true, vid_pred = np.array(vid_true), np.array(vid_pred)
    vid_acc = float((vid_true == vid_pred).mean())

    n = len(cfg.classes)
    conf = np.zeros((n, n), int)
    for t, p in zip(vid_true, vid_pred, strict=True):
        conf[t, p] += 1
    w = max(len(c) for c in cfg.classes)
    lines = [
        f"[{name}] {len(videos)} videos, {len(y)} windows",
        f"window accuracy {win_acc:.3f}   video accuracy {vid_acc:.3f}",
        "video confusion (rows = true, cols = predicted):",
        " " * (w + 1) + " ".join(f"{c[:6]:>6}" for c in cfg.classes),
    ]
    for i, c in enumerate(cfg.classes):
        lines.append(
            f"{c:>{w}} "
            + " ".join(f"{v:>6}" for v in conf[i])
            + f"   recall {conf[i, i] / max(conf[i].sum(), 1):.2f}"
        )
    return "\n".join(lines), probs.max(axis=1), pred == y


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/keypoints/penn"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=200, help="batches per epoch")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)
    videos = load_videos(args.data)
    classes = sorted({v["action"] for v in videos})
    cfg = ClassifierConfig(classes=classes)
    splits = {s: [v for v in videos if v["split"] == s] for s in ("train", "val", "test")}
    print({s: len(v) for s, v in splits.items()}, classes)

    x_val, y_val, _ = eval_windows(splits["val"], cfg)
    model = build_model(cfg)
    total = args.epochs * args.steps
    model.compile(
        optimizer=keras.optimizers.AdamW(
            keras.optimizers.schedules.CosineDecay(args.lr, total), weight_decay=1e-4
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary()

    out = args.out or Path("runs") / f"classifier-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    best = out / "best.weights.h5"
    model.fit(
        WindowSequence(splits["train"], cfg, args.batch_size, args.steps, args.seed),
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                best, monitor="val_accuracy", save_best_only=True, save_weights_only=True
            ),
            keras.callbacks.CSVLogger(out / "history.csv"),
        ],
        verbose=2,
    )
    model.load_weights(best)

    # "unknown" threshold: the confidence below which val windows are mostly wrong.
    _, val_conf, val_ok = report(model, splits["val"], cfg, "val")
    wrong = val_conf[~val_ok]
    cfg.unknown_below = float(np.clip(np.percentile(wrong, 75) if len(wrong) else 0.5, 0.4, 0.8))

    texts = [report(model, splits[s], cfg, s)[0] for s in ("val", "test") if splits[s]]
    texts.append(f"unknown_below = {cfg.unknown_below:.2f}")
    text = "\n\n".join(texts)
    print(text)
    (out / "report.txt").write_text(text + "\n")
    save(out, model, cfg)
    best.unlink()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
