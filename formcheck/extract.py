"""Run the Stage 1 pose model over Penn Action gym videos → per-video keypoint sequences.

The classifier trains on what ViTPose predicts (not Penn's hand labels), so training input
looks like what the harness produces at runtime. Each frame is cropped with Penn's labelled
person box, so no person search is needed.

Splits follow the ``vitpose`` branch (``src/posenet/data/exercise.py``): Penn's official
train/test, with val carved out of train by a stable hash of the video id. Note the pose
model was fine-tuned on Penn train/val frames, so its keypoints are cleaner there than on
test or live video; test is the honest number.

Usage:
    python -m formcheck.extract --out data/keypoints/penn [--limit 5]

Output: one ``<seq>.npz`` per video with keypoints (T, 17, 2), scores (T, 17), action,
split, fps. Existing files are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PENN_GYM_ACTIONS = ("bench_press", "clean_and_jerk", "jumping_jacks", "pullup", "pushup", "situp", "squat")
PENN_FPS = 30.0  # Penn Action clips are 30 fps video


def hash_fraction(key: str) -> float:
    """Deterministic value in [0, 1) per key (same as the vitpose branch's val split)."""
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 2**32


def penn_videos(root: Path, actions=PENN_GYM_ACTIONS, val_fraction: float = 0.1):
    """Yield (seq, action, split, boxes (T, 4) [x, y, w, h] or NaN rows) for Penn gym videos."""
    from scipy.io import loadmat

    for label in sorted((root / "labels").glob("*.mat")):
        m = loadmat(label)
        action = str(m["action"][0])
        if action not in actions:
            continue
        seq = label.stem
        n = int(m["nframes"].ravel()[0])
        x, y = m["x"].astype(np.float64), m["y"].astype(np.float64)
        boxes = np.full((n, 4), np.nan)
        raw = m["bbox"] if "bbox" in m else np.zeros((0, 4))
        for t in range(n):
            if t < len(raw) and raw[t][2] > raw[t][0] and raw[t][3] > raw[t][1]:
                x1, y1, x2, y2 = raw[t]
                boxes[t] = [x1, y1, x2 - x1, y2 - y1]
            else:  # fall back to the labelled joints (all 13, including guessed ones)
                pts = np.stack([x[t], y[t]], axis=-1)
                pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
                if len(pts) >= 2:
                    lo, hi = pts.min(axis=0), pts.max(axis=0)
                    size = np.maximum(hi - lo, 1.0) * 1.15
                    boxes[t] = [*((lo + hi) / 2 - size / 2), *size]
        official_train = int(np.asarray(m["train"]).ravel()[0]) == 1
        split = ("val" if hash_fraction(seq) < val_fraction else "train") if official_train else "test"
        yield seq, action, split, boxes


def fill_boxes(boxes: np.ndarray) -> np.ndarray | None:
    """Fill NaN boxes from the nearest labelled frame; None if no frame has a box."""
    ok = np.isfinite(boxes).all(axis=1)
    if not ok.any():
        return None
    idx = np.where(ok)[0]
    nearest = idx[np.abs(np.arange(len(boxes))[:, None] - idx[None]).argmin(axis=1)]
    return boxes[nearest]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("data/penn_action/Penn_Action"))
    parser.add_argument("--model", type=Path, default=Path("runs/20261011-104645/export/saved_model"))
    parser.add_argument("--out", type=Path, default=Path("data/keypoints/penn"))
    parser.add_argument("--limit", type=int, default=0, help="stop after this many videos (0 = all)")
    parser.add_argument("--no-flip", action="store_true", help="disable flip-test averaging")
    args = parser.parse_args()

    from harness.pose import PoseEstimator

    args.out.mkdir(parents=True, exist_ok=True)
    videos = list(penn_videos(args.root))
    if args.limit:
        videos = videos[: args.limit]
    todo = [v for v in videos if not (args.out / f"{v[0]}.npz").exists()]
    print(f"{len(videos)} gym videos, {len(videos) - len(todo)} already done")
    if not todo:
        return
    estimator = PoseEstimator(args.model, flip_test=not args.no_flip)
    print(f"pose model: {estimator.backend.label}")

    for seq, action, split, boxes in tqdm(todo, unit="video"):
        boxes = fill_boxes(boxes)
        if boxes is None:
            continue
        kps, scores = [], []
        for t, box in enumerate(boxes):
            bgr = cv2.imread(str(args.root / "frames" / seq / f"{t + 1:06d}.jpg"))
            if bgr is None:
                break
            pose = estimator.predict_in_box(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), box)
            kps.append(pose.keypoints)
            scores.append(pose.scores)
        if len(kps) < 8:
            continue
        tmp = args.out / f"{seq}.tmp.npz"
        np.savez_compressed(
            tmp, keypoints=np.stack(kps), scores=np.stack(scores), action=action, split=split, fps=PENN_FPS
        )
        tmp.rename(args.out / f"{seq}.npz")


if __name__ == "__main__":
    main()
