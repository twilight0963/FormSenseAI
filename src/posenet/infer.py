"""Run pose estimation on images and save skeleton overlays + keypoint JSON.

Example:
    python -m posenet.infer --model runs/<run>/best.keras --image photo.jpg [more.jpg ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from posenet.inference import Pose, PoseEstimator
from posenet.keypoints import KEYPOINT_NAMES, SKELETON


def draw_pose(image_bgr: np.ndarray, pose: Pose, threshold: float = 0.3) -> np.ndarray:
    out = image_bgr.copy()
    thickness = max(2, round(max(out.shape[:2]) / 300))
    pts = [tuple(int(round(v)) for v in p) for p in pose.keypoints]
    for a, b in SKELETON:
        if pose.scores[a] >= threshold and pose.scores[b] >= threshold:
            cv2.line(out, pts[a], pts[b], (0, 255, 0), thickness, cv2.LINE_AA)
    for p, s in zip(pts, pose.scores, strict=True):
        if s >= threshold:
            cv2.circle(out, p, thickness + 2, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose estimation on images.")
    parser.add_argument("--model", required=True, help="path to a trained .keras model")
    parser.add_argument("--image", nargs="+", required=True)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--no-flip", action="store_true", help="disable flip-test averaging")
    args = parser.parse_args()

    estimator = PoseEstimator.from_path(
        args.model, flip_test=not args.no_flip, score_threshold=args.threshold
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in map(Path, args.image):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip: cannot read {path}")
            continue
        pose = estimator.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cv2.imwrite(str(out_dir / f"{path.stem}_pose.png"), draw_pose(bgr, pose, args.threshold))
        result = {
            name: {"x": float(x), "y": float(y), "score": float(s)}
            for name, (x, y), s in zip(KEYPOINT_NAMES, pose.keypoints, pose.scores, strict=True)
        }
        (out_dir / f"{path.stem}_pose.json").write_text(json.dumps(result, indent=2))
        print(f"{path} → {out_dir / (path.stem + '_pose.png')} (mean score {pose.score:.2f})")


if __name__ == "__main__":
    main()
