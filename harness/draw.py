"""Stick-figure overlay (``draw_pose`` and ``SKELETON`` copied from the ``training`` branch)."""

from __future__ import annotations

import cv2
import numpy as np

from harness.pose import Pose

# 0-based limb connections for drawing.
SKELETON = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


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
