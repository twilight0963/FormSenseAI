"""COCO 17-keypoint schema: names, skeleton, flip map and OKS sigmas."""

import numpy as np

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

# Per-keypoint OKS sigmas used by the official COCO evaluation.
OKS_SIGMAS = (
    np.array(
        [
            0.26,
            0.25,
            0.25,
            0.35,
            0.35,
            0.79,
            0.79,
            0.72,
            0.72,
            0.62,
            0.62,
            1.07,
            1.07,
            0.87,
            0.87,
            0.89,
            0.89,
        ],
        dtype=np.float32,
    )
    / 10.0
)
