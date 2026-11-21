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


def _label(name: str | None) -> str:
    return name.replace("_", " ").capitalize() if name else "Detecting exercise..."


def draw_feedback(image_bgr: np.ndarray, pose: Pose, feedback, threshold: float = 0.3) -> np.ndarray:
    """Exercise, rep count and form cues in a panel; bones a cue is about drawn in red.

    ``feedback`` is a ``formcheck.coach.Feedback``.
    """
    from formcheck.skeleton import ANGLE_JOINTS

    out = image_bgr.copy()
    h, w = out.shape[:2]
    thickness = max(3, round(max(h, w) / 200))
    pts = [tuple(int(round(v)) for v in p) for p in pose.keypoints]
    for name in feedback.flagged_angles:
        if name not in ANGLE_JOINTS:
            continue
        a, b, c = ANGLE_JOINTS[name]
        for i, j in ((a, b), (b, c)):
            if pose.scores[i] >= threshold and pose.scores[j] >= threshold:
                cv2.line(out, pts[i], pts[j], (0, 0, 255), thickness, cv2.LINE_AA)
        if pose.scores[b] >= threshold:
            cv2.circle(out, pts[b], thickness * 2, (0, 0, 255), -1, cv2.LINE_AA)

    scale = max(h, w) / 1400
    font, line_h, pad = cv2.FONT_HERSHEY_SIMPLEX, int(42 * scale) + 4, int(14 * scale) + 4
    head = _label(feedback.exercise)
    if feedback.exercise:
        head += f"  {feedback.confidence:.0%}   reps: {feedback.reps}"
    lines = [(head, (255, 255, 255))] + [(c.text, (80, 80, 255)) for c in feedback.cues]
    widths = [cv2.getTextSize(t, font, scale, max(1, round(2 * scale)))[0][0] for t, _ in lines]
    box_w, box_h = max(widths) + 2 * pad, line_h * len(lines) + pad
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (min(box_w, w), min(box_h, h)), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    for i, (text, color) in enumerate(lines):
        y = pad + line_h * (i + 1) - int(line_h * 0.3)
        cv2.putText(out, text, (pad, y), font, scale, color, max(1, round(2 * scale)), cv2.LINE_AA)
    return out
