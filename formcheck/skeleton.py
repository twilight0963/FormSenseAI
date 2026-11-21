"""Joint angles and classifier features from COCO-17 keypoint sequences.

All functions work on sequences: ``kps`` is ``(T, 17, 2)`` image pixels (y down) and
``scores`` is ``(T, 17)``. A single frame is just ``T == 1``.
"""

from __future__ import annotations

import warnings

import numpy as np

from harness.pose import FLIP_INDEX, KEYPOINT_NAMES, NUM_KEYPOINTS

KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Angle at the middle joint of each (a, b, c) triple, for both sides.
ANGLE_JOINTS: dict[str, tuple[int, int, int]] = {}
for _side in ("left", "right"):
    _s, _e, _w = KP[f"{_side}_shoulder"], KP[f"{_side}_elbow"], KP[f"{_side}_wrist"]
    _h, _k, _a = KP[f"{_side}_hip"], KP[f"{_side}_knee"], KP[f"{_side}_ankle"]
    ANGLE_JOINTS[f"{_side}_elbow"] = (_s, _e, _w)  # 180 = arm straight
    ANGLE_JOINTS[f"{_side}_shoulder"] = (_h, _s, _e)  # 0 = arm by the side, 180 = overhead
    ANGLE_JOINTS[f"{_side}_hip"] = (_s, _h, _k)  # 180 = standing straight
    ANGLE_JOINTS[f"{_side}_knee"] = (_h, _k, _a)  # 180 = leg straight
    ANGLE_JOINTS[f"{_side}_body"] = (_s, _h, _a)  # 180 = shoulder, hip and ankle in line
ANGLE_NAMES = tuple(ANGLE_JOINTS)

DEFAULT_THRESHOLD = 0.3


def other_side(name: str) -> str:
    """``left_knee`` ↔ ``right_knee``."""
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def angle(
    kps: np.ndarray, scores: np.ndarray, a: int, b: int, c: int, threshold: float = DEFAULT_THRESHOLD
) -> np.ndarray:
    """Angle in degrees at joint ``b`` between bones b→a and b→c, per frame. NaN where any of
    the three joints scores below ``threshold`` or a bone has zero length."""
    kps = np.asarray(kps, np.float64)
    ba, bc = kps[:, a] - kps[:, b], kps[:, c] - kps[:, b]
    na, nc = np.linalg.norm(ba, axis=-1), np.linalg.norm(bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(ba * bc, axis=-1) / (na * nc)
    out = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    ok = (scores[:, a] >= threshold) & (scores[:, b] >= threshold) & (scores[:, c] >= threshold)
    ok &= (na > 1e-6) & (nc > 1e-6)
    return np.where(ok, out, np.nan)


def _midpoint(kps, scores, i, j, threshold):
    """Midpoint of two joints (or the one that is confident), NaN if neither is."""
    ci, cj = scores[:, i] >= threshold, scores[:, j] >= threshold
    both = 0.5 * (kps[:, i] + kps[:, j])
    out = np.where((ci & cj)[:, None], both, np.where(ci[:, None], kps[:, i], kps[:, j]))
    return np.where((ci | cj)[:, None], out, np.nan)


def torso_lean(kps: np.ndarray, scores: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Angle in degrees of the hip→shoulder line from vertical (0 = upright, 90 = horizontal)."""
    kps = np.asarray(kps, np.float64)
    hip = _midpoint(kps, scores, KP["left_hip"], KP["right_hip"], threshold)
    sho = _midpoint(kps, scores, KP["left_shoulder"], KP["right_shoulder"], threshold)
    v = sho - hip
    return np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1]))


def angles(
    kps: np.ndarray, scores: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> dict[str, np.ndarray]:
    """Every named angle plus ``torso_lean``, each ``(T,)`` degrees with NaN where unknown."""
    out = {name: angle(kps, scores, *idx, threshold=threshold) for name, idx in ANGLE_JOINTS.items()}
    out["torso_lean"] = torso_lean(kps, scores, threshold)
    return out


def view_ratio(kps: np.ndarray, scores: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> float:
    """Median shoulder width / body reference length. Roughly 0.6+ facing the camera, under
    0.35 side-on.

    The reference is the longer of the torso and 1.6 × the upper arm (about the same length
    on most people). Taking the longer one keeps the ratio meaningful when the torso points
    at the camera and looks short, as in a push-up filmed from the front.
    """
    kps = np.asarray(kps, np.float64)
    ls, rs, lh, rh = KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]
    le, re = KP["left_elbow"], KP["right_elbow"]
    ok = np.all(scores[:, [ls, rs]] >= threshold, axis=1)
    if not ok.any():
        return float("nan")
    k, s = kps[ok], scores[ok]
    width = np.linalg.norm(k[:, ls] - k[:, rs], axis=-1)
    hips = np.all(s[:, [lh, rh]] >= threshold, axis=1)
    torso = np.where(
        hips, np.linalg.norm(0.5 * (k[:, ls] + k[:, rs]) - 0.5 * (k[:, lh] + k[:, rh]), axis=-1), 0.0
    )
    arms = []
    for sh, el in ((ls, le), (rs, re)):
        arms.append(np.where(s[:, el] >= threshold, np.linalg.norm(k[:, sh] - k[:, el], axis=-1), np.nan))
    with warnings.catch_warnings():  # all-NaN columns: no elbow visible
        warnings.simplefilter("ignore", RuntimeWarning)
        upper_arm = np.nan_to_num(np.nanmax(np.stack(arms), axis=0), nan=0.0)
    ref = np.maximum(torso, 1.6 * upper_arm)
    good = ref > 1e-6
    if not good.any():
        return float("nan")
    return float(np.median(width[good] / ref[good]))


def normalize(
    kps: np.ndarray, scores: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    """Centre a window on its mean hip midpoint and scale by its median torso length, so
    features don't depend on where the person stands or how far from the camera. Motion within
    the window (e.g. rising during a pull-up) is kept.

    Returns (normalized (T, 17, 2) with unconfident joints set to 0, visibility mask (T, 17)).
    """
    kps = np.asarray(kps, np.float64)
    vis = scores >= threshold
    hip = _midpoint(kps, scores, KP["left_hip"], KP["right_hip"], threshold)
    sho = _midpoint(kps, scores, KP["left_shoulder"], KP["right_shoulder"], threshold)
    if np.isfinite(hip).all(axis=1).any():
        center = np.nanmean(hip, axis=0)
    elif vis.any():
        center = kps[vis].mean(axis=0)
    else:
        return np.zeros_like(kps, np.float32), vis
    torso = np.linalg.norm(sho - hip, axis=-1)
    torso = torso[np.isfinite(torso) & (torso > 1e-6)]
    if torso.size:
        scale = float(np.median(torso))
    else:  # no torso visible: a third of the visible extent
        pts = kps[vis]
        scale = max(float(np.ptp(pts[:, 1])) / 3.0, 1.0) if len(pts) > 1 else 1.0
    out = (kps - center) / scale
    return np.where(vis[..., None], out, 0.0).astype(np.float32), vis


def mirror(kps: np.ndarray, scores: np.ndarray, width: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal flip (``x → width - x``) with left/right joints swapped."""
    kps = np.asarray(kps).copy()
    kps[..., 0] = width - kps[..., 0]
    idx = list(FLIP_INDEX)
    return kps[:, idx], np.asarray(scores)[:, idx]


NUM_FEATURES = NUM_KEYPOINTS * 2 + (len(ANGLE_NAMES) + 1) * 2 + NUM_KEYPOINTS


def features(kps: np.ndarray, scores: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Window → ``(T, NUM_FEATURES)`` float32: normalized x/y per joint, sin/cos of every angle
    and the torso lean (0, 0 where unknown), and the joint visibility mask."""
    norm, vis = normalize(kps, scores, threshold)
    t = norm.shape[0]
    ang = angles(kps, scores, threshold)
    rad = np.radians(np.stack([ang[n] for n in (*ANGLE_NAMES, "torso_lean")], axis=-1))
    trig = np.concatenate([np.sin(rad), np.cos(rad)], axis=-1)
    trig = np.nan_to_num(trig, nan=0.0)
    return np.concatenate([norm.reshape(t, -1), trig, vis.astype(np.float32)], axis=-1).astype(np.float32)


def resample(kps: np.ndarray, scores: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray):
    """Linearly resample a sequence from timestamps ``src_t`` to ``dst_t`` (both increasing)."""
    flat = np.asarray(kps, np.float64).reshape(len(src_t), -1)
    k = np.stack([np.interp(dst_t, src_t, flat[:, j]) for j in range(flat.shape[1])], axis=-1)
    s = np.stack([np.interp(dst_t, src_t, scores[:, j]) for j in range(scores.shape[1])], axis=-1)
    return k.reshape(len(dst_t), *np.asarray(kps).shape[1:]).astype(np.float32), s.astype(np.float32)
