from pathlib import Path

import numpy as np
import pytest

from harness.config import HarnessConfig
from harness.draw import draw_pose
from harness.pose import (
    FLIP_INDEX,
    HEATMAP_SIGMA,
    HEATMAP_STRIDE,
    INPUT_SIZE,
    NUM_KEYPOINTS,
    PoseEstimator,
    apply_affine,
    crop_square,
    decode_heatmaps,
    flip_heatmaps,
    invert_affine,
)

HM = INPUT_SIZE // HEATMAP_STRIDE


def _render(kps: np.ndarray) -> np.ndarray:
    """(N, K, 2) input-pixel keypoints → (N, HM, HM, K) Gaussian heatmaps (training's target convention)."""
    centers = np.arange(HM, dtype=np.float32) + 0.5
    mx = kps[..., 0] / HEATMAP_STRIDE  # (N, K) in heatmap continuous coords
    my = kps[..., 1] / HEATMAP_STRIDE
    gx = np.exp(-((centers[None, :, None] - mx[:, None, :]) ** 2) / (2 * HEATMAP_SIGMA**2))
    gy = np.exp(-((centers[None, :, None] - my[:, None, :]) ** 2) / (2 * HEATMAP_SIGMA**2))
    return (gy[:, :, None, :] * gx[:, None, :, :]).astype(np.float32)


def test_decode_recovers_subpixel_locations():
    rng = np.random.default_rng(0)
    kps = rng.uniform(12.0, INPUT_SIZE - 12.0, size=(3, NUM_KEYPOINTS, 2)).astype(np.float32)
    coords, scores = decode_heatmaps(_render(kps), HEATMAP_STRIDE, HEATMAP_SIGMA)
    assert np.abs(coords - kps).max() < 0.5
    assert scores.min() > 0.8


def test_flip_heatmaps_is_an_involution():
    heat = np.random.default_rng(1).random((1, HM, HM, NUM_KEYPOINTS)).astype(np.float32)
    np.testing.assert_array_equal(flip_heatmaps(flip_heatmaps(heat, FLIP_INDEX), FLIP_INDEX), heat)


def test_crop_affine_round_trip():
    image = np.zeros((480, 640, 3), np.uint8)
    _, affine = crop_square(image, np.array([300.0, 200.0]), 250.0, INPUT_SIZE)
    pts = np.array([[300.0, 200.0], [175.0, 75.0], [425.0, 325.0]])
    in_crop = apply_affine(pts, affine)
    np.testing.assert_allclose(in_crop[0], [INPUT_SIZE / 2] * 2)
    np.testing.assert_allclose(in_crop[1], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(apply_affine(in_crop, invert_affine(affine)), pts, atol=1e-9)


@pytest.fixture(scope="module")
def estimator():
    path = Path(__file__).parents[1] / HarnessConfig.model_path
    if not path.is_file():
        pytest.skip(f"exported model not found: {path}")
    return PoseEstimator(path, flip_test=True)


def test_estimator_on_blank_frame(estimator):
    frame = np.zeros((480, 640, 3), np.uint8)
    pose = estimator.predict(frame)
    assert pose.keypoints.shape == (NUM_KEYPOINTS, 2)
    assert pose.scores.shape == (NUM_KEYPOINTS,)
    assert np.isfinite(pose.keypoints).all() and np.isfinite(pose.scores).all()
    tracked = estimator.predict(frame, prev=pose)
    assert tracked.keypoints.shape == (NUM_KEYPOINTS, 2)
    assert draw_pose(frame[..., ::-1], pose).shape == frame.shape
