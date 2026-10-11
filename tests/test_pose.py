from pathlib import Path

import numpy as np
import pytest

from harness.draw import draw_pose
from harness.pose import (
    FLIP_INDEX,
    HEATMAP_SIGMA,
    NUM_KEYPOINTS,
    PoseEstimator,
    apply_affine,
    bbox_to_region,
    crop_region,
    decode_heatmaps,
    flip_heatmaps,
    invert_affine,
)

ROOT = Path(__file__).parents[1]
INPUT_SIZE, HEATMAP_STRIDE = 192, 4
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


def test_udp_decode_recovers_rectangular_locations():
    h, w = 256, 192
    rng = np.random.default_rng(2)
    kps = rng.uniform(12.0, [w - 12.0, h - 12.0], size=(2, NUM_KEYPOINTS, 2)).astype(np.float32)
    sx, sy = (w - 1.0) / (w / HEATMAP_STRIDE - 1.0), (h - 1.0) / (h / HEATMAP_STRIDE - 1.0)
    cx, cy = (kps[..., 0] - 0.5) / sx, (kps[..., 1] - 0.5) / sy  # UDP cell coords
    ys, xs = np.arange(h // HEATMAP_STRIDE), np.arange(w // HEATMAP_STRIDE)
    gx = np.exp(-((xs[None, :, None] - cx[:, None, :]) ** 2) / (2 * HEATMAP_SIGMA**2))
    gy = np.exp(-((ys[None, :, None] - cy[:, None, :]) ** 2) / (2 * HEATMAP_SIGMA**2))
    heat = (gy[:, :, None, :] * gx[:, None, :, :]).astype(np.float32)
    coords, _ = decode_heatmaps(heat, HEATMAP_STRIDE, HEATMAP_SIGMA, udp=True)
    assert np.abs(coords - kps).max() < 0.5


def test_rectangular_crop_and_aspect_box():
    image = np.zeros((480, 640, 3), np.uint8)
    center, size = bbox_to_region(np.array([100.0, 50.0, 60.0, 200.0]), 1.25, 0.75)
    np.testing.assert_allclose(size, (187.5, 250.0))
    crop, affine = crop_region(image, center, size, (192, 256))
    assert crop.shape == (256, 192, 3)
    np.testing.assert_allclose(apply_affine(center, affine), [96.0, 128.0])


def test_crop_affine_round_trip():
    image = np.zeros((480, 640, 3), np.uint8)
    _, affine = crop_region(image, np.array([300.0, 200.0]), (250.0, 250.0), (INPUT_SIZE, INPUT_SIZE))
    pts = np.array([[300.0, 200.0], [175.0, 75.0], [425.0, 325.0]])
    in_crop = apply_affine(pts, affine)
    np.testing.assert_allclose(in_crop[0], [INPUT_SIZE / 2] * 2)
    np.testing.assert_allclose(in_crop[1], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(apply_affine(in_crop, invert_affine(affine)), pts, atol=1e-9)


MODELS = {
    "posenet_tflite": ("runs/20261006-164407/export/posenet_fp16.tflite", (192, 192), False),
    "vitpose_saved_model": ("pretrained/vitpose_b_simple/export/saved_model", (256, 192), True),
}


@pytest.fixture(scope="module", params=sorted(MODELS))
def estimator(request):
    rel, input_hw, udp = MODELS[request.param]
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"exported model not found: {path}")
    est = PoseEstimator(path, flip_test=True)
    assert est.input_hw == input_hw and est.udp == udp and est.stride == 4
    return est


def test_search_windows_cover_wide_frames(estimator):
    windows = estimator.search_windows(720, 1280)
    assert len(windows) >= 2
    (c0, s0), (c1, s1) = windows[0], windows[-1]
    assert c0[0] - s0[0] / 2 == pytest.approx(0.0) and c1[0] + s1[0] / 2 == pytest.approx(1280.0)
    assert all(s[0] / s[1] == pytest.approx(estimator.aspect) for _, s in windows)


def test_estimator_on_blank_frame(estimator):
    frame = np.zeros((480, 640, 3), np.uint8)
    pose = estimator.predict(frame)
    assert pose.keypoints.shape == (NUM_KEYPOINTS, 2)
    assert pose.scores.shape == (NUM_KEYPOINTS,)
    assert np.isfinite(pose.keypoints).all() and np.isfinite(pose.scores).all()
    tracked = estimator.predict(frame, prev=pose)
    assert tracked.keypoints.shape == (NUM_KEYPOINTS, 2)
    assert draw_pose(frame[..., ::-1], pose).shape == frame.shape
