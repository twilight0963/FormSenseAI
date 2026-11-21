import numpy as np
import pytest

from formcheck import skeleton
from formcheck.skeleton import KP


def standing(t: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """A front-facing stick figure, arms straight down, (t, 17, 2) pixels (y down)."""
    p = np.zeros((17, 2))
    p[KP["nose"]] = [100, 20]
    p[[KP["left_eye"], KP["right_eye"], KP["left_ear"], KP["right_ear"]]] = [
        [105, 15],
        [95, 15],
        [110, 20],
        [90, 20],
    ]
    for side, sx in (("left", 1), ("right", -1)):
        x = 100 + sx * 20
        p[KP[f"{side}_shoulder"]] = [x, 50]
        p[KP[f"{side}_elbow"]] = [x, 90]
        p[KP[f"{side}_wrist"]] = [x, 130]
        p[KP[f"{side}_hip"]] = [x, 130]
        p[KP[f"{side}_knee"]] = [x, 180]
        p[KP[f"{side}_ankle"]] = [x, 230]
    return np.repeat(p[None], t, axis=0), np.ones((t, 17))


def test_angle_straight_and_right():
    kps, scores = standing()
    a = skeleton.angles(kps, scores)
    assert a["left_elbow"][0] == pytest.approx(180)
    assert a["left_knee"][0] == pytest.approx(180)
    assert a["right_shoulder"][0] == pytest.approx(0, abs=1e-6)
    assert a["torso_lean"][0] == pytest.approx(0)
    kps[0, KP["left_wrist"]] = [160, 90]  # forearm horizontal → 90° at the elbow
    assert skeleton.angles(kps, scores)["left_elbow"][0] == pytest.approx(90)


def test_angle_nan_when_unconfident():
    kps, scores = standing()
    scores[0, KP["left_wrist"]] = 0.1
    a = skeleton.angles(kps, scores)
    assert np.isnan(a["left_elbow"][0])
    assert np.isfinite(a["right_elbow"][0])


def test_mirror_swaps_sides():
    kps, scores = standing()
    kps[0, KP["left_wrist"]] = [160, 90]
    m, ms = skeleton.mirror(kps, scores, width=200)
    a, am = skeleton.angles(kps, scores), skeleton.angles(m, ms)
    assert am["right_elbow"][0] == pytest.approx(a["left_elbow"][0])
    assert am["left_elbow"][0] == pytest.approx(a["right_elbow"][0])
    # The mirrored "left" shoulder sits where the right one was, reflected.
    assert m[0, KP["left_shoulder"], 0] == pytest.approx(200 - kps[0, KP["right_shoulder"], 0])


def test_features_invariant_to_translation_and_scale():
    kps, scores = standing(4)
    f1 = skeleton.features(kps, scores)
    f2 = skeleton.features(kps * 2.5 + [300, -40], scores)
    assert f1.shape == (4, skeleton.NUM_FEATURES)
    np.testing.assert_allclose(f1, f2, atol=1e-5)


def test_view_ratio_front_vs_side():
    kps, scores = standing()
    assert skeleton.view_ratio(kps, scores) > 0.4
    side = kps.copy()
    side[..., 0] = 100 + (side[..., 0] - 100) * 0.1  # squash horizontally ≈ turned side-on
    assert skeleton.view_ratio(side, scores) < 0.1


def test_resample():
    kps, scores = standing(3)
    kps[:, :, 0] += np.array([0, 10, 20])[:, None]
    k, _ = skeleton.resample(kps, scores, np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.5]))
    assert k[0, 0, 0] == pytest.approx(kps[0, 0, 0] + 5)
    assert k[1, 0, 0] == pytest.approx(kps[0, 0, 0] + 15)
