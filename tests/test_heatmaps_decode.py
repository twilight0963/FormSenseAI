import numpy as np
import tensorflow as tf

from posenet.data.heatmaps import render_heatmaps
from posenet.decode import decode_heatmaps, flip_heatmaps
from posenet.keypoints import FLIP_INDEX

SIZE, STRIDE, SIGMA = 192, 4, 2.0
HM = SIZE // STRIDE


def _random_keypoints(rng, n=4, k=17):
    return rng.uniform(12.0, SIZE - 12.0, size=(n, k, 2)).astype(np.float32)


def test_heatmap_peak_and_shape():
    kps = np.array([[[50.0, 70.0]]], np.float32)
    heat = render_heatmaps(kps, np.ones((1, 1)), HM, STRIDE, SIGMA).numpy()
    assert heat.shape == (1, HM, HM, 1)
    y, x = np.unravel_index(np.argmax(heat[0, ..., 0]), (HM, HM))
    # 50 / 4 = 12.5 → cell 12 centre; 70 / 4 = 17.5 → cell 17 centre.
    assert (x, y) == (12, 17)
    np.testing.assert_allclose(heat.max(), 1.0, atol=1e-6)


def test_invalid_keypoints_render_empty():
    kps = np.full((1, 2, 2), 82.0, np.float32)  # 82 / 4 - 0.5 = cell 20 exactly
    heat = render_heatmaps(kps, np.array([[1.0, 0.0]]), HM, STRIDE, SIGMA).numpy()
    assert heat[..., 0].max() > 0.99
    assert heat[..., 1].max() == 0.0


def test_decode_recovers_subpixel_locations():
    rng = np.random.default_rng(0)
    kps = _random_keypoints(rng)
    heat = render_heatmaps(kps, np.ones(kps.shape[:2]), HM, STRIDE, SIGMA).numpy()
    coords, scores = decode_heatmaps(heat, STRIDE, SIGMA)
    err = np.abs(coords - kps)
    assert err.max() < 0.5, err.max()  # input pixels
    assert scores.min() > 0.8
    # Integer-peak decoding is strictly worse, proving refinement works.
    coarse, _ = decode_heatmaps(heat, STRIDE, SIGMA, refine=False)
    assert np.abs(coarse - kps).mean() > 2 * err.mean()


def test_flip_heatmaps_matches_flipped_keypoints():
    rng = np.random.default_rng(1)
    kps = _random_keypoints(rng, n=2)
    w = np.ones(kps.shape[:2], np.float32)
    flipped = kps.copy()
    flipped[..., 0] = SIZE - flipped[..., 0]
    flipped = flipped[:, list(FLIP_INDEX)]
    heat_flipped = render_heatmaps(flipped, w, HM, STRIDE, SIGMA).numpy()
    heat = render_heatmaps(kps, w, HM, STRIDE, SIGMA).numpy()
    np.testing.assert_allclose(flip_heatmaps(heat_flipped, FLIP_INDEX), heat, atol=1e-5)


def test_image_flip_equals_coordinate_flip():
    img = np.zeros((1, 8, 8, 1), np.float32)
    img[0, 2, 1, 0] = 1.0  # pixel centre x = 1.5
    flipped = tf.reverse(img, axis=[2]).numpy()
    x = np.argwhere(flipped[0, 2, :, 0])[0, 0] + 0.5
    assert x == 8 - 1.5


def test_udp_rectangular_render_decode_roundtrip():
    rng = np.random.default_rng(2)
    h, w = 256, 192
    kps = rng.uniform(8.0, [w - 8.0, h - 8.0], size=(3, 17, 2)).astype(np.float32)
    heat = render_heatmaps(
        kps, np.ones(kps.shape[:2]), (h // STRIDE, w // STRIDE), STRIDE, SIGMA, udp=True
    ).numpy()
    assert heat.shape == (3, 64, 48, 17)
    coords, _ = decode_heatmaps(heat, STRIDE, SIGMA, udp=True)
    assert np.abs(coords - kps).max() < 0.5
    # Decoding UDP heatmaps with the centre convention is biased by up to ~1.5 px.
    wrong, _ = decode_heatmaps(heat, STRIDE, SIGMA, udp=False)
    assert np.abs(wrong - kps).max() > 1.0


def test_udp_corner_cells_align_with_corner_pixels():
    kps = np.array([[[0.5, 0.5], [191.5, 255.5]]], np.float32)  # centres of the corner pixels
    heat = render_heatmaps(kps, np.ones((1, 2)), (64, 48), STRIDE, SIGMA, udp=True).numpy()
    assert np.unravel_index(heat[0, ..., 0].argmax(), (64, 48)) == (0, 0)
    assert np.unravel_index(heat[0, ..., 1].argmax(), (64, 48)) == (63, 47)
    np.testing.assert_allclose(heat.max(axis=(1, 2)), 1.0, atol=1e-6)


def test_udp_flip_heatmaps_matches_flipped_keypoints():
    rng = np.random.default_rng(3)
    kps = rng.uniform(12.0, [180.0, 244.0], size=(2, 17, 2)).astype(np.float32)
    w = np.ones(kps.shape[:2], np.float32)
    flipped = kps.copy()
    flipped[..., 0] = 192 - flipped[..., 0]
    flipped = flipped[:, list(FLIP_INDEX)]
    shape = (64, 48)
    heat = render_heatmaps(kps, w, shape, STRIDE, SIGMA, udp=True).numpy()
    heat_flipped = render_heatmaps(flipped, w, shape, STRIDE, SIGMA, udp=True).numpy()
    np.testing.assert_allclose(flip_heatmaps(heat_flipped, FLIP_INDEX), heat, atol=1e-5)
