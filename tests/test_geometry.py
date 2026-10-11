import numpy as np
import pytest

from posenet.geometry import apply_affine, bbox_to_region, crop_region, crop_square, invert_affine


def _blob_centroid(img: np.ndarray) -> np.ndarray:
    """Intensity centroid in continuous coords (pixel centre = index + 0.5)."""
    g = img.astype(np.float64)
    if g.ndim == 3:
        g = g.sum(axis=-1)
    ys, xs = np.mgrid[0 : g.shape[0], 0 : g.shape[1]]
    total = g.sum()
    return np.array([(g * (xs + 0.5)).sum() / total, (g * (ys + 0.5)).sum() / total])


@pytest.mark.parametrize("side,out_size", [(40.0, 160), (100.0, 100), (300.0, 96)])
def test_crop_maps_content_and_points_identically(side, out_size):
    img = np.zeros((240, 320, 3), np.float32)
    # Smooth blob so resampling preserves its centroid.
    cx, cy = 141.3, 102.7
    ys, xs = np.mgrid[0:240, 0:320] + 0.5
    blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 3.0**2))
    img[..., 0] = blob

    center = np.array([143.0, 104.0])
    crop, affine = crop_square(img, center, side, out_size)
    expected = apply_affine(np.array([cx, cy]), affine)
    got = _blob_centroid(crop)
    tolerance = 0.05 * max(1.0, out_size / side)
    np.testing.assert_allclose(got, expected, atol=tolerance)


def test_invert_affine_roundtrip():
    _, affine = crop_square(np.zeros((50, 50, 3), np.uint8), np.array([20.0, 30.0]), 37.0, 64)
    pts = np.array([[1.0, 2.0], [33.3, 44.4]])
    np.testing.assert_allclose(apply_affine(apply_affine(pts, affine), invert_affine(affine)), pts)


def test_crop_outside_image_is_zero_filled():
    img = np.full((20, 20, 3), 255, np.uint8)
    crop, _ = crop_square(img, np.array([0.0, 0.0]), 40.0, 40)
    assert crop[:15, :15].max() == 0
    assert crop[25:, 25:].min() == 255


def test_rectangular_crop_maps_content_and_points_identically():
    img = np.zeros((240, 320, 3), np.float32)
    cx, cy = 151.3, 92.7
    ys, xs = np.mgrid[0:240, 0:320] + 0.5
    img[..., 0] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 3.0**2))
    crop, affine = crop_region(img, np.array([150.0, 100.0]), (90.0, 120.0), (48, 64))
    assert crop.shape == (64, 48, 3)
    np.testing.assert_allclose(_blob_centroid(crop), apply_affine(np.array([cx, cy]), affine), atol=0.05)
    np.testing.assert_allclose(apply_affine(np.array([150.0, 100.0]), affine), [24.0, 32.0])


@pytest.mark.parametrize(
    "bbox,expected", [([10, 20, 30, 100], (75.0, 100.0)), ([0, 0, 90, 60], (90.0, 120.0))]
)
def test_bbox_to_region_extends_shorter_side(bbox, expected):
    center, size = bbox_to_region(np.array(bbox, float), 1.0, 0.75)
    np.testing.assert_allclose(size, expected)
    np.testing.assert_allclose(center, [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])
    _, (side, _) = bbox_to_region(np.array(bbox, float), 1.25, 1.0)
    assert side == pytest.approx(max(bbox[2:]) * 1.25)
