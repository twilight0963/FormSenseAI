import keras
import numpy as np
import pytest

from posenet.config import Config
from posenet.keypoints import NUM_KEYPOINTS
from posenet.losses import HeatmapLoss
from posenet.metrics import HeatmapPCK, oks
from posenet.model.network import build_posenet


@pytest.fixture(scope="module")
def model():
    return build_posenet(Config().model)


def test_output_shape_and_size(model):
    out = model(np.zeros((2, 192, 192, 3), np.float32))
    assert tuple(out.shape) == (2, 48, 48, NUM_KEYPOINTS)
    assert keras.backend.standardize_dtype(out.dtype) == "float32"
    assert model.count_params() < 4_000_000


def test_no_dense_layers(model):
    assert not any(type(layer).__name__ == "Dense" for layer in model.layers)


def _packed(heat, weights):
    w = np.broadcast_to(weights[:, None, None, :], heat.shape)
    return np.concatenate([heat, w], axis=-1).astype(np.float32)


def test_loss_ignores_masked_keypoints():
    rng = np.random.default_rng(0)
    target = rng.random((2, 48, 48, NUM_KEYPOINTS)).astype(np.float32)
    weights = np.ones((2, NUM_KEYPOINTS), np.float32)
    weights[:, 3] = 0.0
    pred = target.copy()
    pred[..., 3] += 5.0  # huge error only on a masked keypoint
    loss = HeatmapLoss(NUM_KEYPOINTS)(_packed(target, weights), pred)
    assert float(loss) == pytest.approx(0.0, abs=1e-7)
    pred[..., 4] += 1.0
    assert float(HeatmapLoss(NUM_KEYPOINTS)(_packed(target, weights), pred)) > 0


def test_pck_perfect_and_wrong():
    heat = np.zeros((1, 48, 48, NUM_KEYPOINTS), np.float32)
    heat[0, 10, 20, :] = 1.0
    weights = np.ones((1, NUM_KEYPOINTS), np.float32)
    metric = HeatmapPCK(NUM_KEYPOINTS)
    metric.update_state(_packed(heat, weights), heat)
    assert float(metric.result()) == 1.0
    wrong = np.zeros_like(heat)
    wrong[0, 40, 40, :] = 1.0
    metric.reset_state()
    metric.update_state(_packed(heat, weights), wrong)
    assert float(metric.result()) == 0.0


def test_oks_perfect_prediction():
    gt = np.zeros((NUM_KEYPOINTS, 3), np.float32)
    gt[:, :2] = np.random.default_rng(0).random((NUM_KEYPOINTS, 2)) * 100
    gt[:, 2] = 2
    assert oks(gt[:, :2], gt, area=5000.0) == pytest.approx(1.0)
    assert oks(gt[:, :2] + 20.0, gt, area=5000.0) < 0.9
