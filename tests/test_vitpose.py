import importlib.util
from pathlib import Path

import keras
import numpy as np
import pytest

from posenet.config import Config
from posenet.keypoints import NUM_KEYPOINTS
from posenet.model import build_model
from posenet.model.vitpose import VARIANTS, block_index

ROOT = Path(__file__).parents[1]


def _vit_cfg(arch="vitpose_s"):
    cfg = Config()
    cfg.model.arch = arch
    cfg.model.input_hw = [256, 192]
    return cfg


@pytest.fixture(scope="module")
def small():
    return build_model(_vit_cfg().model)


def test_output_shape(small):
    out = small(np.zeros((2, 256, 192, 3), np.float32))
    assert tuple(out.shape) == (2, 64, 48, NUM_KEYPOINTS)
    assert keras.backend.standardize_dtype(out.dtype) == "float32"


def test_save_load_roundtrip(small, tmp_path):
    small.save(tmp_path / "m.keras")
    loaded = keras.models.load_model(tmp_path / "m.keras", compile=False)
    x = np.random.default_rng(0).uniform(0, 255, (1, 256, 192, 3)).astype(np.float32)
    np.testing.assert_allclose(loaded(x), small(x), atol=1e-5)


def test_block_index_covers_all_variables(small):
    depth = VARIANTS["vitpose_s"][1]
    buckets = {block_index(v.path, depth) for v in small.trainable_variables}
    assert buckets == set(range(depth + 2))


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_vitpose", ROOT / "scripts" / "convert_vitpose.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_checkpoint(dim, depth, k=NUM_KEYPOINTS, seed=0):
    """Random tensors laid out like a Hugging Face ViTPose simple-decoder checkpoint."""
    rng = np.random.default_rng(seed)
    t = {
        "backbone.embeddings.patch_embeddings.projection.weight": (dim, 3, 16, 16),
        "backbone.embeddings.patch_embeddings.projection.bias": (dim,),
        "backbone.embeddings.position_embeddings": (1, 193, dim),
        "backbone.layernorm.weight": (dim,),
        "backbone.layernorm.bias": (dim,),
        "head.conv.weight": (k, dim, 3, 3),
        "head.conv.bias": (k,),
    }
    for i in range(depth):
        p = f"backbone.encoder.layer.{i}"
        for part in ("query", "key", "value"):
            t[f"{p}.attention.attention.{part}.weight"] = (dim, dim)
            t[f"{p}.attention.attention.{part}.bias"] = (dim,)
        t[f"{p}.attention.output.dense.weight"] = (dim, dim)
        t[f"{p}.attention.output.dense.bias"] = (dim,)
        for ln in ("layernorm_before", "layernorm_after"):
            t[f"{p}.{ln}.weight"] = (dim,)
            t[f"{p}.{ln}.bias"] = (dim,)
        t[f"{p}.mlp.fc1.weight"] = (4 * dim, dim)
        t[f"{p}.mlp.fc1.bias"] = (4 * dim,)
        t[f"{p}.mlp.fc2.weight"] = (dim, 4 * dim)
        t[f"{p}.mlp.fc2.bias"] = (dim,)
    return {name: (0.02 * rng.standard_normal(shape)).astype(np.float32) for name, shape in t.items()}


def test_converter_uses_every_tensor_and_rejects_extras():
    conv = _load_converter()
    dim, depth, _ = VARIANTS["vitpose_s"]
    tensors = _fake_checkpoint(dim, depth)
    model = conv.convert(dict(tensors), conv.vitpose_config())
    np.testing.assert_array_equal(
        model.get_layer("block3_fc1").kernel.numpy(), tensors["backbone.encoder.layer.3.mlp.fc1.weight"].T
    )
    with pytest.raises(ValueError, match="unused"):
        conv.convert({**tensors, "extra.weight": np.zeros(1, np.float32)}, conv.vitpose_config())
