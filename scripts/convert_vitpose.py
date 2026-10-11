"""Convert a Hugging Face ViTPose (simple decoder) checkpoint to a Keras model.

Reads ``model.safetensors`` with numpy (no PyTorch needed), copies every tensor
into ``posenet.model.vitpose.build_vitpose`` and writes ``<out>/model.keras``
plus a ``config.yaml`` that the rest of the package (evaluate, infer, export,
train --init) picks up automatically.

Usage:
    python scripts/convert_vitpose.py                      # vitpose-base-simple → pretrained/vitpose_b_simple
    python scripts/convert_vitpose.py --repo usyd-community/vitpose-base-simple --out pretrained/x
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

from posenet.config import Config
from posenet.model.vitpose import VARIANTS, build_vitpose

# hidden size → variant name (the simple-decoder checkpoints only differ in width/depth).
_BY_WIDTH = {dim: name for name, (dim, _, _) in VARIANTS.items()}


def vitpose_config() -> Config:
    """Config for a ViTPose model: 256×192 input, UDP heatmaps, 3:4 crops with margin."""
    cfg = Config()
    cfg.model.arch = "vitpose_b"
    cfg.model.input_hw = [256, 192]
    cfg.model.heatmap_coords = "udp"
    cfg.data.tfrecord_dir = "data/tfrecords_hw"
    cfg.data.crop_hw = [320, 240]  # same 1.25x context ratio as the square 256/192 records
    return cfg


class _Weights:
    """Pops tensors by name so unused or missing ones are reported."""

    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = dict(tensors)

    def take(self, name: str) -> np.ndarray:
        if name not in self.tensors:
            raise KeyError(f"checkpoint has no tensor {name!r}")
        return self.tensors.pop(name)


def _assign(var, value: np.ndarray, name: str) -> None:
    if tuple(var.shape) != value.shape:
        raise ValueError(f"{name}: shape {value.shape} does not fit {var.path} {tuple(var.shape)}")
    var.assign(value.astype(np.float32))


def _linear(layer, w: _Weights, prefix: str) -> None:
    _assign(layer.kernel, w.take(f"{prefix}.weight").T, prefix)
    _assign(layer.bias, w.take(f"{prefix}.bias"), prefix)


def _layernorm(layer, w: _Weights, prefix: str) -> None:
    _assign(layer.gamma, w.take(f"{prefix}.weight"), prefix)
    _assign(layer.beta, w.take(f"{prefix}.bias"), prefix)


def _conv(layer, w: _Weights, prefix: str) -> None:
    _assign(layer.kernel, w.take(f"{prefix}.weight").transpose(2, 3, 1, 0), prefix)  # OIHW → HWIO
    _assign(layer.bias, w.take(f"{prefix}.bias"), prefix)


def _attention(layer, w: _Weights, prefix: str, heads: int) -> None:
    for part, dense in (
        ("query", layer._query_dense),
        ("key", layer._key_dense),
        ("value", layer._value_dense),
    ):
        kernel = w.take(f"{prefix}.attention.{part}.weight").T  # (D, heads * hd)
        bias = w.take(f"{prefix}.attention.{part}.bias")
        dim = kernel.shape[0]
        _assign(dense.kernel, kernel.reshape(dim, heads, dim // heads), f"{prefix}.{part}")
        _assign(dense.bias, bias.reshape(heads, dim // heads), f"{prefix}.{part}")
    out = w.take(f"{prefix}.output.dense.weight").T  # (heads * hd, D)
    dim = out.shape[1]
    _assign(layer._output_dense.kernel, out.reshape(heads, dim // heads, dim), f"{prefix}.output")
    _assign(layer._output_dense.bias, w.take(f"{prefix}.output.dense.bias"), f"{prefix}.output")


def convert(tensors: dict[str, np.ndarray], cfg: Config) -> keras.Model:
    dim = tensors["backbone.embeddings.patch_embeddings.projection.weight"].shape[0]
    cfg.model.arch = _BY_WIDTH[dim]
    _, depth, heads = VARIANTS[cfg.model.arch]
    num_keypoints = tensors["head.conv.weight"].shape[0]
    model = build_vitpose(cfg.model, num_keypoints, cfg.model.arch)

    w = _Weights(tensors)
    get = model.get_layer
    _conv(get("patch_embed"), w, "backbone.embeddings.patch_embeddings.projection")
    _assign(get("pos_embed").embedding, w.take("backbone.embeddings.position_embeddings"), "pos_embed")
    for i in range(depth):
        p = f"backbone.encoder.layer.{i}"
        _layernorm(get(f"block{i}_ln1"), w, f"{p}.layernorm_before")
        _attention(get(f"block{i}_attn"), w, f"{p}.attention", heads)
        _layernorm(get(f"block{i}_ln2"), w, f"{p}.layernorm_after")
        _linear(get(f"block{i}_fc1"), w, f"{p}.mlp.fc1")
        _linear(get(f"block{i}_fc2"), w, f"{p}.mlp.fc2")
    _layernorm(get("final_ln"), w, "backbone.layernorm")
    _conv(get("heatmaps"), w, "head.conv")
    if w.tensors:
        raise ValueError(f"unused checkpoint tensors: {sorted(w.tensors)}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="usyd-community/vitpose-base-simple")
    parser.add_argument("--out", default="pretrained/vitpose_b_simple")
    args = parser.parse_args()

    hf_config = json.loads(Path(hf_hub_download(args.repo, "config.json")).read_text())
    if not hf_config.get("use_simple_decoder", False):
        raise SystemExit(
            f"{args.repo} uses the classic decoder; only simple-decoder checkpoints are supported"
        )
    tensors = load_file(hf_hub_download(args.repo, "model.safetensors"))

    cfg = vitpose_config()
    model = convert(tensors, cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    cfg.to_yaml(out / "config.yaml")
    print(
        f"{args.repo} → {out / 'model.keras'} ({model.count_params() / 1e6:.1f}M params, arch {cfg.model.arch})"
    )


if __name__ == "__main__":
    main()
