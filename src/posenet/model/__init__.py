"""Model factory: ``build_model(cfg.model)`` picks the architecture from ``cfg.model.arch``."""

from __future__ import annotations

import keras

from posenet.config import ModelConfig
from posenet.model.network import build_posenet
from posenet.model.vitpose import VARIANTS, build_vitpose  # also registers its custom layers


def build_model(cfg: ModelConfig) -> keras.Model:
    if cfg.arch == "posenet":
        return build_posenet(cfg)
    if cfg.arch not in VARIANTS:
        raise ValueError(f"unknown model.arch {cfg.arch!r}; expected posenet or one of {sorted(VARIANTS)}")
    return build_vitpose(cfg, variant=cfg.arch)
