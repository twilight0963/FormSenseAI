"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    coco_root: str = "data/coco"
    tfrecord_dir: str = "data/tfrecords"
    crop_size: int = 256
    bbox_scale: float = 1.25
    min_keypoints: int = 5
    min_bbox_area: float = 1024.0
    num_shards_train: int = 64
    num_shards_val: int = 8
    jpeg_quality: int = 95


@dataclass
class ModelConfig:
    input_size: int = 192
    heatmap_stride: int = 4
    width_multiplier: float = 1.0
    decoder_channels: int = 96

    @property
    def heatmap_size(self) -> int:
        return self.input_size // self.heatmap_stride


@dataclass
class AugmentConfig:
    scale_min: float = 0.75
    scale_max: float = 1.25
    rotation_deg: float = 30.0
    rotation_prob: float = 0.6
    translate_frac: float = 0.05
    flip_prob: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.03


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 140
    warmup_epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    heatmap_sigma: float = 2.0
    mixed_precision: bool = True
    jit_compile: bool = True
    steps_per_execution: int = 32
    shuffle_buffer: int = 8192
    early_stopping_patience: int = 20
    seed: int = 42
    output_dir: str = "runs"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown_sections = set(raw) - set(_SECTIONS)
        if unknown_sections:
            raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")
        sections = {}
        for name, section_cls in _SECTIONS.items():
            values = raw.get(name, {}) or {}
            known = {sf.name for sf in fields(section_cls)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f"Unknown keys in '{name}': {sorted(unknown)}")
            sections[name] = section_cls(**values)
        return cls(**sections)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "augment": AugmentConfig,
    "train": TrainConfig,
}
