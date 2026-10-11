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
    # [height, width] of the stored crop; overrides crop_size for non-square models.
    crop_hw: list[int] | None = None
    bbox_scale: float = 1.25
    min_keypoints: int = 5
    min_bbox_area: float = 1024.0
    num_shards_train: int = 64
    num_shards_val: int = 8
    jpeg_quality: int = 95
    # Fine-tuning: exercise record dirs (posenet.data.exercise). When set, training batches mix
    # them with the COCO records at finetune_ratio, and validation uses their val splits.
    finetune_dirs: list[str] = field(default_factory=list)
    finetune_ratio: float = 0.7

    @property
    def crop_shape(self) -> tuple[int, int]:
        return tuple(self.crop_hw) if self.crop_hw else (self.crop_size, self.crop_size)


@dataclass
class ModelConfig:
    arch: str = "posenet"  # posenet | vitpose_s | vitpose_b | vitpose_l
    input_size: int = 192
    # [height, width] of the model input; overrides input_size for non-square models.
    input_hw: list[int] | None = None
    heatmap_stride: int = 4
    # Heatmap cell ↔ input pixel convention: "center" (cell j at (j + 0.5) * stride)
    # or "udp" (unbiased data processing: corner cells align with corner pixels).
    heatmap_coords: str = "center"
    width_multiplier: float = 1.0
    decoder_channels: int = 96

    @property
    def input_shape(self) -> tuple[int, int]:
        """(height, width) of the model input."""
        return tuple(self.input_hw) if self.input_hw else (self.input_size, self.input_size)

    @property
    def heatmap_shape(self) -> tuple[int, int]:
        h, w = self.input_shape
        return h // self.heatmap_stride, w // self.heatmap_stride

    @property
    def aspect(self) -> float:
        """Input width / height."""
        h, w = self.input_shape
        return w / h

    @property
    def udp(self) -> bool:
        return self.heatmap_coords == "udp"


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
    freeze_blocks: int = 0  # vitpose: keep the patch embedding and the first N blocks frozen
    layer_decay: float = 1.0  # vitpose: per-block LR multiplier (block i gets decay^(depth - i))
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
