"""Harness settings, loaded from a JSON file (``config.json`` by default)."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class HarnessConfig:
    model_path: str = "runs/20261006-164407/export/posenet_fp16.tflite"
    use_webcam: bool = False
    webcam_index: int = 0
    video_path: str = ""  # empty → pick a file at startup
    loop_video: bool = True
    flip_test: bool = True
    score_threshold: float = 0.3
    track: bool = True  # crop around the previous frame's pose instead of re-searching
    num_threads: int = 4
    window_width: int = 1280
    window_height: int = 720


def load_config(path: str | Path) -> HarnessConfig:
    """Merge the JSON file over the defaults; relative paths resolve against the file's directory."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    known = {f.name for f in fields(HarnessConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    cfg = HarnessConfig(**data)

    base = path.resolve().parent
    cfg.model_path = str(base / cfg.model_path)
    if cfg.video_path:
        cfg.video_path = str(base / Path(cfg.video_path).expanduser())
    if not Path(cfg.model_path).is_file():
        raise FileNotFoundError(f"model not found: {cfg.model_path} (set model_path in {path})")
    return cfg
