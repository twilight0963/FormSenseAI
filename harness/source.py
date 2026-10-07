"""Open the frame source: a webcam if configured and available, otherwise a video file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog

import cv2

from harness.config import HarnessConfig

VIDEO_TYPES = [("Video", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"), ("All files", "*")]


@dataclass
class Source:
    cap: cv2.VideoCapture
    fps: float
    is_live: bool
    label: str


def open_source(cfg: HarnessConfig, parent=None) -> Source | None:
    """Return the opened source, or None if the user cancelled the file picker."""
    if cfg.use_webcam:
        cap = cv2.VideoCapture(cfg.webcam_index)
        if cap.isOpened():
            return Source(cap, cap.get(cv2.CAP_PROP_FPS) or 30.0, True, f"webcam {cfg.webcam_index}")
        cap.release()
        print(f"warning: cannot open webcam {cfg.webcam_index}; falling back to a video file")

    path = cfg.video_path
    if not path or not Path(path).is_file():
        if path:
            print(f"warning: video not found: {path}")
        path = filedialog.askopenfilename(parent=parent, title="Choose a video", filetypes=VIDEO_TYPES)
        if not path:
            return None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    return Source(cap, cap.get(cv2.CAP_PROP_FPS) or 30.0, False, Path(path).name)
