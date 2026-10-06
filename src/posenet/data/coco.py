"""COCO person-keypoint annotation parsing and instance filtering."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PersonInstance:
    image_id: int
    bbox: np.ndarray  # [x, y, w, h]
    keypoints: np.ndarray  # (17, 3): x, y, v


def load_instances(
    annotation_file: str | Path, min_keypoints: int, min_bbox_area: float
) -> tuple[dict[int, str], dict[int, list[PersonInstance]]]:
    """Return ``{image_id: file_name}`` and ``{image_id: [instances]}`` after filtering."""
    data = json.loads(Path(annotation_file).read_text())
    file_names = {img["id"]: img["file_name"] for img in data["images"]}

    by_image: dict[int, list[PersonInstance]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) or ann["num_keypoints"] < min_keypoints:
            continue
        w, h = ann["bbox"][2:]
        if w < 1 or h < 1 or w * h < min_bbox_area:
            continue
        kps = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        kps[kps[:, 2] == 0] = 0.0
        by_image[ann["image_id"]].append(
            PersonInstance(
                image_id=ann["image_id"],
                bbox=np.asarray(ann["bbox"], dtype=np.float64),
                keypoints=kps,
            )
        )
    return file_names, dict(by_image)
