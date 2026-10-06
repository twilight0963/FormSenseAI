"""Render augmented training samples with their keypoints for visual inspection.

Usage:
    python scripts/visualize_samples.py --out outputs/samples.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from posenet.config import Config
from posenet.data.pipeline import parse_example, shard_files, train_transform
from posenet.keypoints import SKELETON


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="outputs/samples.png")
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    ds = tf.data.TFRecordDataset(shard_files(cfg, "train")[:1]).take(args.count)
    tiles = []
    for serialized in ds:
        image, xy, w = train_transform(cfg, parse_example(serialized))
        img = cv2.cvtColor(image.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        pts = [tuple(int(v) for v in p) for p in xy.numpy()]
        valid = w.numpy() > 0
        for a, b in SKELETON:
            if valid[a] and valid[b]:
                cv2.line(img, pts[a], pts[b], (0, 255, 0), 1, cv2.LINE_AA)
        for i, p in enumerate(pts):
            if valid[i]:
                # Left joints red, right joints blue, centre (nose) white.
                color = (255, 255, 255) if i == 0 else ((0, 0, 255) if i % 2 else (255, 0, 0))
                cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)
        tiles.append(img)

    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
