"""PCK on exercise records (``posenet.data.exercise``), per exercise and per joint.

A keypoint is correct when it lies within ``threshold * max(box w, h)`` of the
label in original-image pixels. PCK@0.2 is the usual Penn Action metric, but a
strong pretrained model saturates it, so 0.05 and 0.1 are reported too.
Ground-truth boxes are used, so this measures keypoint accuracy, not detection.

Example:
    python -m posenet.evaluate_exercise --model pretrained/vitpose_b_simple/model.keras --dirs data/exercise/penn
    python -m posenet.evaluate_exercise --model runs/<run>/best.keras        # dirs from the run config
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from posenet.data.pipeline import _FEATURES, center_transform, num_examples, shard_files
from posenet.inference import PoseEstimator
from posenet.keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS

_EXTRA = {
    "video": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "action": tf.io.FixedLenFeature([], tf.string, default_value=""),
}


def _dataset(estimator: PoseEstimator, dirs: list[str], split: str, batch_size: int) -> tf.data.Dataset:
    cfg = estimator.cfg

    def _map(serialized):
        ex = tf.io.parse_single_example(serialized, {**_FEATURES, **_EXTRA})
        ex["image"] = tf.io.decode_jpeg(ex["image"], channels=3)
        ex["keypoints"] = tf.reshape(ex["keypoints"], [NUM_KEYPOINTS, 3])
        image, xy, weights = center_transform(cfg, ex)
        meta = {
            "xy": xy,
            "weights": weights,
            "scale": ex["crop_to_orig"][0],  # crop → original is a uniform scale (plus shift)
            "box_side": tf.reduce_max(ex["bbox"][2:]),
            "action": ex["action"],
        }
        return image, meta

    files = shard_files(cfg, split, dirs)
    return tf.data.TFRecordDataset(files).map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)


def evaluate(
    estimator: PoseEstimator, dirs: list[str], split: str, batch_size: int, thresholds: list[float]
) -> dict:
    t = np.asarray(thresholds)[:, None, None]  # (T, 1, 1)
    hits = defaultdict(lambda: np.zeros((len(thresholds), NUM_KEYPOINTS)))
    counts = defaultdict(lambda: np.zeros(NUM_KEYPOINTS))
    total = num_examples(estimator.cfg, split, dirs)
    with tqdm(total=total, desc=split, unit="frame") as bar:
        for images, meta in _dataset(estimator, dirs, split, batch_size):
            coords, _ = estimator.decode(estimator.heatmaps(images.numpy()))
            dist = np.linalg.norm(coords - meta["xy"].numpy(), axis=-1) * meta["scale"].numpy()[:, None]
            ok = dist[None] <= t * meta["box_side"].numpy()[None, :, None]  # (T, N, K)
            w = meta["weights"].numpy()
            for i, action in enumerate(meta["action"].numpy()):
                for key in ("all", action.decode() or "unknown"):
                    hits[key] += ok[:, i] * w[i]
                    counts[key] += w[i]
            bar.update(len(w))

    def pck(key):
        return {
            f"{th:g}": float(h.sum() / max(counts[key].sum(), 1.0))
            for th, h in zip(thresholds, hits[key], strict=True)
        }

    labeled = np.flatnonzero(counts["all"] > 0)
    return {
        "split": split,
        "frames": total,
        "pck": pck("all"),
        "per_action": {k: pck(k) for k in sorted(hits) if k != "all"},
        "per_joint": {
            KEYPOINT_NAMES[j]: {
                f"{th:g}": float(h[j] / counts["all"][j])
                for th, h in zip(thresholds, hits["all"], strict=True)
            }
            for j in labeled
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PCK on exercise records.")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dirs", nargs="*", help="exercise record dirs (default: the model config's finetune_dirs)"
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-flip", action="store_true")
    args = parser.parse_args()

    estimator = PoseEstimator.from_path(args.model, flip_test=not args.no_flip)
    dirs = args.dirs or estimator.cfg.data.finetune_dirs
    if not dirs:
        parser.error("no exercise record dirs: pass --dirs")
    for d in dirs:
        meta = json.loads((Path(d) / "meta.json").read_text())
        if tuple(meta["crop_hw"]) != estimator.cfg.data.crop_shape:
            parser.error(
                f"{d} holds {meta['crop_hw']} crops; the model config expects {estimator.cfg.data.crop_shape}"
            )
    result = evaluate(estimator, dirs, args.split, args.batch_size, args.thresholds)
    result.update(model=str(args.model), dirs=dirs, flip_test=not args.no_flip)

    out = Path(args.model).parent / f"exercise_{args.split}_pck.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
