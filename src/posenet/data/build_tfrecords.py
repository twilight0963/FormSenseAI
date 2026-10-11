"""Offline preprocessing: crop every COCO person instance into sharded TFRecords.

Each record holds a ``crop_shape`` JPEG centred on the person. The person box is
first extended to the model's input aspect ratio and enlarged by ``bbox_scale``;
the crop covers that box times ``crop / input`` per axis, so the central
input-sized window is exactly the enlarged person box at native scale and the
rest of the crop is context for scale/rotation augmentation.

Usage:
    python -m posenet.data.build_tfrecords --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from posenet.config import Config
from posenet.data.coco import load_instances
from posenet.geometry import apply_affine, bbox_to_region, crop_region, invert_affine

# Validation keeps every annotated instance so COCO AP is comparable to the literature.
SPLITS = {"train": "train2017", "val": "val2017"}


def _process_image(task):
    """Worker: load one image and crop all its instances. Returns plain Python data."""
    image_path, instances, crop_hw, input_hw, bbox_scale, jpeg_quality = task
    cv2.setNumThreads(1)  # parallelism comes from the process pool
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    return [
        crop_record(
            image, inst.bbox, inst.keypoints, inst.image_id, crop_hw, input_hw, bbox_scale, jpeg_quality
        )
        for inst in instances
    ]


def crop_record(
    image: np.ndarray,
    bbox: np.ndarray,
    keypoints: np.ndarray,
    image_id: int,
    crop_hw: tuple[int, int],
    input_hw: tuple[int, int],
    bbox_scale: float,
    jpeg_quality: int,
) -> dict:
    """Crop one person (COCO ``[x, y, w, h]`` box, (K, 3) keypoints) into a record dict."""
    center, (bw, bh) = bbox_to_region(bbox, bbox_scale, input_hw[1] / input_hw[0])
    size = (bw * crop_hw[1] / input_hw[1], bh * crop_hw[0] / input_hw[0])
    crop, affine = crop_region(image, center, size, (crop_hw[1], crop_hw[0]))
    ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")

    kps = keypoints.copy()
    labeled = kps[:, 2] > 0
    kps[labeled, :2] = apply_affine(kps[labeled, :2].astype(np.float64), affine)
    return {
        "jpeg": jpeg.tobytes(),
        "image_id": int(image_id),
        "bbox": np.asarray(bbox, np.float32).tolist(),
        "keypoints": kps.astype(np.float32).ravel().tolist(),
        "crop_to_orig": invert_affine(affine).astype(np.float32).ravel().tolist(),
    }


def _to_example(tf, rec):
    """Serialize a record; optional string fields (e.g. ``video``, ``action``) are kept as bytes."""

    def floats(v):
        return tf.train.Feature(float_list=tf.train.FloatList(value=v))

    feature = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[rec["jpeg"]])),
        "image_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[rec["image_id"]])),
        "bbox": floats(rec["bbox"]),
        "keypoints": floats(rec["keypoints"]),
        "crop_to_orig": floats(rec["crop_to_orig"]),
    }
    for key in ("video", "action"):
        if key in rec:
            feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[rec[key].encode()]))
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def build_split(cfg: Config, split: str, workers: int, seed: int) -> int:
    import tensorflow as tf  # imported lazily: spawned workers never load TF

    coco_root = Path(cfg.data.coco_root)
    ann_file = coco_root / "annotations" / f"person_keypoints_{SPLITS[split]}.json"
    if split == "train":
        min_kps, min_area = cfg.data.min_keypoints, cfg.data.min_bbox_area
        num_shards = cfg.data.num_shards_train
    else:
        min_kps, min_area = 1, 0.0
        num_shards = cfg.data.num_shards_val
    file_names, by_image = load_instances(ann_file, min_kps, min_area)

    image_ids = sorted(by_image)
    random.Random(seed).shuffle(image_ids)
    tasks = [
        (
            coco_root / SPLITS[split] / file_names[i],
            by_image[i],
            cfg.data.crop_shape,
            cfg.model.input_shape,
            cfg.data.bbox_scale,
            cfg.data.jpeg_quality,
        )
        for i in image_ids
    ]
    num_instances = sum(len(v) for v in by_image.values())

    out_dir = Path(cfg.data.tfrecord_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = [
        tf.io.TFRecordWriter(str(out_dir / f"{split}-{s:05d}-of-{num_shards:05d}.tfrecord"))
        for s in range(num_shards)
    ]
    count = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool, tqdm(total=num_instances, desc=split, unit="inst") as bar:
        for results in pool.imap(_process_image, tasks, chunksize=16):
            for rec in results:
                writers[count % num_shards].write(_to_example(tf, rec))
                count += 1
            bar.update(len(results))
    for w in writers:
        w.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build COCO person-crop TFRecords.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--splits", nargs="+", default=["val", "train"], choices=list(SPLITS))
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    out_dir = Path(cfg.data.tfrecord_dir)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for split in args.splits:
        meta[split] = build_split(cfg, split, args.workers, cfg.train.seed)
        meta["crop_hw"] = list(cfg.data.crop_shape)
        meta["input_hw"] = list(cfg.model.input_shape)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"{split}: {meta[split]} instances")


if __name__ == "__main__":
    main()
