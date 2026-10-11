"""Exercise keypoint datasets → person-crop TFRecords for fine-tuning.

Every source is reduced to per-frame annotations (image, COCO-17 keypoints, box,
video, action) and written with the same schema as the COCO records, plus
``video`` and ``action`` strings for per-exercise evaluation. Splits are made
**by video**, never by frame, so near-identical frames cannot leak between
train and test. Keypoints a source does not label get visibility 0, which the
heatmap loss ignores.

Sources:
    penn       Penn Action (13 joints; head is not a COCO keypoint and is left unlabeled)
    coco-json  any COCO-keypoints JSON, e.g. a CVAT export or InfiniteRep; the video is the
               image's parent directory unless the image entry has a ``video`` field

Usage:
    python -m posenet.data.exercise --config configs/vitpose_finetune.yaml \\
        --source penn --root data/penn_action/Penn_Action --out data/exercise/penn
    python -m posenet.data.exercise --config configs/vitpose_finetune.yaml \\
        --source coco-json --ann my_clips.json --images my_clips/ --out data/exercise/mine
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from posenet.config import Config
from posenet.data.build_tfrecords import _to_example, crop_record
from posenet.keypoints import NUM_KEYPOINTS

SPLITS = ("train", "val", "test")

# Penn Action joint index → COCO keypoint index (Penn's 0 = head has no COCO counterpart).
# Penn's "left"/"right" are mirrored relative to COCO's person-centric sides (checked against
# pretrained ViTPose on gym frames: 522 of 524 unambiguous joint pairs matched the swap).
PENN_TO_COCO = {1: 6, 2: 5, 3: 8, 4: 7, 5: 10, 6: 9, 7: 12, 8: 11, 9: 14, 10: 13, 11: 16, 12: 15}
PENN_GYM_ACTIONS = ("bench_press", "clean_and_jerk", "jumping_jacks", "pullup", "pushup", "situp", "squat")


@dataclass
class Frame:
    image: Path
    keypoints: np.ndarray  # (17, 3) x, y, v in original pixels
    bbox: np.ndarray  # [x, y, w, h]
    video: str
    action: str


def _hash_fraction(key: str) -> float:
    """Deterministic value in [0, 1) per key, stable across runs and machines."""
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 2**32


def _keypoint_box(kps: np.ndarray) -> np.ndarray | None:
    vis = kps[:, 2] > 0
    if vis.sum() < 2:
        return None
    lo, hi = kps[vis, :2].min(axis=0), kps[vis, :2].max(axis=0)
    size = np.maximum(hi - lo, 1.0) * 1.15  # joints sit inside the person outline
    return np.array([*((lo + hi) / 2 - size / 2), *size])


def penn_action(
    root: Path, actions: tuple[str, ...], val_fraction: float
) -> dict[str, dict[str, list[Frame]]]:
    """Penn Action → ``{split: {video: [frames]}}``. Official train/test, val carved from train by video."""
    from scipy.io import loadmat

    out: dict[str, dict[str, list[Frame]]] = {s: {} for s in SPLITS}
    for label in sorted((root / "labels").glob("*.mat")):
        m = loadmat(label)
        action = str(m["action"][0])
        if actions and action not in actions:
            continue
        seq = label.stem
        x, y, vis = m["x"], m["y"], m["visibility"].astype(bool)
        boxes = m["bbox"] if "bbox" in m else np.zeros((0, 4))
        frames = []
        for t in range(int(m["nframes"].ravel()[0])):
            kps = np.zeros((NUM_KEYPOINTS, 3), np.float32)
            for p, c in PENN_TO_COCO.items():
                if vis[t, p]:  # invisible joints carry guessed coordinates; leave them unlabeled
                    kps[c] = [x[t, p], y[t, p], 2]
            if t < len(boxes) and boxes[t][2] > boxes[t][0] and boxes[t][3] > boxes[t][1]:
                x1, y1, x2, y2 = boxes[t]
                bbox = np.array([x1, y1, x2 - x1, y2 - y1], np.float64)
            else:
                bbox = _keypoint_box(kps)
            if bbox is None or (kps[:, 2] > 0).sum() < 4:
                continue
            frames.append(Frame(root / "frames" / seq / f"{t + 1:06d}.jpg", kps, bbox, f"penn/{seq}", action))
        if not frames:
            continue
        official_train = int(np.asarray(m["train"]).ravel()[0]) == 1
        split = ("val" if _hash_fraction(seq) < val_fraction else "train") if official_train else "test"
        out[split][seq] = frames
    return out


def coco_json(
    ann_file: Path, images: Path, val_fraction: float, test_fraction: float
) -> dict[str, dict[str, list[Frame]]]:
    """COCO-keypoints JSON (CVAT / InfiniteRep) → ``{split: {video: [frames]}}``, split by video hash."""
    data = json.loads(ann_file.read_text())
    by_id = {img["id"]: img for img in data["images"]}
    videos: dict[str, list[Frame]] = defaultdict(list)
    for ann in data["annotations"]:
        img = by_id[ann["image_id"]]
        kps = np.asarray(ann["keypoints"], np.float32).reshape(-1, 3)
        if kps.shape[0] != NUM_KEYPOINTS or (kps[:, 2] > 0).sum() < 4:
            continue
        kps[kps[:, 2] == 0] = 0.0
        video = str(img.get("video") or Path(img["file_name"]).parent.as_posix() or ann_file.stem)
        bbox = np.asarray(ann["bbox"], np.float64) if ann.get("bbox") else _keypoint_box(kps)
        action = str(img.get("action", ann.get("action", "")))
        videos[video].append(Frame(images / img["file_name"], kps, bbox, video, action))

    out: dict[str, dict[str, list[Frame]]] = {s: {} for s in SPLITS}
    for video, frames in videos.items():
        r = _hash_fraction(video)
        split = "test" if r < test_fraction else "val" if r < test_fraction + val_fraction else "train"
        out[split][video] = sorted(frames, key=lambda f: str(f.image))
    return out


def _process_video(task):
    """Worker: crop every kept frame of one video. Returns plain Python data."""
    frames, every, crop_hw, input_hw, bbox_scale, jpeg_quality = task
    cv2.setNumThreads(1)
    results = []
    for frame in frames[::every]:
        image = cv2.imread(str(frame.image), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(frame.image)
        rec = crop_record(image, frame.bbox, frame.keypoints, 0, crop_hw, input_hw, bbox_scale, jpeg_quality)
        rec["video"], rec["action"] = frame.video, frame.action
        results.append(rec)
    return results


def write_splits(
    cfg: Config, splits: dict[str, dict[str, list[Frame]]], out_dir: Path, every: int, workers: int
) -> dict:
    import tensorflow as tf  # imported lazily: spawned workers never load TF

    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "crop_hw": list(cfg.data.crop_shape),
        "input_hw": list(cfg.model.input_shape),
        "every": every,
    }
    image_id = 0
    for split, videos in splits.items():
        if not videos:
            continue
        names = sorted(videos)
        random.Random(cfg.train.seed).shuffle(names)
        tasks = [
            (
                videos[v],
                every,
                cfg.data.crop_shape,
                cfg.model.input_shape,
                cfg.data.bbox_scale,
                cfg.data.jpeg_quality,
            )
            for v in names
        ]
        num_shards = max(1, min(16, len(names) // 8))
        writers = [
            tf.io.TFRecordWriter(str(out_dir / f"{split}-{s:05d}-of-{num_shards:05d}.tfrecord"))
            for s in range(num_shards)
        ]
        count = 0
        actions: dict[str, int] = defaultdict(int)
        total = sum(len(videos[v][::every]) for v in names)
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool, tqdm(total=total, desc=split, unit="frame") as bar:
            for results in pool.imap(_process_video, tasks):
                for rec in results:
                    rec["image_id"] = image_id
                    image_id += 1
                    writers[count % num_shards].write(_to_example(tf, rec))
                    actions[rec["action"]] += 1
                    count += 1
                bar.update(len(results))
        for w in writers:
            w.close()
        meta[split] = count
        meta[f"{split}_videos"] = len(names)
        meta[f"{split}_actions"] = dict(sorted(actions.items()))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exercise person-crop TFRecords.")
    parser.add_argument("--config", default="configs/vitpose_finetune.yaml", help="defines crop/input shape")
    parser.add_argument("--source", required=True, choices=["penn", "coco-json"])
    parser.add_argument("--root", type=Path, help="penn: dataset root (contains frames/ and labels/)")
    parser.add_argument("--actions", nargs="*", default=list(PENN_GYM_ACTIONS), help="penn: actions to keep")
    parser.add_argument("--ann", type=Path, help="coco-json: annotation file")
    parser.add_argument("--images", type=Path, help="coco-json: directory the file_names are relative to")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--every", type=int, default=3, help="keep every Nth frame (consecutive frames are near-duplicates)"
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--test-fraction", type=float, default=0.15, help="coco-json only; penn uses its official split"
    )
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.source == "penn":
        if not args.root:
            parser.error("--root is required for --source penn")
        splits = penn_action(args.root, tuple(args.actions), args.val_fraction)
    else:
        if not (args.ann and args.images):
            parser.error("--ann and --images are required for --source coco-json")
        splits = coco_json(args.ann, args.images, args.val_fraction, args.test_fraction)
    meta = write_splits(cfg, splits, args.out, args.every, args.workers)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
