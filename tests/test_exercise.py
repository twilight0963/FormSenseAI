import json

import cv2
import numpy as np
import tensorflow as tf

from posenet.config import Config
from posenet.data import pipeline
from posenet.data.exercise import PENN_TO_COCO, coco_json, write_splits
from posenet.keypoints import FLIP_INDEX, KEYPOINT_NAMES, NUM_KEYPOINTS


def test_penn_mapping_is_mirrored_and_skips_head():
    assert 0 not in PENN_TO_COCO
    assert sorted(PENN_TO_COCO.values()) == list(range(5, 17))
    # Penn's "left shoulder" (1) is COCO's right shoulder; pairs stay pairs.
    assert KEYPOINT_NAMES[PENN_TO_COCO[1]] == "right_shoulder"
    assert all(FLIP_INDEX[PENN_TO_COCO[p]] == PENN_TO_COCO[p + 1] for p in range(1, 13, 2))


def _coco_clips(tmp_path, videos=12, frames=3):
    images, anns = [], []
    img = np.full((120, 160, 3), 80, np.uint8)
    for v in range(videos):
        (tmp_path / f"clip{v}").mkdir()
        for f in range(frames):
            name = f"clip{v}/{f:03d}.jpg"
            cv2.imwrite(str(tmp_path / name), img)
            iid = len(images)
            images.append({"id": iid, "file_name": name, "action": "squat"})
            kps = np.zeros((NUM_KEYPOINTS, 3), np.float32)
            kps[5:] = [[60 + 2 * j, 30 + 4 * j, 2] for j in range(12)]
            anns.append({"image_id": iid, "keypoints": kps.ravel().tolist(), "bbox": [50, 20, 40, 70]})
    path = tmp_path / "ann.json"
    path.write_text(json.dumps({"images": images, "annotations": anns}))
    return path


def test_coco_json_splits_by_video_and_mixes_into_training(tmp_path):
    splits = coco_json(_coco_clips(tmp_path), tmp_path, val_fraction=0.3, test_fraction=0.3)
    seen = [set(v) for v in splits.values()]
    assert sum(len(s) for s in seen) == 12 and not (
        seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2]
    )
    assert all(len(frames) == 3 for s in splits.values() for frames in s.values())

    cfg = Config()
    cfg.model.input_hw, cfg.data.crop_hw, cfg.model.heatmap_coords = [256, 192], [320, 240], "udp"
    cfg.train.batch_size = 4
    out = tmp_path / "records"
    meta = write_splits(cfg, splits, out, every=1, workers=1)
    assert meta["train"] == 3 * len(splits["train"]) and meta["train_actions"] == {"squat": meta["train"]}

    # Mixed batches: the "COCO" side is the same records under a second directory.
    cfg.data.tfrecord_dir = str(out)
    cfg.data.finetune_dirs = [str(out)]
    x, y = next(iter(pipeline.build_dataset(cfg, "train", training=True)))
    assert x.shape == (4, 256, 192, 3) and y.shape == (4, 64, 48, 2 * NUM_KEYPOINTS)
    assert pipeline.epoch_size(cfg) == int(meta["train"] / cfg.data.finetune_ratio)
    val = list(pipeline.build_dataset(cfg, "val", training=False))
    assert sum(int(b[0].shape[0]) for b in val) == meta.get("val", 0)
    assert tf.reduce_max(y[..., NUM_KEYPOINTS:]) == 1.0
