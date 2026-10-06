import json

import cv2
import numpy as np
import pytest
import tensorflow as tf

from posenet.config import Config
from posenet.data import pipeline
from posenet.data.build_tfrecords import _to_example
from posenet.keypoints import NUM_KEYPOINTS

CROP = 256
BLOB_XY = (110.0, 140.0)
JOINT = 5  # left_shoulder


def _blob_image():
    ys, xs = np.mgrid[0:CROP, 0:CROP] + 0.5
    blob = np.exp(-((xs - BLOB_XY[0]) ** 2 + (ys - BLOB_XY[1]) ** 2) / (2 * 4.0**2))
    return np.repeat((blob * 255).astype(np.uint8)[..., None], 3, axis=-1)


@pytest.fixture()
def cfg(tmp_path):
    cfg = Config()
    cfg.data.tfrecord_dir = str(tmp_path)
    cfg.train.batch_size = 4
    kps = np.zeros((NUM_KEYPOINTS, 3), np.float32)
    kps[JOINT] = [BLOB_XY[0], BLOB_XY[1], 2]
    _, jpeg = cv2.imencode(".jpg", _blob_image(), [cv2.IMWRITE_JPEG_QUALITY, 100])
    rec = {
        "jpeg": jpeg.tobytes(),
        "image_id": 7,
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "keypoints": kps.ravel().tolist(),
        "crop_to_orig": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }
    for split in ("train", "val"):
        with tf.io.TFRecordWriter(str(tmp_path / f"{split}-00000-of-00001.tfrecord")) as w:
            for _ in range(8):
                w.write(_to_example(tf, rec))
    (tmp_path / "meta.json").write_text(json.dumps({"train": 8, "val": 8}))
    return cfg


def _centroid(img):
    g = np.asarray(img, np.float64).sum(axis=-1)
    g = np.where(g > 0.05 * g.max(), g, 0.0)
    ys, xs = np.mgrid[0 : g.shape[0], 0 : g.shape[1]] + 0.5
    return np.array([(g * xs).sum() / g.sum(), (g * ys).sum() / g.sum()])


def test_augmentation_keeps_image_and_keypoints_aligned(cfg, monkeypatch):
    monkeypatch.setattr(pipeline, "_photometric", lambda cfg, img: img)
    tf.random.set_seed(0)
    serialized = next(iter(tf.data.TFRecordDataset(pipeline.shard_files(cfg, "train"))))
    ex = pipeline.parse_example(serialized)
    flips = set()
    for _ in range(30):
        image, xy, w = pipeline.train_transform(cfg, ex)
        valid = np.flatnonzero(w.numpy())
        assert valid.size == 1
        label = int(valid[0])
        assert label in (JOINT, JOINT + 1)  # left_shoulder or, if mirrored, right_shoulder
        flips.add(label)
        np.testing.assert_allclose(_centroid(image.numpy()), xy.numpy()[label], atol=0.35)
    assert flips == {JOINT, JOINT + 1}


def test_center_transform_is_exact_shift(cfg):
    serialized = next(iter(tf.data.TFRecordDataset(pipeline.shard_files(cfg, "val"))))
    image, xy, w = pipeline.center_transform(cfg, pipeline.parse_example(serialized))
    assert image.shape == (192, 192, 3)
    assert w.numpy()[JOINT] == 1.0 and w.numpy().sum() == 1.0
    np.testing.assert_allclose(_centroid(image.numpy()), xy.numpy()[JOINT], atol=0.05)


def test_build_dataset_shapes(cfg):
    x, y = next(iter(pipeline.build_dataset(cfg, "train", training=True)))
    assert x.shape == (4, 192, 192, 3) and x.dtype == tf.float32
    assert y.shape == (4, 48, 48, 2 * NUM_KEYPOINTS)
    val = list(pipeline.build_dataset(cfg, "val", training=False))
    assert sum(int(b[0].shape[0]) for b in val) == 8


def test_limit_selects_same_records_in_train_and_eval(tmp_path):
    """--overfit must train and validate on the same N records across shards."""
    cfg = Config()
    cfg.data.tfrecord_dir = str(tmp_path)
    _, jpeg = cv2.imencode(".jpg", _blob_image())
    shards, per_shard, n = 4, 4, 6
    for s in range(shards):
        path = tmp_path / f"train-{s:05d}-of-{shards:05d}.tfrecord"
        with tf.io.TFRecordWriter(str(path)) as w:
            for r in range(per_shard):
                kps = np.zeros((NUM_KEYPOINTS, 3), np.float32)
                kps[JOINT] = [40.0 + 8 * (s * per_shard + r), 128.0, 2]  # unique per record
                rec = {
                    "jpeg": jpeg.tobytes(),
                    "image_id": s * per_shard + r,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "keypoints": kps.ravel().tolist(),
                    "crop_to_orig": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                }
                w.write(_to_example(tf, rec))
    (tmp_path / "meta.json").write_text(json.dumps({"train": shards * per_shard}))

    def peaks(ds):
        _, y = next(iter(ds))
        heat = np.asarray(y)[..., JOINT].reshape(n, -1)
        return sorted(heat.argmax(axis=1).tolist())

    cfg.train.batch_size = n
    train = pipeline.build_dataset(cfg, "train", training=True, limit=n, augment=False)
    val = pipeline.build_dataset(cfg, "train", training=False, limit=n)
    expected = peaks(val)
    assert len(set(expected)) == n
    for _ in range(3):
        assert peaks(train) == expected
