"""tf.data input pipelines built from the person-crop TFRecords.

Training samples get one random affine (scale, rotation, translation, flip)
applied to image and keypoints together, then photometric jitter. Targets are
packed as ``concat([heatmaps, weights broadcast to H×W], axis=-1)`` so a
standard Keras loss can mask unlabeled keypoints.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import tensorflow as tf

from posenet.config import Config
from posenet.data.heatmaps import render_heatmaps
from posenet.keypoints import FLIP_INDEX, NUM_KEYPOINTS

AUTOTUNE = tf.data.AUTOTUNE

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.int64),
    "bbox": tf.io.FixedLenFeature([4], tf.float32),
    "keypoints": tf.io.FixedLenFeature([NUM_KEYPOINTS * 3], tf.float32),
    "crop_to_orig": tf.io.FixedLenFeature([6], tf.float32),
}


def shard_files(cfg: Config, split: str) -> list[str]:
    files = sorted(str(p) for p in Path(cfg.data.tfrecord_dir).glob(f"{split}-*.tfrecord"))
    if not files:
        raise FileNotFoundError(
            f"No {split} TFRecords in {cfg.data.tfrecord_dir}; run posenet.data.build_tfrecords"
        )
    return files


def num_examples(cfg: Config, split: str) -> int:
    meta = json.loads((Path(cfg.data.tfrecord_dir) / "meta.json").read_text())
    return int(meta[split])


def parse_example(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
    ex = tf.io.parse_single_example(serialized, _FEATURES)
    ex["image"] = tf.io.decode_jpeg(ex["image"], channels=3)
    ex["keypoints"] = tf.reshape(ex["keypoints"], [NUM_KEYPOINTS, 3])
    return ex


def _translation(tx: tf.Tensor, ty: tf.Tensor) -> tf.Tensor:
    return tf.stack([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _augment_matrix(cfg: Config, flip: tf.Tensor) -> tf.Tensor:
    """3x3 crop→input affine in continuous coordinates."""
    aug, size = cfg.augment, float(cfg.model.input_size)
    crop_c = cfg.data.crop_size / 2.0
    out_c = size / 2.0

    scale = tf.random.uniform([], aug.scale_min, aug.scale_max)
    max_rot = aug.rotation_deg * math.pi / 180.0
    angle = tf.where(
        tf.random.uniform([]) < aug.rotation_prob,
        tf.random.uniform([], -max_rot, max_rot),
        0.0,
    )
    shift = tf.random.uniform([2], -aug.translate_frac * size, aug.translate_frac * size)

    cos, sin = tf.cos(angle) * scale, tf.sin(angle) * scale
    rot_scale = tf.stack([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    m = _translation(out_c + shift[0], out_c + shift[1]) @ rot_scale @ _translation(-crop_c, -crop_c)
    mirror = tf.constant([[-1.0, 0.0, size], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return tf.where(flip, mirror @ m, m)


def _warp_image(image: tf.Tensor, m: tf.Tensor, size: int) -> tf.Tensor:
    """Warp ``image`` with crop→input affine ``m`` (continuous coords)."""
    # ImageProjectiveTransform maps *output* index coords to *input* index coords.
    half = tf.constant([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    inv_half = tf.constant([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]])
    inv = inv_half @ tf.linalg.inv(m) @ half
    inv = inv / inv[2, 2]
    params = tf.reshape(inv, [9])[:8]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None],
        transforms=params[None],
        output_shape=tf.constant([size, size], tf.int32),
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
    )
    return out[0]


def _visible_weights(kps: tf.Tensor, size: int) -> tf.Tensor:
    xy, v = kps[:, :2], kps[:, 2]
    inside = tf.reduce_all((xy >= 0.0) & (xy < float(size)), axis=-1)
    return tf.cast((v > 0) & inside, tf.float32)


def _photometric(cfg: Config, image: tf.Tensor) -> tf.Tensor:
    aug = cfg.augment
    img = image / 255.0
    img = tf.image.random_brightness(img, aug.brightness)
    img = tf.image.random_contrast(img, 1.0 - aug.contrast, 1.0 + aug.contrast)
    img = tf.image.random_saturation(img, 1.0 - aug.saturation, 1.0 + aug.saturation)
    img = tf.image.random_hue(img, aug.hue)
    return tf.clip_by_value(img, 0.0, 1.0) * 255.0


def train_transform(cfg: Config, ex: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    size = cfg.model.input_size
    image = tf.cast(ex["image"], tf.float32)
    kps = ex["keypoints"]

    flip = tf.random.uniform([]) < cfg.augment.flip_prob
    m = _augment_matrix(cfg, flip)
    image = _warp_image(image, m, size)
    image = _photometric(cfg, image)

    xy = kps[:, :2] @ tf.transpose(m[:2, :2]) + m[:2, 2]
    kps = tf.concat([xy, kps[:, 2:]], axis=-1)
    # After mirroring, label i must take the location of its mirrored joint.
    kps = tf.where(flip, tf.gather(kps, FLIP_INDEX), kps)
    return image, kps[:, :2], _visible_weights(kps, size)


def center_transform(cfg: Config, ex: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Deterministic: take the central ``input_size`` window (exact integer shift)."""
    size, offset = cfg.model.input_size, (cfg.data.crop_size - cfg.model.input_size) // 2
    image = tf.cast(ex["image"][offset : offset + size, offset : offset + size], tf.float32)
    image = tf.ensure_shape(image, [size, size, 3])
    kps = ex["keypoints"]
    xy = kps[:, :2] - float(offset)
    kps = tf.concat([xy, kps[:, 2:]], axis=-1)
    return image, xy, _visible_weights(kps, size)


def make_targets(cfg: Config, keypoints: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    hm_size = cfg.model.heatmap_size
    heat = render_heatmaps(
        keypoints, weights, hm_size, cfg.model.heatmap_stride, cfg.train.heatmap_sigma
    )
    w = tf.broadcast_to(weights[:, None, None, :], tf.shape(heat))
    return tf.concat([heat, w], axis=-1)


def _record_dataset(files: list[str], shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(len(files))
        return ds.interleave(
            tf.data.TFRecordDataset,
            cycle_length=min(16, len(files)),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )
    return tf.data.TFRecordDataset(ds, num_parallel_reads=AUTOTUNE)


def build_dataset(
    cfg: Config, split: str, training: bool, limit: int | None = None, augment: bool | None = None
) -> tf.data.Dataset:
    """Batched ``(image, packed_targets)`` dataset.

    Args:
        training: shuffle + repeat (+ augment unless ``augment`` is False).
        limit: use only the first ``limit`` records (overfit / smoke tests).
    """
    augment = training if augment is None else augment
    if limit is not None:
        # Read shards one after another so "the first N records" is the same
        # set in every pipeline. Parallel reads interleave shards, and with
        # deterministic=False (training) that order changes between runs.
        ds = tf.data.TFRecordDataset(shard_files(cfg, split)).take(limit).cache()
    else:
        ds = _record_dataset(shard_files(cfg, split), shuffle=training)
    if training:
        ds = ds.shuffle(cfg.train.shuffle_buffer).repeat()

    transform = train_transform if augment else center_transform
    ds = ds.map(lambda s: transform(cfg, parse_example(s)), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(cfg.train.batch_size, drop_remainder=training)
    ds = ds.map(
        lambda img, kps, w: (img, make_targets(cfg, kps, w)), num_parallel_calls=AUTOTUNE
    )
    options = tf.data.Options()
    options.deterministic = not training
    return ds.with_options(options).prefetch(AUTOTUNE)


def build_eval_dataset(cfg: Config, batch_size: int) -> tf.data.Dataset:
    """Validation crops with metadata for mapping predictions back to the image."""
    offset = (cfg.data.crop_size - cfg.model.input_size) // 2

    def _map(serialized):
        ex = parse_example(serialized)
        image, _, _ = center_transform(cfg, ex)
        meta = {
            "image_id": ex["image_id"],
            "crop_to_orig": ex["crop_to_orig"],
            "offset": tf.constant(float(offset)),
        }
        return image, meta

    ds = _record_dataset(shard_files(cfg, "val"), shuffle=False)
    return ds.map(_map, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
