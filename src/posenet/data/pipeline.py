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


def shard_files(cfg: Config, split: str, dirs: list[str] | None = None) -> list[str]:
    """TFRecord shards of ``split`` in ``dirs`` (default: the COCO ``tfrecord_dir``)."""
    dirs = dirs or [cfg.data.tfrecord_dir]
    files = sorted(str(p) for d in dirs for p in Path(d).glob(f"{split}-*.tfrecord"))
    if not files:
        raise FileNotFoundError(f"No {split} TFRecords in {dirs}; run posenet.data.build_tfrecords")
    return files


def num_examples(cfg: Config, split: str, dirs: list[str] | None = None) -> int:
    return sum(
        int(json.loads((Path(d) / "meta.json").read_text())[split]) for d in dirs or [cfg.data.tfrecord_dir]
    )


def epoch_size(cfg: Config) -> int:
    """Training samples per epoch: one pass over COCO, or over the exercise data when fine-tuning."""
    if cfg.data.finetune_dirs:
        return int(num_examples(cfg, "train", cfg.data.finetune_dirs) / cfg.data.finetune_ratio)
    return num_examples(cfg, "train")


def parse_example(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
    ex = tf.io.parse_single_example(serialized, _FEATURES)
    ex["image"] = tf.io.decode_jpeg(ex["image"], channels=3)
    ex["keypoints"] = tf.reshape(ex["keypoints"], [NUM_KEYPOINTS, 3])
    return ex


def _translation(tx: tf.Tensor, ty: tf.Tensor) -> tf.Tensor:
    return tf.stack([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _augment_matrix(cfg: Config, flip: tf.Tensor) -> tf.Tensor:
    """3x3 crop→input affine in continuous coordinates."""
    aug = cfg.augment
    out_h, out_w = (float(v) for v in cfg.model.input_shape)
    crop_h, crop_w = cfg.data.crop_shape

    scale = tf.random.uniform([], aug.scale_min, aug.scale_max)
    max_rot = aug.rotation_deg * math.pi / 180.0
    angle = tf.where(
        tf.random.uniform([]) < aug.rotation_prob,
        tf.random.uniform([], -max_rot, max_rot),
        0.0,
    )
    shift = tf.random.uniform([2], -aug.translate_frac, aug.translate_frac) * tf.constant([out_w, out_h])

    cos, sin = tf.cos(angle) * scale, tf.sin(angle) * scale
    rot_scale = tf.stack([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    m = (
        _translation(out_w / 2.0 + shift[0], out_h / 2.0 + shift[1])
        @ rot_scale
        @ _translation(-crop_w / 2.0, -crop_h / 2.0)
    )
    mirror = tf.constant([[-1.0, 0.0, out_w], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return tf.where(flip, mirror @ m, m)


def _warp_image(image: tf.Tensor, m: tf.Tensor, size_hw: tuple[int, int]) -> tf.Tensor:
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
        output_shape=tf.constant(size_hw, tf.int32),
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
    )
    return out[0]


def _visible_weights(kps: tf.Tensor, size_hw: tuple[int, int]) -> tf.Tensor:
    xy, v = kps[:, :2], kps[:, 2]
    limit = tf.constant([float(size_hw[1]), float(size_hw[0])])
    inside = tf.reduce_all((xy >= 0.0) & (xy < limit), axis=-1)
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
    size = cfg.model.input_shape
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


def center_offset(cfg: Config) -> tuple[int, int]:
    """(x, y) of the central input window inside the stored crop."""
    (ch, cw), (h, w) = cfg.data.crop_shape, cfg.model.input_shape
    return (cw - w) // 2, (ch - h) // 2


def center_transform(cfg: Config, ex: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Deterministic: take the central input window (exact integer shift)."""
    (h, w), (ox, oy) = cfg.model.input_shape, center_offset(cfg)
    image = tf.cast(ex["image"][oy : oy + h, ox : ox + w], tf.float32)
    image = tf.ensure_shape(image, [h, w, 3])
    kps = ex["keypoints"]
    xy = kps[:, :2] - tf.constant([float(ox), float(oy)])
    kps = tf.concat([xy, kps[:, 2:]], axis=-1)
    return image, xy, _visible_weights(kps, (h, w))


def make_targets(cfg: Config, keypoints: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    heat = render_heatmaps(
        keypoints,
        weights,
        cfg.model.heatmap_shape,
        cfg.model.heatmap_stride,
        cfg.train.heatmap_sigma,
        cfg.model.udp,
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
    finetune = cfg.data.finetune_dirs if limit is None else []
    if limit is not None:
        # Read shards one after another so "the first N records" is the same
        # set in every pipeline. Parallel reads interleave shards, and with
        # deterministic=False (training) that order changes between runs.
        ds = tf.data.TFRecordDataset(shard_files(cfg, split)).take(limit).cache()
        if training:
            ds = ds.shuffle(cfg.train.shuffle_buffer).repeat()
    elif finetune and training:
        exercise = _record_dataset(shard_files(cfg, split, finetune), shuffle=True)
        coco = _record_dataset(shard_files(cfg, split), shuffle=True)
        r = cfg.data.finetune_ratio
        ds = tf.data.Dataset.sample_from_datasets(
            [d.shuffle(cfg.train.shuffle_buffer).repeat() for d in (exercise, coco)],
            weights=[r, 1.0 - r],
            seed=cfg.train.seed,
        )
    else:
        ds = _record_dataset(shard_files(cfg, split, finetune or None), shuffle=training)
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
    offset = [float(v) for v in center_offset(cfg)]

    def _map(serialized):
        ex = parse_example(serialized)
        image, _, _ = center_transform(cfg, ex)
        meta = {
            "image_id": ex["image_id"],
            "crop_to_orig": ex["crop_to_orig"],
            "offset": tf.constant(offset),
        }
        return image, meta

    ds = _record_dataset(shard_files(cfg, "val"), shuffle=False)
    return ds.map(_map, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
