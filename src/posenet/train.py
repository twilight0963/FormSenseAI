"""Train PoseNet on COCO person crops.

Examples:
    python -m posenet.train --config configs/default.yaml
    python -m posenet.train --resume runs/20261006-130000
    python -m posenet.train --overfit 64 --epochs 30      # wiring sanity check
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import keras
import tensorflow as tf

from posenet.config import Config
from posenet.data.pipeline import build_dataset, num_examples
from posenet.keypoints import NUM_KEYPOINTS
from posenet.losses import HeatmapLoss
from posenet.metrics import HeatmapPCK
from posenet.model.network import build_posenet


def setup_runtime(cfg: Config) -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    keras.utils.set_random_seed(cfg.train.seed)
    if cfg.train.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")


def make_optimizer(cfg: Config, steps_per_epoch: int) -> keras.optimizers.Optimizer:
    t = cfg.train
    warmup = min(t.warmup_epochs * steps_per_epoch, t.epochs * steps_per_epoch - 1)
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=t.learning_rate * 0.01,
        decay_steps=t.epochs * steps_per_epoch - warmup,
        alpha=0.01,
        warmup_target=t.learning_rate,
        warmup_steps=warmup,
    )
    opt = keras.optimizers.AdamW(learning_rate=schedule, weight_decay=t.weight_decay)
    opt.exclude_from_weight_decay(var_names=["bias", "gamma", "beta"])
    return opt


def compile_model(cfg: Config, model: keras.Model, steps_per_epoch: int) -> None:
    model.compile(
        optimizer=make_optimizer(cfg, steps_per_epoch),
        loss=HeatmapLoss(NUM_KEYPOINTS),
        metrics=[HeatmapPCK(NUM_KEYPOINTS)],
        jit_compile=cfg.train.jit_compile,
        steps_per_execution=min(cfg.train.steps_per_execution, steps_per_epoch),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PoseNet.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", help="existing run directory to continue")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--batch-size", type=int, help="override train.batch_size")
    parser.add_argument(
        "--overfit",
        type=int,
        metavar="N",
        help="train and validate on the first N train records without augmentation",
    )
    args = parser.parse_args()

    if args.resume:
        run_dir = Path(args.resume)
        cfg = Config.from_yaml(run_dir / "config.yaml")
    else:
        cfg = Config.from_yaml(args.config)
        if args.epochs:
            cfg.train.epochs = args.epochs
        if args.batch_size:
            cfg.train.batch_size = args.batch_size
        suffix = f"-overfit{args.overfit}" if args.overfit else ""
        run_dir = Path(cfg.train.output_dir) / (time.strftime("%Y%m%d-%H%M%S") + suffix)
        run_dir.mkdir(parents=True, exist_ok=False)
        cfg.to_yaml(run_dir / "config.yaml")

    setup_runtime(cfg)

    if args.overfit:
        n = args.overfit
        cfg.train.batch_size = min(cfg.train.batch_size, n)
        train_ds = build_dataset(cfg, "train", training=True, limit=n, augment=False)
        val_ds = build_dataset(cfg, "train", training=False, limit=n)
        steps_per_epoch = max(1, n // cfg.train.batch_size) * 10
    else:
        train_ds = build_dataset(cfg, "train", training=True)
        val_ds = build_dataset(cfg, "val", training=False)
        steps_per_epoch = num_examples(cfg, "train") // cfg.train.batch_size
    # Keep epochs aligned with the on-device loop length.
    spe = min(cfg.train.steps_per_execution, steps_per_epoch)
    steps_per_epoch = max(spe, steps_per_epoch // spe * spe)

    model = build_posenet(cfg.model)
    compile_model(cfg, model, steps_per_epoch)
    model.summary(print_fn=lambda s, **_: print(s))

    callbacks = [
        keras.callbacks.BackupAndRestore(str(run_dir / "backup")),
        keras.callbacks.ModelCheckpoint(
            str(run_dir / "best.keras"), monitor="val_pck", mode="max", save_best_only=True
        ),
        keras.callbacks.ModelCheckpoint(str(run_dir / "last.keras")),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
        keras.callbacks.TensorBoard(str(run_dir / "logs"), update_freq="epoch"),
    ]
    if not args.overfit:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_pck", mode="max", patience=cfg.train.early_stopping_patience
            )
        )

    model.fit(
        train_ds,
        epochs=cfg.train.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
    )
    if (run_dir / "backup").exists():
        shutil.rmtree(run_dir / "backup")
    print(f"Done. Best model: {run_dir / 'best.keras'}")


if __name__ == "__main__":
    main()
