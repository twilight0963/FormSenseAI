"""Train (or fine-tune) a pose model on person crops.

Examples:
    python -m posenet.train --config configs/default.yaml
    python -m posenet.train --resume runs/20261006-130000
    python -m posenet.train --overfit 64 --epochs 30      # wiring sanity check
    python -m posenet.train --config configs/vitpose_finetune.yaml --init pretrained/vitpose_b_simple/model.keras
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import keras
import tensorflow as tf

from posenet.config import Config
from posenet.data.pipeline import build_dataset, epoch_size
from posenet.keypoints import NUM_KEYPOINTS
from posenet.losses import HeatmapLoss
from posenet.metrics import HeatmapPCK
from posenet.model import build_model


def setup_runtime(cfg: Config) -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    keras.utils.set_random_seed(cfg.train.seed)
    if cfg.train.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")


@keras.saving.register_keras_serializable(package="posenet")
class LayerDecayAdamW(keras.optimizers.AdamW):
    """AdamW whose learning rate shrinks by ``layer_decay`` per transformer block below the head.

    Scaling gradients would not work here (Adam normalizes them away), so the
    per-variable rate is applied in ``update_step``.
    """

    def __init__(self, layer_decay: float = 1.0, depth: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.layer_decay = layer_decay
        self.depth = depth

    def build(self, var_list):
        from posenet.model.vitpose import block_index

        super().build(var_list)
        # update_step sees backend variables without a Keras path, so resolve scales here.
        self._lr_scales = [
            self.layer_decay ** (self.depth + 1 - block_index(v.path, self.depth)) for v in var_list
        ]

    def update_step(self, gradient, variable, learning_rate):
        scale = self._lr_scales[self._get_variable_index(variable)]
        super().update_step(gradient, variable, learning_rate * scale)

    def get_config(self):
        return {**super().get_config(), "layer_decay": self.layer_decay, "depth": self.depth}


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
    if cfg.model.arch.startswith("vitpose") and t.layer_decay != 1.0:
        from posenet.model.vitpose import VARIANTS

        opt = LayerDecayAdamW(
            layer_decay=t.layer_decay,
            depth=VARIANTS[cfg.model.arch][1],
            learning_rate=schedule,
            weight_decay=t.weight_decay,
        )
    else:
        opt = keras.optimizers.AdamW(learning_rate=schedule, weight_decay=t.weight_decay)
    opt.exclude_from_weight_decay(var_names=["bias", "gamma", "beta", "embedding"])
    return opt


def compile_model(cfg: Config, model: keras.Model, steps_per_epoch: int) -> None:
    model.compile(
        optimizer=make_optimizer(cfg, steps_per_epoch),
        loss=HeatmapLoss(NUM_KEYPOINTS),
        metrics=[HeatmapPCK(NUM_KEYPOINTS)],
        jit_compile=cfg.train.jit_compile,
        steps_per_execution=min(cfg.train.steps_per_execution, steps_per_epoch),
    )


def freeze_blocks(model: keras.Model, n: int) -> None:
    """Freeze the patch/position embedding and the first ``n`` transformer blocks (ViTPose)."""
    if n <= 0:
        return
    prefixes = ("patch_embed", "pos_embed", *(f"block{i}_" for i in range(n)))
    frozen = [layer.name for layer in model.layers if layer.name.startswith(prefixes)]
    for layer in model.layers:
        if layer.name in frozen:
            layer.trainable = False
    print(f"froze {len(frozen)} layers (embedding + {n} blocks)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PoseNet.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", help="existing run directory to continue")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--batch-size", type=int, help="override train.batch_size")
    parser.add_argument("--init", help="start from this model's weights (e.g. converted ViTPose)")
    parser.add_argument("--freeze-blocks", type=int, help="override train.freeze_blocks")
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
        if args.freeze_blocks is not None:
            cfg.train.freeze_blocks = args.freeze_blocks
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
        steps_per_epoch = epoch_size(cfg) // cfg.train.batch_size
    # Keep epochs aligned with the on-device loop length.
    spe = min(cfg.train.steps_per_execution, steps_per_epoch)
    steps_per_epoch = max(spe, steps_per_epoch // spe * spe)

    model = build_model(cfg.model)
    if args.init and not args.resume:
        model.set_weights(keras.models.load_model(args.init, compile=False).get_weights())
        print(f"initialized from {args.init}")
    freeze_blocks(model, cfg.train.freeze_blocks)
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
