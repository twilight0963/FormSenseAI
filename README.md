# Gym Posture Detector: Stage 1, Pose Estimation

This is a lightweight, real-time **single-person 2D pose estimator** written in TensorFlow/Keras. The CNN is built from scratch out of `Conv2D`, `DepthwiseConv2D`, `BatchNormalization` and similar layers, with no pretrained backbones. It is trained on **COCO Keypoints 2017** and predicts 17 keypoints.

| | |
|---|---|
| Input | 192×192 RGB, values in [0, 255] (normalization happens inside the model) |
| Output | 48×48×17 heatmaps decoded to (x, y, score) per keypoint |
| Backbone | MobileNetV2-style inverted residuals (strides 4/8/16/32) |
| Decoder | FPN-style top-down fusion to stride 4, depthwise-separable refinement |
| Size | ~2.4M parameters |

## Setup

Requires Python 3.10, pinned via `.python-version` and pyenv.

```bash
python -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
pip install -r requirements-dev.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Data

```bash
./scripts/download_coco.sh                                    # ~19 GB → data/coco/
python -m posenet.data.build_tfrecords                         # person crops → data/tfrecords/
python scripts/visualize_samples.py --out outputs/samples.png  # sanity-check augmentation
```

Preprocessing crops every person (enlarged box ×1.25, square) to a 256² JPEG once, in parallel. Training then reads ~3 GB of small crops instead of decoding full COCO images each epoch.
- **Train split:** instances with ≥5 keypoints and a box of at least 32×32.
- **Val split:** every non-crowd instance with ≥1 keypoint, so AP is comparable to the literature.

## Train

```bash
python -m posenet.train --overfit 64 --epochs 20   # wiring check: pck → ~1.0
python -m posenet.train                            # full run → runs/<timestamp>/
python -m posenet.train --resume runs/<timestamp>  # continue after interruption
tensorboard --logdir runs
```

- **Speed settings:** mixed precision, XLA, multi-step device loops and a fully parallel `tf.data` pipeline.
- **Augmentation:** a single affine transform (scale, rotation, shift, mirror with left/right label swap) applied identically to image and keypoints, plus color jitter.

## Evaluate, infer, export

```bash
python -m posenet.evaluate --model runs/<run>/best.keras         # COCO OKS AP (GT boxes, flip test)
python -m posenet.infer --model runs/<run>/best.keras --image photo.jpg
python -m posenet.export --model runs/<run>/best.keras           # SavedModel + TFLite fp16/int8
python -m posenet.benchmark --model runs/<run>/best.keras        # latency / FPS
```

Inference on an arbitrary photo needs no person detector. Pass 1 runs on the whole image. Pass 2 re-crops around the confident keypoints, matching the framing seen during training.

## Development

```bash
pytest          # geometry, augmentation alignment, heatmap/decoding, model, loss/metrics
ruff check .
```

## Layout

```
configs/default.yaml     all hyperparameters
scripts/                 COCO download, sample visualization
src/posenet/
  geometry.py            crop/affine utilities (one coordinate convention everywhere)
  data/                  COCO parsing, TFRecord builder, tf.data pipeline, heatmaps
  model/                 layer blocks + network
  losses.py metrics.py decode.py inference.py
  train.py evaluate.py infer.py export.py benchmark.py
tests/
```
