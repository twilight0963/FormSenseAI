# Gym Posture Detector: Stage 1, Pose Estimation

This is a lightweight, real-time **single-person 2D pose estimator** written in TensorFlow/Keras. The CNN is built from scratch out of `Conv2D`, `DepthwiseConv2D`, `BatchNormalization` and similar layers, with no pretrained backbones. It is trained on **COCO Keypoints 2017** and predicts 17 keypoints.

| | |
|---|---|
| Input | 192×192 RGB, values in [0, 255] (normalization happens inside the model) |
| Output | 48×48×17 heatmaps decoded to (x, y, score) per keypoint |
| Backbone | MobileNetV2-style inverted residuals (strides 4/8/16/32) |
| Decoder | FPN-style top-down fusion to stride 4, depthwise-separable refinement |
| Size | 1.94M parameters; 3.8 MB as fp16 TFLite |

## Results

Reference run: 140 epochs, about 6.3 h on an RTX 3060 Laptop GPU (6 GB), using the default config.

**COCO val2017 keypoints** (ground-truth person boxes, flip test, official `pycocotools` OKS evaluation):

| AP | AP50 | AP75 | AP (medium) | AP (large) | AR |
|---|---|---|---|---|---|
| 0.606 | 0.853 | 0.679 | 0.582 | 0.642 | 0.640 |

PCK@0.2 on val crops: 0.840.

**Speed** (batch 1 unless noted; RTX 3060 Laptop GPU, Ryzen 9 5900HX):

| Backend | p50 latency | Throughput |
|---|---|---|
| GPU, Keras fp32 | 2.39 ms | 418 img/s |
| GPU, Keras fp16, batch 32 | 9.32 ms | 3,433 img/s |
| CPU, TFLite fp16 (4 threads) | 2.80 ms | 358 img/s |
| CPU, TFLite int8 (4 threads) | 2.62 ms | 382 img/s |
| CPU, Keras fp32 | 31.8 ms | 31 img/s |

Each latency is one network forward pass. Flip test doubles it. With the two-pass smart crop, `infer` runs up to four passes per image.

**Export accuracy** compared with the Keras model, measured on 1,024 val crops:

| Format | Size | PCK@0.2 | Keypoint shift (median / p90) |
|---|---|---|---|
| Keras fp32 | — | 0.8404 | — |
| TFLite fp16 | 3.8 MB | 0.8404 | 0.01 / 0.04 px |
| TFLite int8 | 2.4 MB | 0.8346 | 0.48 / 3.11 px |

Use the **fp16** export by default: it matches the Keras model exactly and is about as fast as int8. Choose int8 only when model size matters.

## Setup

Requires Python 3.10, pinned via `.python-version` and pyenv.

```bash
python -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
pip install -r requirements-dev.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

`tensorflow[and-cuda]` installs the CUDA 12 and cuDNN libraries as pip wheels. Only an NVIDIA driver needs to be installed system-wide.

## Data

```bash
./scripts/download_coco.sh                                    # ~19 GB → data/coco/
python -m posenet.data.build_tfrecords                         # person crops → data/tfrecords/
python scripts/visualize_samples.py --out outputs/samples.png  # sanity-check augmentation
```

`download_coco.sh` can be resumed after an interruption. It uses `aria2c` with 16 connections when it is installed, because single connections to the COCO server are often throttled. If a download slows to a crawl partway through, stop the script and run it again to restart with fresh connections.

Preprocessing crops every person (enlarged box ×1.25, square) to a 256² JPEG once, in parallel. Training then reads about 3.5 GB of small crops instead of decoding full COCO images each epoch.
- **Train split:** instances with ≥5 keypoints and a box of at least 32×32 (134,214 instances).
- **Val split:** every non-crowd instance with ≥1 keypoint (6,352 instances), so AP is comparable to the literature.

## Train

```bash
python -m posenet.train --overfit 64 --epochs 150  # wiring check: pck and val_pck → 1.0
python -m posenet.train                            # full run → runs/<timestamp>/
python -m posenet.train --resume runs/<timestamp>  # continue after interruption
tensorboard --logdir runs
```

- **Overfit check:** train PCK climbs quickly. Val PCK stays near 0 for roughly the first 75–90 epochs, until the BatchNorm moving statistics catch up with the fast-changing weights, then jumps to 1.0. A short run with val PCK near 0 is expected and does not mean anything is wrong.
- **Run folder contents:** `best.keras` (best val PCK), `last.keras`, `history.csv` (per-epoch loss/PCK), TensorBoard logs and a copy of the config.
- **Speed settings:** mixed precision, XLA, multi-step device loops and a fully parallel `tf.data` pipeline. A full epoch takes about 2.7 min on an RTX 3060 Laptop GPU, using about 4.3 GB of VRAM at batch 64.
- **Augmentation:** a single affine transform (scale, rotation, shift, mirror with left/right label swap) applied identically to image and keypoints, plus color jitter.

## Usage

### Command line

```bash
python -m posenet.evaluate --model runs/<run>/best.keras         # COCO OKS AP (GT boxes, flip test)
python -m posenet.infer --model runs/<run>/best.keras --image photo.jpg [more.jpg ...]
python -m posenet.export --model runs/<run>/best.keras           # SavedModel + TFLite fp16/int8
python -m posenet.benchmark --model runs/<run>/best.keras        # latency / FPS
```

**`infer` options and output:**
- Options: `--out-dir` (default `outputs/`), `--threshold` (minimum score for drawing a keypoint, default 0.3) and `--no-flip` (single pass per crop, about twice as fast).
- `<name>_pose.png`: the skeleton overlay.
- `<name>_pose.json`: keypoint positions in original-image pixels, keyed by name, for example:

```json
{
  "nose":     {"x": 218.95, "y": 186.46, "score": 0.955},
  "left_eye": {"x": 231.94, "y": 176.14, "score": 0.979},
  ...
}
```

**`evaluate`** writes `val2017_metrics.json` and the raw COCO results file next to the model. **`export`** writes to `<run>/export/` and reports how closely each TFLite model matches the Keras output.

### Python

```python
import cv2
from posenet.inference import PoseEstimator
from posenet.keypoints import KEYPOINT_NAMES

estimator = PoseEstimator.from_path("runs/<run>/best.keras", flip_test=True)
image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)

pose = estimator.predict(image)                  # no person box: two-pass smart crop
# pose = estimator.predict(image, bbox=[x, y, w, h])  # known COCO-style person box

for name, (x, y), score in zip(KEYPOINT_NAMES, pose.keypoints, pose.scores):
    print(f"{name:15s} {x:7.1f} {y:7.1f} {score:.2f}")
```

Inference on an arbitrary photo needs no person detector. Pass 1 runs on the whole image. Pass 2 re-crops around the confident keypoints, matching the framing seen during training. If a person box is already known, for example from a tracker, pass it as `bbox` to skip pass 1.

### TFLite

The exported `.tflite` models take one `(1, 192, 192, 3)` float32 RGB crop with values in [0, 255]. They return `(1, 48, 48, 17)` heatmaps. Decode them with `posenet.decode.decode_heatmaps(heatmaps, stride=4, sigma=2.0)`, which gives coordinates in crop pixels.

## Development

```bash
pytest          # geometry, augmentation alignment, heatmap/decoding, model, loss/metrics, GPU regression
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
