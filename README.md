# Gym Posture Detector: Video Harness

This branch holds the video harness for the gym posture detector. It is a work in progress.

The pose estimation model (Stage 1) is [ViTPose-B](https://github.com/ViTAE-Transformer/ViTPose) (Apache-2.0), ported to Keras and fine-tuned on gym exercise footage. Its conversion, fine-tuning code and results are on the [`vitpose`](https://github.com/twilight0963/FormSenseAI/tree/vitpose) branch. The original from-scratch PoseNet is on the [`training`](https://github.com/twilight0963/FormSenseAI/tree/training) branch and still runs in the harness.

## Video harness

A tkinter player that runs the exported pose model on each frame of a webcam or video and overlays the stick figure on the footage. It is self-contained: it carries its own copy of the crop and heatmap-decoding code, so the training package is not needed. It loads either export format:

- **SavedModel directory** (`<run>/export/saved_model`, from `python -m posenet.export`): runs on the GPU when TensorFlow sees one. Use this for ViTPose; a ViT-B pass is too slow on the CPU.
- **TFLite file** (`*.tflite`): runs on the CPU, e.g. the original PoseNet (`runs/<run>/export/posenet_fp16.tflite`).

The input size and heatmap stride come from the model. The heatmap convention (ViTPose uses UDP) comes from the training run's `config.yaml`, found next to the export.

```bash
python -m venv .venv && source .venv/bin/activate   # fish: source .venv/bin/activate.fish
pip install -r requirements-dev.txt
python -m harness                       # reads config.json
python -m harness --config other.json
```

### config.json

| Key | Default | Meaning |
|---|---|---|
| `model_path` | `pretrained/vitpose_b_simple/export/saved_model` | SavedModel directory or `.tflite` file (relative paths resolve against the config file) |
| `use_webcam` | `false` | Use a camera; falls back to the video if the camera can't be opened |
| `webcam_index` | `0` | OpenCV camera index |
| `video_path` | `""` | Video to play; empty or missing opens a file picker at startup |
| `loop_video` | `true` | Restart the video when it ends |
| `flip_test` | `true` | Average with a mirrored pass (more accurate, twice the cost) |
| `score_threshold` | `0.3` | Minimum keypoint score to draw |
| `track` | `true` | Crop around the previous frame's pose instead of re-searching the whole frame |
| `num_threads` | `4` | TFLite CPU threads (TFLite models only) |
| `heatmap_coords` | `"auto"` | `auto` reads it from the run's `config.yaml`; or force `center` / `udp` |
| `window_width`, `window_height` | `1280`, `720` | Initial window size |

**Controls:** Space plays/pauses, R restarts, S toggles the skeleton. The status bar shows the backend, inference time, mean keypoint score and display FPS. If inference is slower than the video's frame rate, playback slows down rather than dropping frames.

**Finding the person:** with no previous pose to track, the frame is searched with model-shaped windows (several full-height windows across a wide frame), the most confident one is kept, and the model runs again on a crop around its keypoints. While tracking, each frame is cropped around the previous pose.

### Layout

```
config.json        harness settings
harness/
  config.py        JSON → HarnessConfig
  pose.py          PoseEstimator: TFLite / SavedModel backends, crops, decoding, window search, tracking
  draw.py          stick-figure overlay
  source.py        webcam / video / file picker
  app.py           Tk window + inference worker thread
tests/             pytest
```
