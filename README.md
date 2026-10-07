# Gym Posture Detector: Video Harness

This branch holds the video harness for the gym posture detector. It is a work in progress.

The pose estimation model (Stage 1), along with its training code, results and Python/TFLite API, is on the [`training`](https://github.com/twilight0963/FormSenseAI/tree/training) branch.

## Video harness

A tkinter player that runs the exported pose model on each frame of a webcam or video and overlays the stick figure on the footage. It is self-contained: it loads the **TFLite export** from the `training` branch (`runs/<run>/export/posenet_fp16.tflite`) and carries its own copy of the crop and heatmap-decoding code, so the training package is not needed.

```bash
python -m venv .venv && source .venv/bin/activate   # fish: source .venv/bin/activate.fish
pip install -r requirements-dev.txt
python -m harness                       # reads config.json
python -m harness --config other.json
```

### config.json

| Key | Default | Meaning |
|---|---|---|
| `model_path` | `runs/20261006-164407/export/posenet_fp16.tflite` | Exported TFLite model (relative paths resolve against the config file) |
| `use_webcam` | `false` | Use a camera; falls back to the video if the camera can't be opened |
| `webcam_index` | `0` | OpenCV camera index |
| `video_path` | `""` | Video to play; empty or missing opens a file picker at startup |
| `loop_video` | `true` | Restart the video when it ends |
| `flip_test` | `true` | Average with a mirrored pass (more accurate, twice the cost) |
| `score_threshold` | `0.3` | Minimum keypoint score to draw |
| `track` | `true` | Crop around the previous frame's pose instead of re-searching the whole frame |
| `num_threads` | `4` | TFLite CPU threads |
| `window_width`, `window_height` | `1280`, `720` | Initial window size |

**Controls:** Space plays/pauses, R restarts, S toggles the skeleton. The status bar shows inference time, mean keypoint score and display FPS. If inference is slower than the video's frame rate, playback slows down rather than dropping frames.

### Layout

```
config.json        harness settings
harness/
  config.py        JSON → HarnessConfig
  pose.py          TFLite PoseEstimator (crop, decode, two-pass smart crop, tracking)
  draw.py          stick-figure overlay
  source.py        webcam / video / file picker
  app.py           Tk window + inference worker thread
tests/             pytest
```
