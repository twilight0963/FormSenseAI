"""Measure model latency / throughput.

Example:
    python -m posenet.benchmark --model runs/<run>/best.keras
    python -m posenet.benchmark                       # untrained weights, architecture speed only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from posenet.config import Config
from posenet.inference import load_config_for, load_inference_model
from posenet.model import build_model


def _time(fn, iters: int, warmup: int = 10) -> np.ndarray:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.array(times) * 1000.0


def _report(label: str, ms: np.ndarray, batch: int) -> None:
    p50, p95 = np.percentile(ms, 50), np.percentile(ms, 95)
    print(f"{label:<28} p50 {p50:7.2f} ms  p95 {p95:7.2f} ms  {batch * 1000 / p50:8.1f} img/s")


def bench_keras(model, size: tuple[int, int], batch: int, device: str, iters: int) -> np.ndarray:
    with tf.device(device):
        # XLA cannot read variables placed on another device, so give the
        # benchmark its own copy of the weights on the target device.
        local = keras.models.clone_model(model)
        local.set_weights(model.get_weights())
        x = tf.random.uniform((batch, *size, 3), 0, 255)
        fn = tf.function(local, jit_compile=True)
        return _time(lambda: fn(x).numpy(), iters)


def bench_tflite(path: Path, size: tuple[int, int], iters: int, threads: int) -> np.ndarray:
    interp = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    x = np.random.default_rng(0).uniform(0, 255, (1, *size, 3)).astype(inp["dtype"])

    def run():
        interp.set_tensor(inp["index"], x)
        interp.invoke()

    return _time(run, iters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PoseNet.")
    parser.add_argument("--model", help="trained .keras model (default: random init)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4, help="TFLite CPU threads")
    args = parser.parse_args()

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    if args.model:
        cfg = load_config_for(args.model)
        model = load_inference_model(args.model, cfg)
        fp16 = load_inference_model(args.model, cfg, dtype="float16")
    else:
        cfg = Config.from_yaml(args.config)
        model = build_model(cfg.model)
        keras.mixed_precision.set_global_policy("mixed_float16")
        fp16 = build_model(cfg.model)
        keras.mixed_precision.set_global_policy("float32")
    size = cfg.model.input_shape
    print(f"{cfg.model.arch} {size[0]}x{size[1]}, {model.count_params() / 1e6:.2f}M params")

    if tf.config.list_physical_devices("GPU"):
        for batch in (1, 32):
            _report(f"GPU fp32 batch {batch}", bench_keras(model, size, batch, "/GPU:0", args.iters), batch)
            _report(f"GPU fp16 batch {batch}", bench_keras(fp16, size, batch, "/GPU:0", args.iters), batch)
    _report("CPU fp32 batch 1", bench_keras(model, size, 1, "/CPU:0", args.iters // 4), 1)

    if args.model:
        export_dir = Path(args.model).parent / "export"
        for name in ("fp16", "int8"):
            path = export_dir / f"{cfg.model.arch}_{name}.tflite"
            if path.exists():
                ms = bench_tflite(path, size, args.iters // 4, args.threads)
                _report(f"TFLite {name} ({args.threads} thr)", ms, 1)


if __name__ == "__main__":
    main()
