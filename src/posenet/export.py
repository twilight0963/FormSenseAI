"""Export a trained model to SavedModel and TFLite (FP16 / INT8).

Each TFLite model is checked against the Keras model on real validation crops.

Example:
    python -m posenet.export --model runs/<run>/best.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from posenet.data.pipeline import build_dataset
from posenet.inference import load_config_for, load_inference_model


def run_tflite(path: Path, images: np.ndarray) -> np.ndarray:
    interp = tf.lite.Interpreter(model_path=str(path))
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.resize_tensor_input(inp["index"], images.shape)
    interp.allocate_tensors()
    interp.set_tensor(inp["index"], images.astype(inp["dtype"]))
    interp.invoke()
    return interp.get_tensor(out["index"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a pose model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", help="defaults to <run>/export")
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--variants", nargs="*", default=["fp16", "int8"], choices=["fp16", "int8"])
    args = parser.parse_args()

    cfg = load_config_for(args.model)
    model = load_inference_model(args.model, cfg)
    out_dir = Path(args.out_dir or Path(args.model).parent / "export")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = out_dir / "saved_model"
    model.export(str(saved))
    print(f"SavedModel → {saved}")
    if not args.variants:  # TFLite calibration and checks need the val records; skip them
        return

    val = build_dataset(cfg, "val", training=False)
    calib = [x.numpy() for x, _ in val.take(args.calibration_batches)]
    reference_in = calib[0][:16]
    reference_out = model.predict(reference_in, verbose=0)

    def representative():
        for batch in calib:
            for img in batch:
                yield [img[None].astype(np.float32)]

    variants = {"fp16": tf.float16, "int8": tf.int8}
    for name in args.variants:
        dtype = variants[name]
        conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if dtype == tf.float16:
            conv.target_spec.supported_types = [tf.float16]
        else:
            conv.representative_dataset = representative
        path = out_dir / f"{cfg.model.arch}_{name}.tflite"
        path.write_bytes(conv.convert())

        pred = run_tflite(path, reference_in)
        err = float(np.max(np.abs(pred - reference_out)))
        peak = np.argmax(pred.reshape(len(pred), -1, pred.shape[-1]), axis=1)
        ref_peak = np.argmax(reference_out.reshape(len(pred), -1, pred.shape[-1]), axis=1)
        agree = float(np.mean(peak == ref_peak))
        print(
            f"TFLite {name} → {path} ({path.stat().st_size / 1e6:.1f} MB); "
            f"max |Δheatmap| {err:.4f}, peak agreement {agree:.1%}"
        )


if __name__ == "__main__":
    main()
