"""COCO val2017 keypoint evaluation (OKS AP/AR) using ground-truth person boxes.

Example:
    python -m posenet.evaluate --model runs/<run>/best.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from posenet.data.pipeline import build_eval_dataset, num_examples
from posenet.geometry import apply_affine
from posenet.inference import PoseEstimator

STAT_NAMES = ("AP", "AP50", "AP75", "AP_M", "AP_L", "AR", "AR50", "AR75", "AR_M", "AR_L")


def instance_score(scores: np.ndarray, threshold: float = 0.2) -> float:
    """Mean confidence of the keypoints the model is reasonably sure about."""
    sure = scores[scores > threshold]
    return float(sure.mean()) if sure.size else float(scores.mean())


def predict_val(estimator: PoseEstimator, batch_size: int) -> list[dict]:
    cfg = estimator.cfg
    results = []
    total = num_examples(cfg, "val")
    with tqdm(total=total, desc="val", unit="inst") as bar:
        for images, meta in build_eval_dataset(cfg, batch_size):
            heat = estimator.heatmaps(images.numpy())
            coords, scores = estimator.decode(heat)
            offsets = meta["offset"].numpy()
            mats = meta["crop_to_orig"].numpy().reshape(-1, 2, 3).astype(np.float64)
            for i, image_id in enumerate(meta["image_id"].numpy()):
                crop_xy = coords[i].astype(np.float64) + offsets[i]
                xy = apply_affine(crop_xy, mats[i])
                kps = np.concatenate([xy, scores[i][:, None]], axis=1)
                results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": 1,
                        "keypoints": [round(float(v), 3) for v in kps.ravel()],
                        "score": instance_score(scores[i]),
                    }
                )
            bar.update(len(offsets))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO keypoint evaluation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-flip", action="store_true")
    args = parser.parse_args()

    estimator = PoseEstimator.from_path(args.model, flip_test=not args.no_flip)
    cfg = estimator.cfg
    results = predict_val(estimator, args.batch_size)

    out_dir = Path(args.model).parent
    det_file = out_dir / "val2017_keypoints_results.json"
    det_file.write_text(json.dumps(results))

    gt = COCO(str(Path(cfg.data.coco_root) / "annotations" / "person_keypoints_val2017.json"))
    dt = gt.loadRes(str(det_file))
    ev = COCOeval(gt, dt, "keypoints")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stats = {name: float(v) for name, v in zip(STAT_NAMES, ev.stats, strict=True)}
    stats["flip_test"] = not args.no_flip
    (out_dir / "val2017_metrics.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
