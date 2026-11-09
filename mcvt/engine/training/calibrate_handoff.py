"""Measure how similar the SAME car looks in DIFFERENT cameras (for hand-off tuning).

Samples the first minutes of every camera, keeps moving cars, embeds them with
the ResNet-IBN ReID model and prints the most similar cross-camera pairs that
occur close in time, with crop pictures for checking by eye.

Usage:  python training/calibrate_handoff.py [--minutes 6] [--step 1.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcvt.config import load_config  # noqa: E402
from mcvt.detector import VehicleDetector  # noqa: E402
from mcvt.reid.extractor import FeatureExtractor  # noqa: E402


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=6)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--window", type=float, default=60)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    cfg = load_config()
    weights = args.weights or str(cfg.path("detection.model").parent / "yolov8s.pt")
    det = VehicleDetector(weights, confidence=0.45, imgsz=640)
    fe = FeatureExtractor(cfg.path("reid.weights"), cfg.get("reid.input_size", [256, 256]))
    out = cfg.path("paths.storage") / "training" / "handoff"
    out.mkdir(parents=True, exist_ok=True)

    items = []  # (cam, t, crop, box)
    for spec in cfg.camera_specs():
        cap = cv2.VideoCapture(spec.analysis_source or spec.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        prev_boxes: list[list[float]] = []
        t = 0.0
        while t < args.minutes * 60:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok:
                break
            boxes = []
            for d in det.detect(frame):
                b = [d.x1, d.y1, d.x2, d.y2]
                boxes.append(b)
                if any(iou(b, p) > 0.6 for p in prev_boxes):
                    continue  # same place as a moment ago -> parked
                if d.x2 - d.x1 < 40 or d.y2 - d.y1 < 30 or d.label not in ("car", "truck", "bus"):
                    continue
                items.append((spec.id, t, frame[int(d.y1):int(d.y2), int(d.x1):int(d.x2)].copy(), b))
            prev_boxes = boxes
            t += args.step
        cap.release()
        print(f"{spec.id}: {sum(1 for i in items if i[0] == spec.id)} moving-car crops", flush=True)

    crops = [i[2] for i in items]
    feats = np.concatenate([fe.extractFeatures(crops[k:k + 16]) for k in range(0, len(crops), 16)]) \
        if crops else np.zeros((0, 2048), np.float32)
    sims = feats @ feats.T
    cross, same = [], []
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            dt = abs(items[a][1] - items[b][1])
            if items[a][0] != items[b][0] and dt <= args.window:
                cross.append((float(sims[a, b]), a, b))
            elif items[a][0] == items[b][0] and 0 < dt <= 3:
                same.append(float(sims[a, b]))
    cross.sort(reverse=True)
    print(f"\nSame camera, <=3 s apart (mostly the same car): median {np.median(same):.2f}, p10 {np.percentile(same, 10):.2f}"
          if same else "\nno same-camera pairs")
    print(f"Cross-camera pairs within {args.window:.0f} s: {len(cross)}; top 15:")
    for k, (s, a, b) in enumerate(cross[:15]):
        A, B = items[a], items[b]
        print(f"  {s:.3f}  {A[0]}@{A[1]:6.1f}s  <->  {B[0]}@{B[1]:6.1f}s")
        h = 160
        pa = cv2.resize(A[2], (int(A[2].shape[1] * h / A[2].shape[0]), h))
        pb = cv2.resize(B[2], (int(B[2].shape[1] * h / B[2].shape[0]), h))
        pair = np.hstack([pa, np.zeros((h, 10, 3), np.uint8), pb])
        cv2.putText(pair, f"{s:.2f} {A[0]}-{B[0]}", (5, 20), 0, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(out / f"pair_{k:02d}.jpg"), pair)
    if cross:
        vals = [c[0] for c in cross]
        print(f"Cross-camera similarity distribution: median {np.median(vals):.2f}, p90 {np.percentile(vals, 90):.2f}, "
              f"max {max(vals):.2f}")


if __name__ == "__main__":
    main()
