"""Compare detectors on the held-out (val) frames of the camera dataset.

Reports precision / recall / F1 at IoU 0.5 against the teacher's labels, plus
speed, for each model given.

Usage:  python training/evaluate_detector.py engine/models/yolov8n.pt engine/models/yolov8n_mcvt.pt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcvt.config import load_config  # noqa: E402
from mcvt.detector import VehicleDetector  # noqa: E402


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    cfg = load_config()
    root = cfg.path("paths.storage") / "training" / "vehicles"
    images = sorted((root / "images" / "val").glob("*.jpg"))
    conf = float(cfg.get("detection.confidence", 0.5))
    print(f"{len(images)} validation frames, confidence {conf}")
    for weights in sys.argv[1:]:
        det = VehicleDetector(weights, confidence=conf, imgsz=640)
        tp = fp = fn = 0
        t = 0.0
        for img_path in images:
            frame = cv2.imread(str(img_path))
            h, w = frame.shape[:2]
            gt = []
            for line in (root / "labels" / "val" / f"{img_path.stem}.txt").read_text().splitlines():
                _, cx, cy, bw, bh = map(float, line.split())
                gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            t0 = time.perf_counter()
            pred = [[d.x1, d.y1, d.x2, d.y2] for d in det.detect(frame)]
            t += time.perf_counter() - t0
            used = set()
            for p in pred:
                best, bi = 0.0, -1
                for i, g in enumerate(gt):
                    if i not in used and (v := iou(p, g)) > best:
                        best, bi = v, i
                if best >= 0.5:
                    tp += 1
                    used.add(bi)
                else:
                    fp += 1
            fn += len(gt) - len(used)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        print(f"{Path(weights).name:22s} precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}  "
              f"({tp} found, {fn} missed, {fp} false)  {1000 * t / max(1, len(images)):.0f} ms/frame")


if __name__ == "__main__":
    main()
