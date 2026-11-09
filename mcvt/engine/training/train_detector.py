"""Fine-tune YOLOv8n on the camera dataset built by make_dataset.py.

Usage:  python training/train_detector.py [--epochs 30] [--imgsz 640] [--batch 8]
The best weights are copied to engine/models/yolov8n_mcvt.pt; point
detection.model in config.yaml at that file to use them.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcvt.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", default=None, help="starting weights (default: engine/models/yolov8n.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default="yolov8n_mcvt", help="run name under storage/training/runs")
    ap.add_argument("--out", default="yolov8n_mcvt.pt", help="file name written to engine/models")
    args = ap.parse_args()

    from ultralytics import YOLO

    from mcvt.detector import resolve_device

    cfg = load_config(args.config)
    models = cfg.path("detection.model").parent
    data = cfg.path("paths.storage") / "training" / "vehicles" / "data.yaml"
    if not data.exists():
        raise SystemExit("Run training/make_dataset.py first")
    base = Path(args.base) if args.base else models / "yolov8n.pt"
    runs = cfg.path("paths.storage") / "training" / "runs"
    model = YOLO(str(base))
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=resolve_device(args.device).replace("cuda:", ""), workers=2, project=str(runs),
                name=args.name, exist_ok=True, patience=10, plots=False, verbose=False,
                hsv_v=0.5, degrees=0, mosaic=1.0, close_mosaic=5)
    best = runs / args.name / "weights" / "best.pt"
    dest = models / args.out
    shutil.copy2(best, dest)
    print(f"Saved fine-tuned detector to {dest}")


if __name__ == "__main__":
    main()
