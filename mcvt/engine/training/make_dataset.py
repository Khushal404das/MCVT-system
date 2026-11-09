"""Build a vehicle-detection training set from the project's own cameras.

No hand-labelled data exists for these cameras, so a large, accurate teacher
model (YOLOv8m at high resolution) labels frames sampled from every camera
("pseudo-labels"). A small, fast student (YOLOv8n) is then fine-tuned on them
(train_detector.py), adapting it to these exact viewpoints, lenses and lighting.

Usage:  python training/make_dataset.py [--per-camera 120] [--teacher engine/models/yolov8m.pt]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcvt.config import load_config  # noqa: E402

CLASSES = ["car", "motorcycle", "bus", "truck"]  # student class order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--teacher", default=None, help="teacher weights (default: engine/models/yolov8m.pt)")
    ap.add_argument("--teacher-imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--per-camera", type=int, default=120)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = load_config(args.config)
    teacher_path = Path(args.teacher) if args.teacher else cfg.path("detection.model").parent / "yolov8m.pt"
    teacher = YOLO(str(teacher_path))
    t_ids = {i: n for i, n in teacher.names.items() if n in CLASSES}
    out = Path(args.out) if args.out else cfg.path("paths.storage") / "training" / "vehicles"
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(7)
    total, boxes = 0, 0
    for spec in cfg.camera_specs():
        src = spec.analysis_source or spec.source  # train on the same stream the engine analyses
        cap = cv2.VideoCapture(src)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            print(f"{spec.id}: cannot read frame count, skipped")
            continue
        # spread samples over the whole recording, with a little jitter
        step = n / args.per_camera
        idxs = sorted({min(n - 1, int(i * step + rng.uniform(0, step * 0.5))) for i in range(args.per_camera)})
        kept = 0
        for k, idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            res = teacher.predict(frame, imgsz=args.teacher_imgsz, conf=args.conf, classes=list(t_ids), verbose=False)[0]
            lines = []
            for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
                cls = CLASSES.index(t_ids[int(c)])
                lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            split = "val" if rng.random() < args.val_fraction else "train"
            name = f"{spec.id}_{idx:07d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
            kept += 1
            boxes += len(lines)
            if k % 20 == 0:
                print(f"{spec.id}: {k}/{len(idxs)} frames", flush=True)
        cap.release()
        total += kept
        print(f"{spec.id}: {kept} frames labelled", flush=True)

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8",
    )
    print(f"Done: {total} frames, {boxes} vehicle boxes -> {yaml_path}")


if __name__ == "__main__":
    main()
