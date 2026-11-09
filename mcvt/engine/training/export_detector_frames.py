"""Export frames from this project's cameras for fine-tuning the vehicle detector on a GPU
(Kaggle notebook engine/training/kaggle_finetune_detector.py).

Problem it targets: a vehicle that is already in view (small, far away, entering the frame) gets
no box until it is close, because the detector misses it while it is small. So most frames are
taken exactly where that happens: from the stored sightings of moving vehicles, at the moment
each vehicle was first detected and 0.5-4 s BEFORE it (when it was in view but still missed).
The rest are spread evenly over each video (parked cars, empty scenes, other lighting).
No hand labels: on the GPU a large teacher (YOLOv8x at 1280 px, test-time augmentation) labels
every frame, and YOLOv8n (the engine's real-time detector) is trained on those labels.

Usage:  python engine/training/export_detector_frames.py [--per-camera 300]
Output: storage/training/detector_kaggle/frames/*.jpg  (+ frames.json)
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcvt.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-camera", type=int, default=300)
    ap.add_argument("--arrival-share", type=float, default=0.7, help="share of frames taken around vehicle arrivals")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out) if args.out else cfg.path("paths.storage") / "training" / "detector_kaggle"
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(cfg.path("paths.database")))
    rng = random.Random(11)
    meta = []
    for spec in cfg.camera_specs():
        src = spec.analysis_source or spec.source
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            print(f"{spec.id}: cannot read {src}, skipped")
            continue
        dur = n / fps
        # first detections of moving vehicles on this camera (same media file only)
        rows = db.execute("SELECT start_offset FROM sightings WHERE camera_id=? AND media=? AND start_offset IS NOT NULL",
                          (spec.id, str(Path(src)))).fetchall()
        if not rows:  # media path stored differently (relative / other separators)
            rows = db.execute("SELECT start_offset FROM sightings WHERE camera_id=? AND start_offset IS NOT NULL",
                              (spec.id,)).fetchall()
        starts = sorted({round(float(r[0]), 1) for r in rows if 0 < float(r[0]) < dur})
        n_arr = min(len(starts) * 3, int(args.per_camera * args.arrival_share))
        times = []
        for i in range(n_arr):
            t0 = rng.choice(starts)
            times.append(("arrival", max(0.0, t0 - rng.uniform(0.5, 4.0)) if i % 3 else t0))
        step = dur / max(1, args.per_camera - n_arr)
        times += [("even", min(dur - 0.1, i * step + rng.uniform(0, step))) for i in range(args.per_camera - n_arr)]
        times.sort(key=lambda x: x[1])
        kept, last = 0, -9.0
        for kind, t in times:
            if t - last < 0.2:
                continue  # practically the same frame
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            last = t
            name = f"{spec.id}_{int(t * 1000):08d}.jpg"
            cv2.imwrite(str(frames_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            meta.append({"file": name, "camera": spec.id, "t": round(t, 2), "kind": kind,
                         "w": frame.shape[1], "h": frame.shape[0]})
            kept += 1
        cap.release()
        print(f"{spec.id}: {kept} frames ({sum(1 for m in meta if m['camera'] == spec.id and m['kind'] == 'arrival')} "
              f"around {len(starts)} vehicle arrivals) from {Path(src).name}", flush=True)
    (out / "frames.json").write_text(json.dumps(meta), encoding="utf-8")
    size = sum(p.stat().st_size for p in frames_dir.glob("*.jpg")) / 1e6
    print(f"Done: {len(meta)} frames, {size:.0f} MB -> {frames_dir}")


if __name__ == "__main__":
    main()
