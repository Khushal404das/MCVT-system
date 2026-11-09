"""Build a vehicle re-identification set from the project's own cameras (no hand labels).

One pass over every camera's video (the same stream the engine analyses):
YOLOv8 finds vehicles, the background model keeps only MOVING ones, and a simple
box tracker groups each vehicle's detections into a tracklet. For every tracklet up
to --crops crops of the whole vehicle (not cut by the frame edge, largest first,
spread over time) are saved. One tracklet = one identity inside its camera.

Cross-camera identities are added by train_reid_adapter.py from the cameras listed in
reid.overlapping_cameras (they watch the same place at the same time).

Usage:  python training/make_reid_dataset.py [--fps 3] [--minutes 0] [--crops 4]
Output: storage/training/reid/{crops/<camera>/*.jpg, tracklets.json}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcvt.camera import MotionMask  # noqa: E402
from mcvt.config import load_config  # noqa: E402
from mcvt.detector import VehicleDetector  # noqa: E402


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracklet:
    def __init__(self, tid: int, t: float, box, label: str):
        self.tid, self.t0, self.t1, self.label = tid, t, t, label
        self.boxes: list[tuple[float, list[float]]] = [(t, list(box))]
        self.velocity = np.zeros(2)
        self.crops: list[tuple[float, float, np.ndarray]] = []  # (area, t, crop)

    def predict(self, t: float) -> np.ndarray:
        x1, y1, x2, y2 = self.boxes[-1][1]
        d = self.velocity * (t - self.t1)
        return np.array([x1 + d[0], y1 + d[1], x2 + d[0], y2 + d[1]])

    def add(self, t: float, box) -> None:
        (tp, bp) = self.boxes[-1]
        if t > tp:
            c_new = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            c_old = np.array([(bp[0] + bp[2]) / 2, (bp[1] + bp[3]) / 2])
            self.velocity = 0.5 * self.velocity + 0.5 * (c_new - c_old) / (t - tp)
        self.boxes.append((t, list(box)))
        self.t1 = t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--fps", type=float, default=3.0, help="frames analysed per second of video")
    ap.add_argument("--minutes", type=float, default=0, help="0 = whole video")
    ap.add_argument("--crops", type=int, default=4, help="crops saved per tracklet")
    ap.add_argument("--min-detections", type=int, default=3)
    ap.add_argument("--max-gap", type=float, default=1.5, help="seconds a tracklet may go undetected")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cameras", default="", help="comma list (default: all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    out = Path(args.out) if args.out else cfg.path("paths.storage") / "training" / "reid"
    det = VehicleDetector(cfg.path("detection.model"), cfg.get("detection.confidence", 0.5),
                          cfg.get("detection.classes", ["car", "motorcycle", "bus", "truck"]),
                          cfg.get("detection.imgsz", 640), "cpu", cfg.get("detection.min_box_area_ratio", 0.0),
                          bool(cfg.get("detection.agnostic_nms", True)))
    motion_frac = float(cfg.get("detection.motion_box_fraction", 0.05))
    blob_frac = float(cfg.get("pipeline.motion_blob_fraction", 0.002))
    min_crop = int(cfg.get("reid.min_crop_size", 28)) * 2
    wanted = {c.strip() for c in args.cameras.split(",") if c.strip()}

    meta = {"created": time.time(), "fps": args.fps, "tracklets": []}
    for spec in cfg.camera_specs():
        if wanted and spec.id not in wanted:
            continue
        src = spec.analysis_source or spec.source
        skip = float(spec.extra.get("analysis_start_seconds" if spec.analysis_source else "start_seconds", 0.0))
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if skip:
            cap.set(cv2.CAP_PROP_POS_MSEC, skip * 1000)
        step = max(1, round(fps / args.fps))
        limit = n if not args.minutes else min(n, int((skip + args.minutes * 60) * fps))
        mm = MotionMask(int(cfg.get("detection.motion_mask_width", 320)),
                        float(cfg.get("detection.motion_history_seconds", 20)), fps / step)
        (out / "crops" / spec.id).mkdir(parents=True, exist_ok=True)
        active: list[Tracklet] = []
        done: list[Tracklet] = []
        next_id, idx, t_start = 0, int(skip * fps), time.time()
        while idx < limit:
            if not cap.grab():
                break
            idx += 1
            if idx % step:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = idx / fps - skip  # seconds from the start of this camera's playback
            mask = mm.apply(frame)
            for tr in [a for a in active if t - a.t1 > args.max_gap]:
                active.remove(tr)
                done.append(tr)
            if t < 8:
                continue  # background model warm-up
            ncc, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if ncc <= 1 or stats[1:, cv2.CC_STAT_AREA].max() / mask.size < blob_frac:
                continue
            h, w = frame.shape[:2]
            mh, mw = mask.shape[:2]
            moving = []
            for d in det.detect(frame):
                a, b = int(d.x1 / w * mw), int(d.y1 / h * mh)
                c, e = max(a + 1, int(d.x2 / w * mw)), max(b + 1, int(d.y2 / h * mh))
                roi = mask[max(0, b):e, max(0, a):c]
                if roi.size and (roi > 0).mean() >= motion_frac:
                    moving.append(d)
            # greedy association: IoU with the predicted box, else centre distance
            pairs = []
            for i, d in enumerate(moving):
                box = (d.x1, d.y1, d.x2, d.y2)
                diag = float(np.hypot(d.x2 - d.x1, d.y2 - d.y1))
                for tr in active:
                    p = tr.predict(t)
                    o = iou(box, p)
                    dist = float(np.hypot((box[0] + box[2] - p[0] - p[2]) / 2, (box[1] + box[3] - p[1] - p[3]) / 2))
                    if o >= 0.2 or dist <= 0.6 * diag:
                        pairs.append((-(o + 1.0 - dist / max(diag, 1.0)), i, tr))
            used_d, used_t = set(), set()
            assigned = {}
            for _s, i, tr in sorted(pairs, key=lambda p: p[0]):
                if i in used_d or id(tr) in used_t:
                    continue
                used_d.add(i)
                used_t.add(id(tr))
                assigned[i] = tr
            for i, d in enumerate(moving):
                box = (d.x1, d.y1, d.x2, d.y2)
                tr = assigned.get(i)
                if tr is None:
                    tr = Tracklet(next_id, t, box, d.label)
                    next_id += 1
                    active.append(tr)
                else:
                    tr.add(t, box)
                cut = d.x1 <= 3 or d.y1 <= 3 or d.x2 >= w - 3 or d.y2 >= h - 3
                cw, ch = d.x2 - d.x1, d.y2 - d.y1
                if cut or cw < min_crop or ch < min_crop:
                    continue
                crop = frame[int(d.y1):int(d.y2), int(d.x1):int(d.x2)].copy()
                tr.crops.append((cw * ch, t, crop))
                if len(tr.crops) > 12:
                    tr.crops.sort(key=lambda c: -c[0])
                    tr.crops = tr.crops[:12]
        done.extend(active)
        kept = 0
        for tr in done:
            if len(tr.boxes) < args.min_detections or not tr.crops:
                continue
            # largest crops, spread over time
            chosen = sorted(tr.crops, key=lambda c: -c[0])[: args.crops * 2]
            chosen = sorted(chosen, key=lambda c: c[1])
            if len(chosen) > args.crops:
                pick = np.linspace(0, len(chosen) - 1, args.crops).round().astype(int)
                chosen = [chosen[i] for i in sorted(set(pick))]
            files = []
            for k, (_a, tc, crop) in enumerate(chosen):
                rel = f"crops/{spec.id}/{spec.id}_{tr.tid:05d}_{k}.jpg"
                cv2.imwrite(str(out / rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                files.append({"file": rel, "t": round(tc, 2)})
            meta["tracklets"].append({"camera": spec.id, "tid": tr.tid, "label": tr.label,
                                      "t0": round(tr.t0, 2), "t1": round(tr.t1, 2),
                                      "detections": len(tr.boxes), "crops": files})
            kept += 1
        print(f"{spec.id}: {kept} moving-vehicle tracklets ({len(done)} raw) in {time.time() - t_start:.0f} s", flush=True)
        # one file per run, so several runs (e.g. one per camera group, in parallel) can be combined
        name = "tracklets" + ("_" + "_".join(sorted(wanted)) if wanted else "") + ".json"
        (out / name).write_text(json.dumps(meta, indent=1))
    print(f"Saved {len(meta['tracklets'])} tracklets to {out}")


if __name__ == "__main__":
    main()
