"""Fine-tune the engine's YOLOv8n vehicle detector on this project's cameras (Kaggle GPU script).

Inputs (Kaggle dataset, made by engine/training/export_detector_frames.py):
  frames/*.jpg + frames.json   frames of all six cameras, most taken when a vehicle arrived and
                               0.5-4 s before (in view but still missed by the current detector)
  yolov8n_mcvt.pt              the detector the engine uses now (for the before/after comparison)

1. Teacher labels: YOLOv8x at 1280 px with test-time augmentation labels every frame
   (car / motorcycle / bus / truck) - far stronger on small, distant vehicles than YOLOv8n.
2. Student: YOLOv8n (COCO weights) trained on those labels at the engine's 640 px, so it stays
   real-time on the CPU. Validation = the last 15% of every camera's video (never trained on).
3. Before/after on the validation frames at the engine's settings (conf 0.50, 640 px):
   precision / recall, recall on SMALL vehicles (< 40 px tall) and on arrival frames.
Output (/kaggle/working): yolov8n_mcvt_v2.pt, detector_metrics.json, results.csv
"""

import glob
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.202"], check=True)
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp_det")  # not part of the saved output
TMP.mkdir(parents=True, exist_ok=True)
CLASSES = ["car", "motorcycle", "bus", "truck"]
COCO = {2: 0, 3: 1, 5: 2, 7: 3}

# ---------------------------------------------------------------- inputs
meta_path = glob.glob("/kaggle/input/**/frames.json", recursive=True)[0]
META = json.loads(Path(meta_path).read_text())
jpgs = glob.glob("/kaggle/input/**/C11_*.jpg", recursive=True)
if not jpgs:
    z = glob.glob("/kaggle/input/**/frames.zip", recursive=True)[0]
    zipfile.ZipFile(z).extractall(TMP / "unz")
    jpgs = glob.glob(str(TMP / "unz" / "**" / "C11_*.jpg"), recursive=True)
FRAMES = Path(jpgs[0]).parent
OLD = (glob.glob("/kaggle/input/**/yolov8n_mcvt.pt", recursive=True) or [None])[0]
print(f"{len(META)} frames in {FRAMES}; current detector: {OLD}; GPU: {torch.cuda.get_device_name(0)}", flush=True)

# validation = last 15% of each camera's time span
span = {}
for m in META:
    lo, hi = span.get(m["camera"], (1e9, -1e9))
    span[m["camera"]] = (min(lo, m["t"]), max(hi, m["t"]))
for m in META:
    lo, hi = span[m["camera"]]
    m["split"] = "val" if m["t"] >= lo + 0.85 * (hi - lo) else "train"

DS = TMP / "ds"
for s in ("train", "val"):
    (DS / "images" / s).mkdir(parents=True, exist_ok=True)
    (DS / "labels" / s).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. teacher labels
teacher = YOLO("yolov8x.pt")
n_boxes = 0
for i in range(0, len(META), 8):
    batch = META[i:i + 8]
    res = teacher.predict([str(FRAMES / m["file"]) for m in batch], imgsz=1280, conf=0.35, iou=0.6,
                          classes=list(COCO), agnostic_nms=True, augment=True, half=True, device=0, verbose=False)
    for m, r in zip(batch, res):
        h, w = r.orig_shape
        lines = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            if (x2 - x1) * (y2 - y1) < 0.0001 * w * h:
                continue  # a few pixels: not a vehicle anyone can see
            lines.append(f"{COCO[int(c)]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        m["gt"] = [[float(v) for v in l.split()[1:]] for l in lines]
        stem = Path(m["file"]).stem
        os.symlink(FRAMES / m["file"], DS / "images" / m["split"] / m["file"])
        (DS / "labels" / m["split"] / f"{stem}.txt").write_text("\n".join(lines))
        n_boxes += len(lines)
    if i % 200 == 0:
        print(f"teacher: {i}/{len(META)} frames", flush=True)
print(f"teacher labelled {len(META)} frames, {n_boxes} vehicles "
      f"(train {sum(m['split'] == 'train' for m in META)}, val {sum(m['split'] == 'val' for m in META)} frames)", flush=True)
del teacher
torch.cuda.empty_cache()
data_yaml = DS / "data.yaml"
data_yaml.write_text(f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n"
                     + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))

# ---------------------------------------------------------------- 2. student
EPOCHS = int(os.environ.get("MCVT_DET_EPOCHS", 150))
student = YOLO("yolov8n.pt")
student.train(data=str(data_yaml), epochs=EPOCHS, imgsz=640, batch=32, device=0, workers=4, patience=40,
              cos_lr=True, close_mosaic=15, mosaic=1.0, scale=0.5, degrees=0, fliplr=0.5, hsv_v=0.5,
              project=str(TMP / "runs"), name="student", exist_ok=True, plots=False, verbose=False)
best = TMP / "runs" / "student" / "weights" / "best.pt"
shutil.copy2(best, WORK / "yolov8n_mcvt_v2.pt")
shutil.copy2(TMP / "runs" / "student" / "results.csv", WORK / "results.csv")


# ---------------------------------------------------------------- 3. before / after
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def score(weights, conf=0.50):
    model = YOLO(str(weights))
    names = {i: n for i, n in model.names.items()}
    keep = [i for i, n in names.items() if n in CLASSES]
    tp = fp = fn = 0
    small_tp = small_n = arr_tp = arr_n = 0
    val = [m for m in META if m["split"] == "val"]
    for m in val:
        r = model.predict(str(FRAMES / m["file"]), imgsz=640, conf=conf, classes=keep, agnostic_nms=True,
                          device=0, verbose=False)[0]
        h, w = r.orig_shape
        pred = r.boxes.xyxy.tolist()
        gt = [[(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h] for cx, cy, bw, bh in m["gt"]]
        used = set()
        for p in pred:
            best_v, bi = 0.0, -1
            for i, g in enumerate(gt):
                if i not in used and (v := iou(p, g)) > best_v:
                    best_v, bi = v, i
            if best_v >= 0.5:
                tp += 1
                used.add(bi)
            else:
                fp += 1
        fn += len(gt) - len(used)
        for i, g in enumerate(gt):
            if g[3] - g[1] < 40:
                small_n += 1
                small_tp += i in used
            if m["kind"] == "arrival":
                arr_n += 1
                arr_tp += i in used
    out = {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "found": tp, "missed": fn, "false": fp,
           "small_recall": small_tp / max(1, small_n), "small_vehicles": small_n,
           "arrival_recall": arr_tp / max(1, arr_n), "arrival_vehicles": arr_n}
    v = model.val(data=str(data_yaml), imgsz=640, batch=16, device=0, verbose=False, plots=False,
                  project=str(TMP / "val"), name=Path(weights).stem, exist_ok=True)
    out["mAP50"], out["mAP50_95"] = float(v.box.map50), float(v.box.map)
    return out


metrics = {}
for label, w in (("current yolov8n_mcvt.pt", OLD), ("new yolov8n_mcvt_v2.pt", WORK / "yolov8n_mcvt_v2.pt")):
    if w:
        metrics[label] = score(w)
        m = metrics[label]
        print(f"== {label}: precision {m['precision']:.3f} recall {m['recall']:.3f} | small (<40 px) recall "
              f"{m['small_recall']:.3f} of {m['small_vehicles']} | arrival-frame recall {m['arrival_recall']:.3f} | "
              f"mAP50 {m['mAP50']:.3f} mAP50-95 {m['mAP50_95']:.3f}", flush=True)
(WORK / "detector_metrics.json").write_text(json.dumps(metrics, indent=2))
print("saved", WORK / "yolov8n_mcvt_v2.pt")
