"""Train a GRU that turns a TRACK's sequence of ReID crop features into one vehicle signature
(sequence / video re-identification). Kaggle GPU script (also runs on CPU for a quick test).

Why: the engine identifies a track from the plain AVERAGE of its crop features. A car seen
front -> side -> rear while it turns, or small and blurred far away, gives crops that disagree,
and the average of those is a poor signature. A recurrent network reads the crops in time order,
learns which ones to trust (attention over the GRU states) and how the views of one car relate.

Inputs (Kaggle datasets, found under /kaggle/input):
  labels.json + crops/   this project's crops: camera, tracklet, vehicle id (across cameras), time
                         (khushaldas4/mcvt-reid-data, made by engine/training/export_colab_dataset.py)
  mcvt_reid_finetuned.pth + resnet_ibn.py   the engine's fine-tuned ResNet50-IBN (kept frozen)
  image_train/ image_test/                  VeRi-776 (vehicle_camera_frame.jpg = sequences per camera)

1. Features: every crop -> 2048-d ResNet50-IBN feature (+ a far-away copy: shrunk to 24-48 px).
2. Sequences: one tracklet = a time-ordered list of crop features; VeRi: one vehicle on one camera.
3. Model: Linear -> bidirectional GRU -> attention pooling, + the plain mean as a skip input
   -> 512-d signature. Loss: batch-hard triplet + ID cross-entropy (label smoothing), PK sampling
   with sequences of 1-8 crops from DIFFERENT cameras of one vehicle in a batch.
4. Evaluation (tracks it never trained on), same query/gallery for both methods:
   AVERAGE of features (the engine now) vs GRU, with 1, 2 and 4 crops per track:
   cross-camera rank-1 / mAP on our cameras (test split) and on VeRi-776 test.
   Thresholds for the engine: same-vehicle median and the 99th percentile of different vehicles.
Output (WORK): mcvt_reid_gru.pt, gru_metrics.json
"""

import glob
import json
import math
import os
import random
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ON_KAGGLE = os.path.isdir("/kaggle/input")
WORK = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("MCVT_WORK", "gru_work"))
WORK.mkdir(parents=True, exist_ok=True)
TMP = Path("/kaggle/temp_gru") if ON_KAGGLE else WORK / "tmp"
TMP.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 7
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
QUICK = bool(os.environ.get("MCVT_QUICK"))  # local smoke test: few images / steps


def find(pattern, roots=("/kaggle/input",)):
    for r in roots:
        hits = glob.glob(f"{r}/**/{pattern}", recursive=True)
        if hits:
            return hits[0]
    return None


ROOTS = ("/kaggle/input",) if ON_KAGGLE else tuple(os.environ.get("MCVT_ROOTS", ".").split(";"))
WEIGHTS = os.environ.get("MCVT_WEIGHTS") or find("mcvt_reid_finetuned.pth", ROOTS)
MODEL_PY = os.environ.get("MCVT_MODEL_PY") or find("resnet_ibn.py", ROOTS)
LABELS = os.environ.get("MCVT_LABELS") or find("labels.json", ROOTS)
if not LABELS:
    z = find("mcvt_reid_dataset.zip", ROOTS)
    zipfile.ZipFile(z).extractall(TMP / "ours")
    LABELS = str(next((TMP / "ours").rglob("labels.json")))
VERI_TRAIN = os.environ.get("MCVT_VERI_TRAIN") or next(iter(glob.glob(f"{ROOTS[0]}/**/image_train", recursive=True)), None)
VERI_TEST = os.environ.get("MCVT_VERI_TEST") or next(iter(glob.glob(f"{ROOTS[0]}/**/image_test", recursive=True)), None)
print(f"device {DEV} | weights {WEIGHTS} | model {MODEL_PY} | labels {LABELS} | VeRi {VERI_TRAIN} / {VERI_TEST}", flush=True)
assert WEIGHTS and MODEL_PY and LABELS, "fine-tuned weights, resnet_ibn.py and labels.json are needed"

sys.path.insert(0, str(Path(MODEL_PY).parent))
from resnet_ibn import ResNetIBNReID  # noqa: E402

# ------------------------------------------------------------------ 1. features
net = ResNetIBNReID()
missing, _ = net.load_fastreid_checkpoint(WEIGHTS)
assert not [m for m in missing if not m.endswith("num_batches_tracked")], missing[:5]
net.eval().to(DEV)
if DEV.type == "cuda":
    net.half()


def load(path, far=False):
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((64, 64, 3), np.uint8)
    if far:  # what a far-away car gives: few pixels, blurred
        h, w = img.shape[:2]
        s = random.uniform(24, 48) / max(h, w)
        if s < 1:
            img = cv2.resize(img, (max(4, int(w * s)), max(4, int(h * s))), interpolation=cv2.INTER_AREA)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_LINEAR)
    return img


class ImgSet(torch.utils.data.Dataset):
    def __init__(self, paths, far):
        self.paths, self.far = paths, far

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(load(self.paths[i], self.far)).permute(2, 0, 1)


@torch.inference_mode()
def features(paths, far=False):
    out = np.zeros((len(paths), 2048), np.float16)
    dl = torch.utils.data.DataLoader(ImgSet(paths, far), batch_size=128, num_workers=4 if ON_KAGGLE else 0)
    k, t = 0, time.time()
    for x in dl:
        x = x.to(DEV).float()
        f = net(x.half() if DEV.type == "cuda" else x)
        f = F.normalize(f.float(), dim=1).cpu().numpy()
        out[k:k + len(f)] = f
        k += len(f)
    print(f"  {len(paths)} images{' (far copies)' if far else ''}: {time.time() - t:.0f} s", flush=True)
    return out


# tracklet: {"key", "vid" (identity), "cam", "items": [feature index ...] in time order, "split"}
tracklets = []
all_paths = []

lab = json.loads(Path(LABELS).read_text())
root = Path(LABELS).parent
by_tr = defaultdict(list)
for c in lab["crops"]:
    by_tr[(c["camera"], c["tracklet"])].append(c)
tr_keys = list(by_tr)
if QUICK:  # small random sample of tracklets (all cameras) for a CPU smoke test
    tr_keys = random.sample(tr_keys, min(70, len(tr_keys)))
for cam, tid in tr_keys:
    cs = by_tr[(cam, tid)]
    cs.sort(key=lambda c: c["t"])
    idx = []
    for c in cs:
        idx.append(len(all_paths))
        all_paths.append(str(root / c["file"]))
    tracklets.append({"key": f"ours/{cam}/{tid}", "vid": f"ours_{cs[0]['vehicle']}", "cam": cam, "items": idx,
                      "split": cs[0].get("split", "train"), "src": "ours"})
n_ours = len(all_paths)


def veri_tracklets(folder, split, limit=None):
    groups = defaultdict(list)
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    if limit:
        files = files[:limit]
    for f in files:
        p = Path(f).stem.split("_")  # 0002_c002_00030600_0
        if len(p) < 3:
            continue
        groups[(p[0], p[1])].append((int(p[2]), f))
    for (vid, cam), fs in groups.items():
        fs.sort()
        idx = []
        for _fr, f in fs:
            idx.append(len(all_paths))
            all_paths.append(f)
        tracklets.append({"key": f"veri/{vid}/{cam}", "vid": f"veri_{vid}", "cam": cam, "items": idx,
                          "split": split, "src": "veri"})


if VERI_TRAIN:
    veri_tracklets(VERI_TRAIN, "train", 3000 if QUICK else None)
if VERI_TEST:
    veri_tracklets(VERI_TEST, "veri_test", 1500 if QUICK else None)
print(f"{len(tracklets)} tracklets, {len(all_paths)} crops ({n_ours} of our cameras)", flush=True)

FEAT = features(all_paths)
FAR = features(all_paths, far=True)
del net
torch.cuda.empty_cache() if DEV.type == "cuda" else None
FEAT_T = torch.from_numpy(FEAT.astype(np.float32))
FAR_T = torch.from_numpy(FAR.astype(np.float32))


# ------------------------------------------------------------------ 2./3. model + training
class SeqReID(nn.Module):
    """Sequence of crop features (B x L x 2048, L >= 1) -> one 512-d vehicle signature."""

    def __init__(self, d_in=2048, d_h=512, d_out=512):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d_h), nn.LayerNorm(d_h), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(d_h, d_h, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * d_h, 1)
        self.out = nn.Linear(2 * d_h + d_in, d_out)

    def forward(self, x, lengths):
        B, L, _ = x.shape
        mask = torch.arange(L, device=x.device)[None, :] < lengths[:, None]
        h = self.inp(x)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=L)
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(1)
        pooled = (a.unsqueeze(-1) * h).sum(1)
        mean = (x * mask.unsqueeze(-1)).sum(1) / lengths[:, None].clamp(min=1)
        return self.out(torch.cat([pooled, F.normalize(mean, dim=1)], 1))


train_tr = [t for t in tracklets if t["split"] == "train"]
ids = sorted({t["vid"] for t in train_tr})
id_index = {v: i for i, v in enumerate(ids)}
by_id = defaultdict(list)
for t in train_tr:
    by_id[t["vid"]].append(t)
multi = [v for v in ids if len(by_id[v]) >= 2]  # vehicles seen on 2+ tracklets (cameras)
print(f"training: {len(train_tr)} tracklets of {len(ids)} vehicles ({len(multi)} on 2+ tracklets)", flush=True)

MAX_L = 8


def sample_seq(t, far_p=0.3):
    n = len(t["items"])
    L = random.randint(1, min(MAX_L, n))
    s = random.randint(0, n - L)
    idx = t["items"][s:s + L]
    if n > L and random.random() < 0.3:  # sometimes non-contiguous crops (engine samples every few checks)
        idx = sorted(random.sample(t["items"], L), key=t["items"].index)
    far = torch.rand(L) < far_p
    x = torch.where(far[:, None], FAR_T[idx], FEAT_T[idx])
    return x


def batch(P=48, K=4):
    vids = random.sample(multi, min(P, len(multi)))
    xs, ys = [], []
    for v in vids:
        trs = by_id[v]
        pick = random.sample(trs, K) if len(trs) >= K else [random.choice(trs) for _ in range(K)]
        for t in pick:
            xs.append(sample_seq(t))
            ys.append(id_index[v])
    lengths = torch.tensor([len(x) for x in xs])
    X = torch.zeros(len(xs), int(lengths.max()), 2048)
    for i, x in enumerate(xs):
        X[i, :len(x)] = x
    return X.to(DEV), lengths.to(DEV), torch.tensor(ys, device=DEV)


def batch_hard(e, y, margin=0.3):
    d = torch.cdist(e, e)
    same = y[:, None] == y[None, :]
    pos = (d * same.float()).max(1).values
    neg = (d + same.float() * 1e4).min(1).values
    return F.relu(pos - neg + margin).mean()


model = SeqReID().to(DEV)
bnneck = nn.BatchNorm1d(512).to(DEV)
bnneck.bias.requires_grad_(False)
clf = nn.Linear(512, len(ids), bias=False).to(DEV)
params = list(model.parameters()) + list(bnneck.parameters()) + list(clf.parameters())
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=5e-4)
STEPS = 300 if QUICK else int(os.environ.get("MCVT_GRU_STEPS", 6000))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
t0 = time.time()
for step in range(STEPS):
    model.train(); bnneck.train()
    X, Ln, Y = batch()
    e = model(X, Ln)
    loss_t = batch_hard(F.normalize(e, dim=1), Y)
    loss_c = F.cross_entropy(clf(bnneck(e)), Y, label_smoothing=0.1)
    loss = loss_t + loss_c
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step(); sched.step()
    if step % 500 == 0 or step == STEPS - 1:
        print(f"step {step}/{STEPS}: triplet {loss_t.item():.3f} id {loss_c.item():.3f} ({time.time() - t0:.0f} s)", flush=True)


# ------------------------------------------------------------------ 4. evaluation
@torch.inference_mode()
def signatures(trs, L, method, far_first=False):
    """One signature per tracklet from its first L crops (the engine identifies a track from its
    first samples). far_first: the first crop is the far-away copy (car entering the view)."""
    model.eval()
    out = []
    for t in trs:
        idx = t["items"][:L]
        x = FEAT_T[idx].clone()
        if far_first:
            x[0] = FAR_T[idx[0]]
        if method == "average":
            out.append(F.normalize(x.mean(0), dim=0))
        else:
            out.append(F.normalize(model(x[None].to(DEV), torch.tensor([len(idx)], device=DEV))[0], dim=0).cpu())
    return torch.stack(out)


def retrieval(trs, L, method, far_first=False):
    """Cross-camera: each tracklet looks for the same vehicle among tracklets of OTHER cameras."""
    S = signatures(trs, L, method, far_first)
    sim = S @ S.T
    vids = [t["vid"] for t in trs]
    cams = [t["cam"] for t in trs]
    r1, aps, pos, neg = [], [], [], []
    for i in range(len(trs)):
        g = [j for j in range(len(trs)) if cams[j] != cams[i]]
        rel = [vids[j] == vids[i] for j in g]
        if not any(rel):
            continue
        s = sim[i, g].numpy()
        order = np.argsort(-s)
        hits = np.array(rel)[order]
        r1.append(float(hits[0]))
        prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        aps.append(float((prec * hits).sum() / hits.sum()))
        pos += [float(v) for v, r in zip(s, rel) if r]
        neg += [float(v) for v, r in zip(s, rel) if not r]
    CAL[(len(trs), L, method, far_first)] = (pos, neg)
    return {"rank1": float(np.mean(r1)) if r1 else None, "mAP": float(np.mean(aps)) if aps else None,
            "queries": len(r1), "same_median": float(np.median(pos)) if pos else None,
            "diff_p99": float(np.quantile(neg, 0.99)) if neg else None}


metrics = {}
CAL = {}
ours_test = [t for t in tracklets if t["src"] == "ours" and t["split"] == "test"]
veri_test = [t for t in tracklets if t["split"] == "veri_test"]
for name, trs in (("our cameras (test split)", ours_test), ("VeRi-776 test", veri_test)):
    if len(trs) < 4:
        continue
    for L in (1, 2, 4):
        for far_first in ((False, True) if name.startswith("our") else (False,)):
            key = f"{name} | {L} crop{'s' if L > 1 else ''}{' | first crop far away' if far_first else ''}"
            metrics[key] = {m: retrieval(trs, L, m, far_first) for m in ("average", "gru")}
            a, g = metrics[key]["average"], metrics[key]["gru"]
            if not g["queries"]:
                print(f"{key}: no cross-camera queries", flush=True)
                continue
            print(f"{key}: rank-1 {a['rank1']:.3f} -> {g['rank1']:.3f} | mAP {a['mAP']:.3f} -> {g['mAP']:.3f} | "
                  f"same median {g['same_median']:.2f}, different p99 {g['diff_p99']:.2f} ({g['queries']} queries)",
                  flush=True)

# Calibration for the engine: cross-camera similarities of same / different vehicles on our test
# tracks (2 crops, as the engine has when it gives an id), plain average vs GRU. The engine maps
# its thresholds (set in average units) to GRU units at the same share of different-vehicle pairs.
cal = {}
key2 = next((k for k in CAL if k[1] == 2 and not k[3] and k[0] == len(ours_test)), None)
if key2:
    for m in ("average", "gru"):
        pos, neg = CAL[(key2[0], 2, m, False)]
        cal[m] = {"pos": np.round(pos, 4).tolist(), "neg": np.round(neg, 4).tolist()}
torch.save({"state_dict": model.state_dict(), "config": {"d_in": 2048, "d_h": 512, "d_out": 512, "max_len": MAX_L},
            "trained": time.strftime("%Y-%m-%d %H:%M"), "vehicles": len(ids), "steps": STEPS,
            "calibration": cal},
           WORK / "mcvt_reid_gru.pt")
(WORK / "gru_metrics.json").write_text(json.dumps(metrics, indent=2))
print("saved", WORK / "mcvt_reid_gru.pt")
