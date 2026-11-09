"""Writes engine/training/colab_finetune_reid.ipynb (run once after editing the cells below).

The notebook fine-tunes the whole ResNet50-IBN ReID network on a GPU (Google Colab or Kaggle)
with this project's crops (export_colab_dataset.py) and, optionally, VeRi-776.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_SRC = (HERE.parent / "mcvt" / "reid" / "resnet_ibn.py").read_text(encoding="utf-8")
MODEL_SRC = MODEL_SRC.split("from __future__ import annotations", 1)[1]

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md(r"""
# MCVT - fine-tune the vehicle ReID network (ResNet50-IBN + BNNeck) on a GPU

Trains the **whole** ResNet50-IBN (VeRi-776 weights) so that crops of **one vehicle** - other moments,
other cameras, far-away or close-up - get nearly the same features, and **different vehicles** are
pushed apart:

* **ID loss** (cross-entropy with label smoothing) on the BNNeck feature
* **batch-hard triplet loss** (margin 0.3) on the pooled feature, P x K batches (16 vehicles x 4 crops)
* augmentation: random **low resolution** (a car shrunk to 16-48 px, like far away on C41/C61), flip,
  crop/pad, colour jitter, random erasing

**Before you start**

1. *Colab:* Runtime -> Change runtime type -> **T4 GPU**. *Kaggle:* Settings -> Accelerator -> **GPU**.
2. You need two files from the project PC:
   * `D:\MCVT-system\mcvt\storage\training\colab\mcvt_reid_dataset.zip` (10 MB)
   * `D:\MCVT-system\mcvt\engine\models\veri_sbs_R50-ibn.pth` (190 MB)

   *Colab:* the first cell asks you to upload them. *Kaggle:* add them as a Dataset (+ Add Input) first.
3. Optional, more data: add the **VeRi-776** dataset (Kaggle: search "VeRi-776", it has an `image_train`
   folder). It is used automatically if found.
4. Run all cells (about 15-30 min on a T4). The last cell downloads **`mcvt_reid_finetuned.pth`**.

**Back on the PC:** copy it to `D:\MCVT-system\mcvt\engine\models\` and in `config.yaml` set
`reid.weights: engine/models/mcvt_reid_finetuned.pth` (and `reid.adapter: ""`), then restart the engine.
Use the threshold the evaluation cell recommends for `reid.similarity_threshold`.
""")

code(r"""
# 1) Files: upload (Colab) or find them in /kaggle/input (Kaggle) - or local paths for a test run
import os, glob, zipfile, json, math, random, time
from pathlib import Path

WORK = Path(os.environ.get("MCVT_WORK", "/content" if os.path.isdir("/content") else "/kaggle/working"
                           if os.path.isdir("/kaggle/working") else "./mcvt_work"))
WORK.mkdir(parents=True, exist_ok=True)

def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True) + glob.glob(str(WORK / "**" / pattern), recursive=True)
    return hits[0] if hits else None

DATA_ZIP = os.environ.get("MCVT_DATA_ZIP") or find("mcvt_reid_dataset.zip")
WEIGHTS = os.environ.get("MCVT_WEIGHTS") or find("veri_sbs_R50-ibn.pth")
ON_KAGGLE = os.path.isdir("/kaggle/input")
if ON_KAGGLE and not os.environ.get("MCVT_WORK"):
    WORK = Path("/kaggle/working")  # newer Kaggle images also have /content
    WORK.mkdir(parents=True, exist_ok=True)
LABELS0 = None if DATA_ZIP else find("labels.json")  # Kaggle unpacks uploaded zips itself
print("found:", DATA_ZIP or LABELS0, WEIGHTS)
if (not (DATA_ZIP or LABELS0) or not WEIGHTS) and not ON_KAGGLE and os.path.isdir("/content"):
    from google.colab import files
    print("Upload mcvt_reid_dataset.zip and veri_sbs_R50-ibn.pth")
    for name, data in files.upload().items():
        (WORK / name).write_bytes(data)
    DATA_ZIP = DATA_ZIP or find("mcvt_reid_dataset.zip")
    WEIGHTS = WEIGHTS or find("veri_sbs_R50-ibn.pth")
LABELS = None if DATA_ZIP else find("labels.json")  # Kaggle unpacks uploaded zips itself
assert (DATA_ZIP or LABELS) and WEIGHTS, "mcvt_reid_dataset.zip and veri_sbs_R50-ibn.pth are needed (see the top cell)"
if LABELS:
    DATA = Path(LABELS).parent
    DATA_ZIP = str(DATA)
else:
    DATA = WORK / "mcvt_data"
    if not (DATA / "labels.json").exists():
        zipfile.ZipFile(DATA_ZIP).extractall(DATA)
META = json.loads((DATA / "labels.json").read_text())
VERI = os.environ.get("MCVT_VERI") or next((d for d in glob.glob("/kaggle/input/**/image_train", recursive=True)), None)
print("data:", DATA_ZIP, "| weights:", WEIGHTS, "| VeRi-776:", VERI or "not used")
print(f"{len(META['crops'])} crops of {META['vehicles']} vehicles, overlapping cameras {META['overlapping_cameras']}")
""")

code(r"""
# 2) The model - same code as engine/mcvt/reid/resnet_ibn.py, so the weights load back into the engine
import torch
import torch.nn as nn
import torch.nn.functional as F
""" + MODEL_SRC + r"""

class Trainable(nn.Module):
    '''ResNetIBNReID + an ID classifier; returns (pooled feature, BNNeck feature, logits).'''
    def __init__(self, n_ids):
        super().__init__()
        self.reid = ResNetIBNReID()
        self.classifier = nn.Linear(2048, n_ids, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, images):
        x = (images - self.reid.pixel_mean) / self.reid.pixel_std
        pooled = self.reid.heads.pool_layer(self.reid.backbone(x))
        bn = self.reid.heads.bottleneck(pooled)[..., 0, 0]
        return pooled[..., 0, 0], bn, self.classifier(bn)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "(no GPU - very slow!)")
""")

code(r"""
# 3) Data: this project's crops (+ VeRi-776 if added) with augmentation
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as T

SIZE = 256
rows = META["crops"]
train_rows = [r for r in rows if r["split"] == "train"]
test_rows = [r for r in rows if r["split"] == "test"]
if os.environ.get("MCVT_TEST_LIMIT"):  # quick local check only
    keep_t = sorted({r["tracklet"] for r in test_rows})[: int(os.environ["MCVT_TEST_LIMIT"])]
    test_rows = [r for r in test_rows if r["tracklet"] in keep_t]
    train_rows = train_rows[: 4 * int(os.environ["MCVT_TEST_LIMIT"])]
items = [(str(DATA / r["file"]), f"m{r['vehicle']}") for r in train_rows]
if VERI:
    vfiles = sorted(glob.glob(os.path.join(VERI, "*.jpg")))
    by = {}
    for f in vfiles:
        by.setdefault(os.path.basename(f).split("_")[0], []).append(f)
    keep = sorted(by)[: int(os.environ.get("MCVT_VERI_IDS", 576))]
    for vid in keep:
        for f in by[vid][:24]:
            items.append((f, f"v{vid}"))
names = sorted({v for _, v in items})
idx_of = {v: i for i, v in enumerate(names)}
items = [(f, idx_of[v]) for f, v in items]
print(f"training images {len(items)}, identities {len(names)} (project {len({v for _, v in items if names[v][0]=='m'})})")

class LowRes:
    '''A car far away: shrink to 16-48 px, then back up (blurry), like C41/C61 views.'''
    def __init__(self, p=0.5, lo=16, hi=48):
        self.p, self.lo, self.hi = p, lo, hi
    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.randint(self.lo, self.hi) / max(w, h)
        if s >= 1:
            return img
        small = img.resize((max(4, int(w * s)), max(4, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

to255 = T.Lambda(lambda t: t * 255.0)
train_tf = T.Compose([LowRes(), T.Resize((SIZE, SIZE)), T.RandomHorizontalFlip(), T.Pad(10), T.RandomCrop(SIZE),
                      T.ColorJitter(0.25, 0.25, 0.2, 0.03), T.ToTensor(), T.RandomErasing(p=0.5, value=0.5), to255])
test_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), to255])

class Crops(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        f, y = self.items[i]
        return self.tf(Image.open(f).convert("RGB")), y

class PKSampler(torch.utils.data.Sampler):
    '''P identities x K images per batch (needed by the batch-hard triplet loss).'''
    def __init__(self, labels, P=16, K=4):
        self.by = {}
        for i, y in enumerate(labels):
            self.by.setdefault(y, []).append(i)
        self.ids, self.P, self.K = list(self.by), P, K
    def __iter__(self):
        ids = self.ids[:]
        random.shuffle(ids)
        for b in range(0, len(ids) - self.P + 1, self.P):
            for y in ids[b:b + self.P]:
                pool = self.by[y]
                yield from (random.sample(pool, self.K) if len(pool) >= self.K else random.choices(pool, k=self.K))
    def __len__(self):
        return (len(self.ids) // self.P) * self.P * self.K

P, K = int(os.environ.get("MCVT_P", 16)), 4
train_dl = torch.utils.data.DataLoader(Crops(items, train_tf), batch_size=P * K,
                                       sampler=PKSampler([y for _, y in items], P, K),
                                       num_workers=0 if os.name == "nt" else 2, drop_last=True,
                                       pin_memory=DEVICE == "cuda")
""")

code(r"""
# 4) Evaluation on footage the model was NOT trained on (later part of the videos)
@torch.no_grad()
def embed(model, files, lowres=False):
    model.eval()
    out = []
    rng = random.Random(0)
    for i in range(0, len(files), 64):
        ims = []
        for f in files[i:i + 64]:
            im = Image.open(f).convert("RGB")
            if lowres:  # deterministic far-away copy
                w, h = im.size
                s = rng.randint(18, 40) / max(w, h)
                if s < 1:
                    im = im.resize((max(4, int(w * s)), max(4, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
            ims.append(test_tf(im))
        x = torch.stack(ims).to(DEVICE)
        bn = model(x)[1] if isinstance(model, Trainable) else model(x)
        out.append(F.normalize(bn.float(), dim=1).cpu())
    return torch.cat(out).numpy()

def evaluate(model, title):
    files = [str(DATA / r["file"]) for r in test_rows]
    E, L = embed(model, files), embed(model, files, lowres=True)
    trk = np.array([r["tracklet"] for r in test_rows]); veh = np.array([r["vehicle"] for r in test_rows])
    cam = np.array([r["camera"] for r in test_rows])
    tids = sorted(set(trk))
    def tmean(M, t):
        v = M[trk == t].mean(0); return v / np.linalg.norm(v)
    M = {t: tmean(E, t) for t in tids}; LM = {t: tmean(L, t) for t in tids}
    tv = {t: veh[trk == t][0] for t in tids}; tc = {t: cam[trk == t][0] for t in tids}
    # cross-camera search between overlapping cameras + the live ID rule (best >= threshold)
    r1, rows_ = [], []
    for ca, cb in META["overlapping_cameras"]:
        gal = [t for t in tids if tc[t] == ca]
        if not gal:
            continue
        G = np.stack([M[t] for t in gal])
        for q in (t for t in tids if tc[t] == cb):
            s = G @ M[q]; truth = {j for j, g in enumerate(gal) if tv[g] == tv[q]}
            rows_.append((s, truth))
            if truth:
                r1.append(int(np.argmax(s)) in truth)
    # far-away view -> the sharp crops of all vehicles
    G = np.stack([M[t] for t in tids])
    lr1 = np.mean([tv[tids[int(np.argmax(G @ LM[q]))]] == tv[q] for q in tids])
    S = G @ G.T
    diff = np.array([S[a, b] for a in range(len(tids)) for b in range(a + 1, len(tids)) if tv[tids[a]] != tv[tids[b]]])
    same = np.array([float(M[t] @ LM[t]) for t in tids] +
                    [S[a, b] for a in range(len(tids)) for b in range(a + 1, len(tids)) if tv[tids[a]] == tv[tids[b]]])
    print(f"== {title}: cross-camera rank-1 {np.mean(r1):.2f} ({len(r1)} queries) | far-away->sharp rank-1 {lr1:.2f}")
    print(f"   different vehicles: 99% below {np.percentile(diff, 99):.2f} | same vehicle: 90% above "
          f"{np.percentile(same, 10):.2f}, median {np.median(same):.2f}")
    if not rows_:
        print("   (no cross-camera test queries in this subset)")
        return None
    best = None
    for thr in np.arange(0.30, 0.86, 0.05):
        c = dict(right=0, wrong=0, dup=0, newok=0)
        for s, truth in rows_:
            j = int(np.argmax(s))
            if s[j] >= thr: c["right" if j in truth else "wrong"] += 1
            else: c["dup" if truth else "newok"] += 1
        n = max(1, len(rows_))
        print(f"   threshold {thr:.2f}: right id {c['right']/n:4.0%}  WRONG id {c['wrong']/n:4.0%}  "
              f"duplicate {c['dup']/n:4.0%}  new id ok {c['newok']/n:4.0%}")
        score = (c["right"] + c["newok"]) / n - 2 * c["wrong"] / n  # a wrong id costs twice a duplicate
        if best is None or score > best[0]:
            best = (score, float(thr))
    print(f"   -> suggested reid.similarity_threshold: {best[1]:.2f}")
    return best[1]

base = ResNetIBNReID()
base.load_fastreid_checkpoint(WEIGHTS)
base = base.to(DEVICE)
evaluate(base, "BEFORE fine-tuning (VeRi-776 weights)")
""")

code(r"""
# 5) Fine-tune: ID loss (label smoothing) + batch-hard triplet loss
EPOCHS = int(os.environ.get("MCVT_EPOCHS", 40))
model = Trainable(len(names)).to(DEVICE)
model.reid.load_fastreid_checkpoint(WEIGHTS)
for p in list(model.reid.backbone.conv1.parameters()) + list(model.reid.backbone.bn1.parameters()) + \
         list(model.reid.backbone.layer1.parameters()):
    p.requires_grad = False  # low-level filters stay (little data)
params = [{"params": [p for n, p in model.reid.named_parameters() if p.requires_grad], "lr": 1e-4},
          {"params": model.classifier.parameters(), "lr": 1e-3}]
opt = torch.optim.Adam(params, weight_decay=5e-4)
steps = EPOCHS * max(1, len(train_dl))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")

def triplet(feat, y, margin=0.3):
    d = torch.cdist(feat, feat)
    same = y[:, None] == y[None, :]
    return F.relu((d * same).max(1).values - (d + same * 1e4).min(1).values + margin).mean()

t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    tot = n = 0
    for x, y in train_dl:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE)
        with torch.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
            pooled, bn, logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1) + triplet(pooled.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += float(loss); n += 1
    print(f"epoch {ep + 1}/{EPOCHS}  loss {tot / max(n, 1):.3f}  ({time.time() - t0:.0f} s)")
""")

code(r"""
# 6) Result on the held-out footage, then save the weights for the engine
THR = evaluate(model.reid, "AFTER fine-tuning")
OUT = WORK / "mcvt_reid_finetuned.pth"
state = {k: v.detach().cpu() for k, v in model.reid.state_dict().items()}  # no classifier: engine format
torch.save({"model": state, "suggested_threshold": THR}, OUT)
print("saved", OUT, f"({OUT.stat().st_size / 1e6:.0f} MB) - suggested reid.similarity_threshold", THR)
if os.path.isdir("/content") and not ON_KAGGLE:
    from google.colab import files
    files.download(str(OUT))
elif ON_KAGGLE:
    print("Kaggle: download it from the Output panel (right side) -> mcvt_reid_finetuned.pth")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python",
                                                                     "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = HERE / "colab_finetune_reid.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
