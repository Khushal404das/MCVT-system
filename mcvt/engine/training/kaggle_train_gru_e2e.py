"""End-to-end sequence ReID: the ResNet50-IBN and a GRU are trained TOGETHER on image sequences
(Kaggle GPU script; MCVT_QUICK=1 runs a tiny CPU smoke test).

Why (after kaggle_train_gru.py): a GRU trained on FROZEN features of the fine-tuned ResNet learned
nothing useful - that ResNet had already been trained on the same VeRi-776 / project crops, so in
training every vehicle was already perfectly separated (triplet loss 0 after one minute) and the GRU
was worse than the plain average with 1-2 crops. Here every crop is augmented on every step (far-away
blur, crop/flip, colour, erasing) and passes through the ResNet being trained, so the GRU sees
imperfect features like on new vehicles, and the ResNet learns features that combine well over a track.

Model: ResNet50-IBN (starts from engine/models/mcvt_reid_finetuned.pth) -> per-crop 2048-d feature
       -> GRU (bidirectional) + attention + mean skip -> 512-d track signature.
Loss:  per crop: batch-hard triplet + ID (keeps single-crop ReID good)
       per sequence: batch-hard triplet + ID on the signature. P vehicles x K tracks (other cameras)
       x 1-4 crops per track.
Test:  tracks never trained on (our cameras' test split, VeRi-776 test), cross-camera rank-1 / mAP:
       CURRENT engine (fine-tuned ResNet, average of crops) vs new ResNet + average vs new ResNet + GRU,
       with 1, 2, 4 crops (+ first crop far away).
Output (WORK): mcvt_reid_e2e.pth (new ResNet), mcvt_reid_gru_e2e.pt (GRU + calibration), e2e_metrics.json
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

ON_KAGGLE = os.path.isdir("/kaggle/input")
WORK = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("MCVT_WORK", "e2e_work"))
WORK.mkdir(parents=True, exist_ok=True)
TMP = Path("/kaggle/temp_e2e") if ON_KAGGLE else WORK / "tmp"
TMP.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUICK = bool(os.environ.get("MCVT_QUICK"))
# image-loading processes (Kaggle / Colab GPU machines: 4; MCVT_WORKERS overrides)
WORKERS = int(os.environ.get("MCVT_WORKERS", 4 if ON_KAGGLE else 0))
random.seed(11); np.random.seed(11); torch.manual_seed(11)
ROOTS = ("/kaggle/input",) if ON_KAGGLE else tuple(os.environ.get("MCVT_ROOTS", ".").split(";"))


def find(pattern):
    for r in ROOTS:
        hits = glob.glob(f"{r}/**/{pattern}", recursive=True)
        if hits:
            return hits[0]
    return None


WEIGHTS = os.environ.get("MCVT_WEIGHTS") or find("mcvt_reid_finetuned.pth")
MODEL_PY = os.environ.get("MCVT_MODEL_PY") or find("resnet_ibn.py")
LABELS = find("labels.json")
if not LABELS:
    zipfile.ZipFile(find("mcvt_reid_dataset.zip")).extractall(TMP / "ours")
    LABELS = str(next((TMP / "ours").rglob("labels.json")))
VERI_TRAIN = next(iter(glob.glob(f"{ROOTS[0]}/**/image_train", recursive=True)), None)
VERI_TEST = next(iter(glob.glob(f"{ROOTS[0]}/**/image_test", recursive=True)), None)
print(f"device {DEV} | weights {WEIGHTS} | labels {LABELS} | VeRi {VERI_TRAIN} / {VERI_TEST}", flush=True)
sys.path.insert(0, str(Path(MODEL_PY).parent))
from resnet_ibn import ResNetIBNReID  # noqa: E402

# ------------------------------------------------------------------ tracks
tracklets = []  # {"vid", "cam", "files" (time order), "split", "src"}
lab = json.loads(Path(LABELS).read_text())
root = Path(LABELS).parent
by_tr = defaultdict(list)
for c in lab["crops"]:
    by_tr[(c["camera"], c["tracklet"])].append(c)
keys = list(by_tr)
if QUICK:
    keys = random.sample(keys, min(60, len(keys)))
for k in keys:
    cs = sorted(by_tr[k], key=lambda c: c["t"])
    tracklets.append({"vid": f"ours_{cs[0]['vehicle']}", "cam": k[0], "files": [str(root / c["file"]) for c in cs],
                      "split": cs[0].get("split", "train"), "src": "ours"})


def veri(folder, split, limit=None):
    groups = defaultdict(list)
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    if limit:
        files = files[:limit]
    for f in files:
        p = Path(f).stem.split("_")
        if len(p) >= 3:
            groups[(p[0], p[1])].append((int(p[2]), f))
    for (vid, cam), fs in groups.items():
        tracklets.append({"vid": f"veri_{vid}", "cam": cam, "files": [f for _, f in sorted(fs)], "split": split,
                          "src": "veri"})


if VERI_TRAIN:
    veri(VERI_TRAIN, "train", 2000 if QUICK else None)
if VERI_TEST:
    veri(VERI_TEST, "veri_test", 800 if QUICK else None)
train_tr = [t for t in tracklets if t["split"] == "train"]
ids = sorted({t["vid"] for t in train_tr})
id_of = {v: i for i, v in enumerate(ids)}
by_id = defaultdict(list)
for t in train_tr:
    by_id[t["vid"]].append(t)
multi = [v for v in ids if len(by_id[v]) >= 2]
print(f"{len(tracklets)} tracks; training {len(train_tr)} tracks of {len(ids)} vehicles ({len(multi)} on 2+ cameras)",
      flush=True)

# ------------------------------------------------------------------ data
SIZE = 256


class LowRes:
    """A car far away: shrink to 16-48 px and back (blurry)."""

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.randint(16, 48) / max(w, h)
        if s >= 1:
            return img
        return img.resize((max(4, int(w * s)), max(4, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)


to255 = T.Lambda(lambda t: t * 255.0)
train_tf = T.Compose([LowRes(0.3), T.Resize((SIZE, SIZE)), T.RandomHorizontalFlip(), T.Pad(10), T.RandomCrop(SIZE),
                      T.ColorJitter(0.25, 0.25, 0.2, 0.03), T.ToTensor(), T.RandomErasing(p=0.5, value=0.5), to255])
test_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), to255])
far_tf = T.Compose([LowRes(1.0), T.Resize((SIZE, SIZE)), T.ToTensor(), to255])
P = int(os.environ.get("MCVT_P", 6 if not QUICK else 2))  # 6 x 4 tracks x <=4 crops <= 96 images (T4 16 GB)
K = 4 if not QUICK else 2
MAX_L = 4


class SeqBatches(torch.utils.data.IterableDataset):
    """Endless batches: P vehicles x K tracks (different cameras when possible) x 1-4 crops each."""

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(1000 + (info.id if info else 0) + int(time.time()))
        while True:
            imgs, lengths, labels = [], [], []
            for v in rng.sample(multi, min(P, len(multi))):
                trs = by_id[v]
                pick = rng.sample(trs, K) if len(trs) >= K else [rng.choice(trs) for _ in range(K)]
                for t in pick:
                    n = len(t["files"])
                    L = rng.randint(1, min(MAX_L, n))
                    s = rng.randint(0, n - L)
                    for f in t["files"][s:s + L]:
                        imgs.append(train_tf(Image.open(f).convert("RGB")))
                    lengths.append(L)
                    labels.append(id_of[v])
            yield torch.stack(imgs), torch.tensor(lengths), torch.tensor(labels)


# ------------------------------------------------------------------ model
class SeqReID(nn.Module):
    """Same layers as engine/mcvt/reid/sequence.py."""

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


def crop_feats(reid, images):
    """(pooled feature for the triplet loss, BNNeck feature = what the engine uses)"""
    x = (images - reid.pixel_mean) / reid.pixel_std
    pooled = reid.heads.pool_layer(reid.backbone(x))
    bn = reid.heads.bottleneck(pooled)[..., 0, 0]
    return pooled[..., 0, 0], bn


def to_sequences(feats, lengths):
    X = torch.zeros(len(lengths), int(lengths.max()), feats.shape[1], device=feats.device, dtype=feats.dtype)
    k = 0
    for i, L in enumerate(lengths.tolist()):
        X[i, :L] = feats[k:k + L]
        k += L
    return X


def batch_hard(e, y, margin=0.3):
    d = torch.cdist(e, e)
    same = y[:, None] == y[None, :]
    pos = (d * same.float()).max(1).values
    neg = (d + same.float() * 1e4).min(1).values
    return F.relu(pos - neg + margin).mean()


reid = ResNetIBNReID()
missing, _ = reid.load_fastreid_checkpoint(WEIGHTS)
assert not [m for m in missing if not m.endswith("num_batches_tracked")], missing[:5]
reid.to(DEV)
seq = SeqReID().to(DEV)
clf_crop = nn.Linear(2048, len(ids), bias=False).to(DEV)
nn.init.normal_(clf_crop.weight, std=0.001)
seq_bn = nn.BatchNorm1d(512).to(DEV)
seq_bn.bias.requires_grad_(False)
clf_seq = nn.Linear(512, len(ids), bias=False).to(DEV)
heads = list(seq.parameters()) + list(clf_crop.parameters()) + list(seq_bn.parameters()) + list(clf_seq.parameters())
opt = torch.optim.Adam([{"params": [p for p in reid.parameters() if p.requires_grad], "lr": 3.5e-5},
                        {"params": heads, "lr": 3e-4}], weight_decay=5e-4)
STEPS = 3 if QUICK else int(os.environ.get("MCVT_E2E_STEPS", 4000))
WARM = max(1, STEPS // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
scaler = torch.cuda.amp.GradScaler(enabled=DEV.type == "cuda")
dl = torch.utils.data.DataLoader(SeqBatches(), batch_size=None, num_workers=WORKERS,
                                 persistent_workers=WORKERS > 0, prefetch_factor=4 if WORKERS else None)
t0 = time.time()
reid.train(); seq.train()
for step, (imgs, lengths, labels) in enumerate(dl):
    if step >= STEPS:
        break
    imgs, lengths, labels = imgs.to(DEV, non_blocking=True), lengths.to(DEV), labels.to(DEV)
    crop_labels = labels.repeat_interleave(lengths)
    with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
        pooled, bn = crop_feats(reid, imgs)
        l_crop = batch_hard(F.normalize(pooled.float(), dim=1), crop_labels) + \
            F.cross_entropy(clf_crop(bn).float(), crop_labels, label_smoothing=0.1)
    X = to_sequences(F.normalize(bn.float(), dim=1), lengths)  # GRU in fp32 (cudnn GRU + packing)
    e = seq(X, lengths)
    l_seq = batch_hard(F.normalize(e, dim=1), labels) + F.cross_entropy(clf_seq(seq_bn(e)), labels, label_smoothing=0.1)
    loss = l_crop + l_seq
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 5.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    if step % 200 == 0 or step == STEPS - 1:
        print(f"step {step}/{STEPS}: crop loss {l_crop.item():.3f} | track loss {l_seq.item():.3f} | "
              f"{len(imgs)} crops ({time.time() - t0:.0f} s)", flush=True)

# ------------------------------------------------------------------ evaluation
test_tracks = [t for t in tracklets if (t["src"] == "ours" and t["split"] == "test") or t["split"] == "veri_test"]
test_files = sorted({f for t in test_tracks for f in t["files"]})
ours_first = sorted({t["files"][0] for t in test_tracks if t["src"] == "ours"})


class ImgSet(torch.utils.data.Dataset):
    def __init__(self, files, tf):
        self.files, self.tf = files, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB"))


@torch.inference_mode()
def extract(net, files, tf):
    net.eval()
    out = {}
    loader = torch.utils.data.DataLoader(ImgSet(files, tf), batch_size=128, num_workers=WORKERS)
    k = 0
    for x in loader:
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            f = net(x.to(DEV))
        f = F.normalize(f.float(), dim=1).cpu()
        for row in f:
            out[files[k]] = row
            k += 1
    return out


old = ResNetIBNReID()
old.load_fastreid_checkpoint(WEIGHTS)
old.to(DEV)
FEATS = {"old": extract(old, test_files, test_tf), "new": extract(reid, test_files, test_tf)}
FAR = {"old": extract(old, ours_first, far_tf), "new": extract(reid, ours_first, far_tf)}
del old
seq.eval()


@torch.inference_mode()
def signatures(trs, L, method, far_first):
    net = "old" if method == "current" else "new"
    out = []
    for t in trs:
        fs = t["files"][:L]
        x = torch.stack([FEATS[net][f] for f in fs])
        if far_first:
            x[0] = FAR[net][fs[0]]
        if method == "gru":
            out.append(F.normalize(seq(x[None].to(DEV), torch.tensor([len(fs)], device=DEV))[0], dim=0).cpu())
        else:
            out.append(F.normalize(x.mean(0), dim=0))
    return torch.stack(out)


CAL = {}


def retrieval(name, trs, L, method, far_first=False):
    S = signatures(trs, L, method, far_first)
    sim = (S @ S.T).numpy()
    vids = np.array([t["vid"] for t in trs])
    cams = np.array([t["cam"] for t in trs])
    r1, aps, pos, neg = [], [], [], []
    for i in range(len(trs)):
        g = np.where(cams != cams[i])[0]
        rel = vids[g] == vids[i]
        if not rel.any():
            continue
        s = sim[i, g]
        hits = rel[np.argsort(-s)]
        r1.append(float(hits[0]))
        prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        aps.append(float((prec * hits).sum() / hits.sum()))
        pos += s[rel].tolist()
        neg += s[~rel].tolist()
    CAL[(name, L, method, far_first)] = (pos, neg)
    return {"rank1": float(np.mean(r1)) if r1 else None, "mAP": float(np.mean(aps)) if aps else None,
            "queries": len(r1), "same_median": float(np.median(pos)) if pos else None,
            "diff_p99": float(np.quantile(neg, 0.99)) if neg else None}


metrics = {}
METHODS = ("current", "new_average", "gru")
groups = (("ours", [t for t in test_tracks if t["src"] == "ours"]), ("veri", [t for t in test_tracks if t["src"] == "veri"]))
for name, trs in groups:
    if len(trs) < 4:
        continue
    for L in (1, 2, 4):
        for far_first in ((False, True) if name == "ours" else (False,)):
            key = f"{name} | {L} crop{'s' if L > 1 else ''}{' | first far away' if far_first else ''}"
            res = {m: retrieval(name, trs, L, m, far_first) for m in METHODS}
            metrics[key] = res
            if not res["gru"]["queries"]:
                continue
            print(f"{key}: rank-1 " + " / ".join(f"{m} {res[m]['rank1']:.3f}" for m in METHODS)
                  + " | mAP " + " / ".join(f"{res[m]['mAP']:.3f}" for m in METHODS)
                  + f" ({res['gru']['queries']} queries)", flush=True)

# calibration for the engine: thresholds are set for the CURRENT method (average of the old ResNet)
cal = {}
for m, src in (("average", "current"), ("gru", "gru"), ("new_average", "new_average")):
    if ("ours", 2, src, False) in CAL:
        pos, neg = CAL[("ours", 2, src, False)]
        cal[m] = {"pos": np.round(pos, 4).tolist(), "neg": np.round(neg, 4).tolist()}
torch.save({"model": {k: v.cpu() for k, v in reid.state_dict().items()}}, WORK / "mcvt_reid_e2e.pth")
torch.save({"state_dict": {k: v.cpu() for k, v in seq.state_dict().items()},
            "config": {"d_in": 2048, "d_h": 512, "d_out": 512, "max_len": MAX_L},
            "backbone": "mcvt_reid_e2e.pth", "trained": time.strftime("%Y-%m-%d %H:%M"), "steps": STEPS,
            "calibration": cal}, WORK / "mcvt_reid_gru_e2e.pt")
(WORK / "e2e_metrics.json").write_text(json.dumps(metrics, indent=2))
print("saved", WORK / "mcvt_reid_e2e.pth", WORK / "mcvt_reid_gru_e2e.pt", flush=True)
